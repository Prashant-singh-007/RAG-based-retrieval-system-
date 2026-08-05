"""End-to-end evaluation runner - reproduces the paper's two evaluation
tracks (Section 4.3) against dataset.csv (columns: question, reference_answer):

  1. Information Retrieval: Recall@5/7/10/20 (proxy - see recall_eval.py's
     module docstring for why this can't be the paper's exact ground-truth
     metric given dataset.csv has no ground-truth chunk labels), reported
     for both your HSG-style pipeline and a VanillaRAG baseline
     (Table 4-style output).
  2. Response Generation quality: LLM-judge win-rate over Specificity /
     Conciseness / Completeness, HSG-style pipeline (A) vs VanillaRAG
     baseline (B) (Figure 6-style output).

Setup (run once):
    python baseline_build_db.py        # builds the VanillaRAG comparison index

Usage:
    python run_evaluation.py --dataset dataset.csv --n 50
    python run_evaluation.py --dataset dataset.csv --recall-mode llm --no-baseline
"""
import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from rag import RAGPipeline
from baseline_rag import VanillaRAGPipeline
from llm_judge import judge_pair_debiased
from recall_eval import lexical_recall_at_k, llm_groundedness_at_k

K_VALUES = [5, 7, 10, 20]


def load_dataset(path, limit=None):
    """Accepts either a 'reference_answer' column or a plain 'answer' column
    (your dataset.csv uses id, category, question, answer) - normalized to
    'question' / 'reference_answer' keys internally either way. 'category'
    is carried through when present so results can be broken down per
    category (Specifications, Pinout, General, AbsMax, Registers, Theory,
    Timing, Applications, etc.)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if "question" not in raw:
                raise ValueError(f"dataset.csv must have a 'question' column; got {list(raw.keys())}")
            answer = raw.get("reference_answer") or raw.get("answer")
            if answer is None:
                raise ValueError(
                    f"dataset.csv must have a 'reference_answer' or 'answer' column; got {list(raw.keys())}"
                )
            rows.append({
                "id": raw.get("id", ""),
                "category": raw.get("category", ""),
                "question": raw["question"],
                "reference_answer": answer,
            })
            if limit and len(rows) >= limit:
                break
    return rows


def docs_by_k(retrieve_fn, query, k_values=K_VALUES):
    """Retrieve once at the largest k, then slice down for smaller k -
    avoids re-querying the vector store/graph once per k value."""
    max_k = max(k_values)
    docs = retrieve_fn(query, max_k)
    return {k: docs[:k] for k in k_values}


def run(dataset_path, n=None, recall_mode="lexical", use_baseline=True, out_dir="eval_output"):
    os.makedirs(out_dir, exist_ok=True)
    rows = load_dataset(dataset_path, limit=n)
    print(f"Loaded {len(rows)} questions from {dataset_path}.")

    hsg = RAGPipeline()
    vanilla = VanillaRAGPipeline() if use_baseline else None
    recall_fn = lexical_recall_at_k if recall_mode == "lexical" else llm_groundedness_at_k

    recall_totals_hsg = {k: 0.0 for k in K_VALUES}
    recall_totals_vanilla = {k: 0.0 for k in K_VALUES}
    judge_wins = {
        "specificity": {"A": 0, "B": 0, "Tie": 0},
        "conciseness": {"A": 0, "B": 0, "Tie": 0},
        "completeness": {"A": 0, "B": 0, "Tie": 0},
    }
    per_question_results = []

    for i, row in enumerate(rows):
        question = row["question"]
        reference_answer = row["reference_answer"]
        print(f"\n[{i + 1}/{len(rows)}] {question}")

        hsg_answer, _ = hsg.answer(question)
        hsg_docs_by_k = docs_by_k(lambda q, k: hsg.retrieve(q, k=k), question)
        hsg_recall = recall_fn(reference_answer, hsg_docs_by_k)
        for k in K_VALUES:
            recall_totals_hsg[k] += hsg_recall[k]

        result = {
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "question": question,
            "reference_answer": reference_answer,
            "hsg_answer": hsg_answer,
            "hsg_recall": hsg_recall,
        }

        if use_baseline:
            vanilla_answer, _ = vanilla.answer(question)
            vanilla_docs_by_k = docs_by_k(lambda q, k: vanilla.retrieve(q, k=k), question)
            vanilla_recall = recall_fn(reference_answer, vanilla_docs_by_k)
            for k in K_VALUES:
                recall_totals_vanilla[k] += vanilla_recall[k]

            verdict = judge_pair_debiased(question, reference_answer, hsg_answer, vanilla_answer)
            for metric, winner in verdict.items():
                judge_wins[metric][winner] += 1

            result.update({
                "vanilla_answer": vanilla_answer,
                "vanilla_recall": vanilla_recall,
                "judge_verdict": verdict,
            })

        per_question_results.append(result)

    n_q = len(rows)

    print("\n" + "=" * 60)
    print(f"RETRIEVAL: Recall@k (proxy mode: {recall_mode})")
    print("=" * 60)
    header = f"{'Setting':<12}" + "".join(f"R@{k:<8}" for k in K_VALUES)
    print(header)
    print(f"{'HSG-style':<12}" + "".join(f"{recall_totals_hsg[k] / n_q * 100:<10.1f}" for k in K_VALUES))
    if use_baseline:
        print(f"{'VanillaRAG':<12}" + "".join(f"{recall_totals_vanilla[k] / n_q * 100:<10.1f}" for k in K_VALUES))

    if use_baseline:
        print("\n" + "=" * 60)
        print("RESPONSE QUALITY: Win rate, HSG-style (A) vs VanillaRAG (B)")
        print("=" * 60)
        for metric, counts in judge_wins.items():
            total = sum(counts.values()) or 1
            a_pct = counts["A"] / total * 100
            b_pct = counts["B"] / total * 100
            tie_pct = counts["Tie"] / total * 100
            print(f"{metric.capitalize():<14} A(HSG): {a_pct:5.1f}%   B(Vanilla): {b_pct:5.1f}%   Tie: {tie_pct:5.1f}%")

    categories = sorted({r["category"] for r in per_question_results if r.get("category")})
    if categories:
        print("\n" + "=" * 60)
        print("RETRIEVAL: Recall@10 by category (HSG-style)")
        print("=" * 60)
        for cat in categories:
            cat_rows = [r for r in per_question_results if r["category"] == cat]
            avg = sum(r["hsg_recall"][10] for r in cat_rows) / len(cat_rows)
            print(f"  {cat:<16} n={len(cat_rows):<4} R@10={avg * 100:5.1f}%")

    out_path = os.path.join(out_dir, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_questions": n_q,
                "recall_mode": recall_mode,
                "recall_hsg": {k: recall_totals_hsg[k] / n_q for k in K_VALUES},
                "recall_vanilla": ({k: recall_totals_vanilla[k] / n_q for k in K_VALUES} if use_baseline else None),
                "judge_wins": judge_wins if use_baseline else None,
                "per_question": per_question_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nFull per-question results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.csv")
    parser.add_argument("--n", type=int, default=None, help="Limit number of questions (omit for full dataset)")
    parser.add_argument("--recall-mode", choices=["lexical", "llm"], default="lexical")
    parser.add_argument("--no-baseline", action="store_true", help="Skip VanillaRAG comparison, score HSG pipeline alone")
    parser.add_argument("--out-dir", default="eval_output")
    args = parser.parse_args()

    run(
        dataset_path=args.dataset,
        n=args.n,
        recall_mode=args.recall_mode,
        use_baseline=not args.no_baseline,
        out_dir=args.out_dir,
    )
