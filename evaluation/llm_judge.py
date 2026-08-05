"""LLM-based response evaluator (Appendix A1, Figure A1): pairwise judge over
Specificity, Conciseness, and Completeness, each scored "A" / "B" / "Tie".

Reproduces the paper's judge prompt structure and criteria wording, adapted
to a plain-JSON output schema so it can be parsed reliably from a local
model (the paper's own schema is a JSON object with an enum ["A","B","Tie"]
per metric - see Figure A1).
"""
import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Same family as rag.py / baseline_rag.py's generation model. Swap for a
# stronger local model if your judge scores look noisy - LLM-as-judge
# quality tracks the underlying model's reasoning ability.
JUDGE_MODEL = "llama3.2"

JUDGE_PROMPT = """You are an expert in embedded system customization. You are knowledgeable about general domain knowledge about RTOS, MCU, etc.

-Goal-
Given a question with a reference answer, and two responses written by two people, your goal is to evaluate the responses and select the better one. You should read the criteria carefully and for each metric, decide which response is better, or whether it is a tie.

-Criteria-
- Specificity: Identify which answer is more specific, such as explicitly mentioning the exact functions, commands, or steps required, rather than providing a vague or general description.
- Conciseness: Determine which response directly addresses the question in a clear and focused manner, avoiding unnecessary or unrelated information.
- Completeness: Assess which answer provides a more comprehensive explanation, covering all the key points necessary to fully resolve the question.

-Output Format-
Output valid JSON only, no additional text, in exactly this schema:
{{
  "specificity": "<A|B|Tie>",
  "conciseness": "<A|B|Tie>",
  "completeness": "<A|B|Tie>"
}}

-Question-
{question}

-Reference Answer-
{reference_answer}

-Response A-
{response_a}

-Response B-
{response_b}
"""


def _get_llm():
    return ChatOllama(model=JUDGE_MODEL, temperature=0)


def _parse_verdict(text):
    cleaned = text.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        result = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Local models don't always return clean JSON - fail safe to Tie
        # rather than crashing a full evaluation run over one bad response.
        return {"specificity": "Tie", "conciseness": "Tie", "completeness": "Tie", "raw": text}

    for key in ("specificity", "conciseness", "completeness"):
        value = str(result.get(key, "Tie")).strip()
        result[key] = value if value in ("A", "B", "Tie") else "Tie"
    return result


def judge_pair(question, reference_answer, response_a, response_b, llm=None):
    """One judge call, exactly as the paper's Figure A1 prompt is structured.
    Returns {"specificity": "A"|"B"|"Tie", "conciseness": ..., "completeness": ...}.
    """
    llm = llm or _get_llm()
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
    message = prompt.format(
        question=question,
        reference_answer=reference_answer,
        response_a=response_a,
        response_b=response_b,
    )
    text = llm.invoke(message).content
    return _parse_verdict(text)


def judge_pair_debiased(question, reference_answer, response_a, response_b, llm=None):
    """Runs judge_pair twice with A/B swapped and keeps only the metrics
    that agree in both orderings (unresolved ones fall back to Tie). This
    is NOT part of the paper's method - it's a cheap mitigation for LLM
    judges' well-documented tendency to favor whichever response appears
    first. Use judge_pair directly instead if you want to reproduce the
    paper's evaluation exactly, with its associated position-bias risk.
    """
    llm = llm or _get_llm()
    first = judge_pair(question, reference_answer, response_a, response_b, llm=llm)
    second_raw = judge_pair(question, reference_answer, response_b, response_a, llm=llm)

    flip = {"A": "B", "B": "A", "Tie": "Tie"}
    second = {
        key: flip.get(second_raw.get(key, "Tie"), "Tie")
        for key in ("specificity", "conciseness", "completeness")
    }

    reconciled = {}
    for key in ("specificity", "conciseness", "completeness"):
        reconciled[key] = first.get(key, "Tie") if first.get(key) == second.get(key) else "Tie"
    return reconciled


# ---- Standalone (non-comparative) scoring ----
#
# The paper's judge (above) is inherently pairwise - it exists to produce
# the win-rate numbers in Figure 6, comparing two systems. If you only want
# to evaluate your own pipeline (no VanillaRAG or other second system to
# compare against), there's nothing to be "A" or "B" against. This section
# adapts the same three criteria to an absolute per-question scale instead.

SINGLE_JUDGE_PROMPT = """You are an expert in embedded system customization. You are knowledgeable about general domain knowledge about RTOS, MCU, etc.

-Goal-
Given a question, a reference answer, and a response generated by a RAG system, score the response on each criterion below using a 1-5 scale (1 = very poor, 5 = excellent).

-Criteria-
- Specificity: Does the response mention exact functions, commands, register names, values, or steps, rather than vague or general description?
- Conciseness: Does the response directly address the question without unnecessary or unrelated information?
- Completeness: Does the response cover all the key points needed to fully resolve the question, judged against the reference answer?
- Correctness: Does the response's factual content agree with the reference answer, without contradicting or inventing information?

-Output Format-
Output valid JSON only, no additional text, in exactly this schema:
{{
  "specificity": <1-5>,
  "conciseness": <1-5>,
  "completeness": <1-5>,
  "correctness": <1-5>
}}

-Question-
{question}

-Reference Answer-
{reference_answer}

-Response-
{response}
"""


def _parse_single_verdict(text):
    cleaned = text.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        result = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {"specificity": None, "conciseness": None, "completeness": None, "correctness": None, "raw": text}

    for key in ("specificity", "conciseness", "completeness", "correctness"):
        try:
            value = int(result.get(key))
            result[key] = value if 1 <= value <= 5 else None
        except (TypeError, ValueError):
            result[key] = None
    return result


def score_single(question, reference_answer, response, llm=None):
    """Absolute 1-5 scoring of a single pipeline's response - no second
    system needed. Returns
    {"specificity": int|None, "conciseness": int|None, "completeness": int|None,
     "correctness": int|None}. None indicates the judge's output couldn't be
    parsed for that question (excluded from averages rather than treated as 0).
    """
    llm = llm or _get_llm()
    prompt = ChatPromptTemplate.from_template(SINGLE_JUDGE_PROMPT)
    message = prompt.format(question=question, reference_answer=reference_answer, response=response)
    text = llm.invoke(message).content
    return _parse_single_verdict(text)