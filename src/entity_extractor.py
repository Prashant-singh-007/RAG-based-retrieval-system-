"""Entity and relation extraction for GraphRAG.

Uses spaCy (en_core_web_sm) for fast, deterministic named-entity recognition,
plus a couple of regex passes for domain entities spaCy's default model
doesn't know about (electronic part numbers, voltage figures). This is the
"spaCy" arm of "extract entities and relationships using an LLM or spaCy" -
chosen over an LLM extractor so graph construction doesn't cost one model
call per chunk, which would dominate build_db.py's runtime on a real corpus.

Relations are extracted per-sentence: any two recognized entities that
co-occur in the same sentence are linked. This is intentionally simple
(no dependency-parse relation typing) but is enough to connect entities
that graph_retriever.py can then traverse between.
"""

import re

import spacy

_NLP = None

# spaCy's default entity types that aren't useful for electronics datasheets
# (dates, percentages, generic cardinals, etc.) - filtered out so the graph
# doesn't fill up with noise nodes.
IGNORED_SPACY_LABELS = {
    "DATE", "TIME", "PERCENT", "CARDINAL", "ORDINAL", "MONEY", "QUANTITY",
}

# Electronic part numbers: sequences like "TPS54331", "ATmega328P", "LM358".
PART_NUMBER_RE = re.compile(r"\b[A-Z]{2,}[0-9]{2,}[A-Za-z0-9\-]*\b")


def get_nlp():
    """Lazily load the spaCy model (loading it is slow, so do it once)."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. Run:\n"
                "  python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


def _dedup_key(name):
    return name.strip().lower()


def extract_entities_relations(text):
    """Extract {'entities': [...], 'relations': [...]} from a chunk (or
    query) of text.

    entities: [{"name": str, "type": str}, ...] - deduplicated by
    case-insensitive name within this text.
    relations: [{"source": str, "target": str, "relation": "CO_OCCURS_WITH"}]
    for every pair of recognized entities that appear in the same sentence.
    """
    nlp = get_nlp()
    # Bound spaCy's per-call cost - chunks are already ~1000 chars, this is
    # just a safety cap in case of unusually long input (e.g. a full-text
    # datasheet passed through compare_datasheet's summarizer).
    doc = nlp(text[:5000])

    entities = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ in IGNORED_SPACY_LABELS:
            continue
        key = _dedup_key(ent.text)
        if key and key not in seen:
            seen.add(key)
            entities.append({"name": ent.text.strip(), "type": ent.label_})

    for match in PART_NUMBER_RE.finditer(doc.text):
        name = match.group(0)
        key = _dedup_key(name)
        if key not in seen:
            seen.add(key)
            entities.append({"name": name, "type": "PART_NUMBER"})

    # Per-sentence co-occurrence relations. Rebuild the entity name list per
    # sentence (spaCy ents + regex part numbers that fall inside that span).
    relations = []
    for sent in doc.sents:
        sent_names = []
        for ent in doc.ents:
            if ent.label_ in IGNORED_SPACY_LABELS:
                continue
            if ent.start >= sent.start and ent.end <= sent.end:
                sent_names.append(ent.text.strip())
        for match in PART_NUMBER_RE.finditer(sent.text):
            sent_names.append(match.group(0))

        # dedupe within the sentence, preserve first-seen casing
        unique_names = []
        seen_in_sent = set()
        for name in sent_names:
            key = _dedup_key(name)
            if key not in seen_in_sent:
                seen_in_sent.add(key)
                unique_names.append(name)

        for i in range(len(unique_names)):
            for j in range(i + 1, len(unique_names)):
                relations.append({
                    "source": unique_names[i],
                    "target": unique_names[j],
                    "relation": "CO_OCCURS_WITH",
                })

    return {"entities": entities, "relations": relations}
