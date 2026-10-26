"""Model A (Section 3.2): summarizing LLM used in the on-demand inference
pipeline. It converts the full text of an input datasheet into a compact
semantic summary; the embedding of that summary becomes the query vector
used for summary-driven similarity retrieval (Figure 2).

The paper's headline real-data result uses Claude Sonnet 4.5 as Model A; this
implementation uses a local model (llama3.2 via Ollama) so the pipeline can
run fully offline, matching the paper's local-model comparison arm
(Section 5.3 / Table 4) rather than the Claude-based configuration.
"""

import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from image_utils import build_multimodal_content

# Vision-capable so Model A can jointly analyze "Text and Figures" as
# described in Figure 2, rather than only seeing [IMAGE_REF_n] placeholders.
SUMMARIZER_MODEL = "llama3.2"

# Adapted from Appendix B.1, generalized from MCU-only fields to any
# electronic component family (ADC, OPAMP, VREG, sensor, transistor, etc.).
STRUCTURED_SUMMARY_PROMPT = """
You are an expert electronics engineer analyzing a component datasheet.
Generate a structured JSON summary of the component with the following schema:

{{
  "Device": "<full device/part name>",
  "Category": "<component family, e.g. MCU, ADC, OPAMP, VREG, sensor, transistor>",
  "Manufacturer": "<manufacturer name if stated>",
  "Architecture": "<architecture/type if applicable>",
  "Key_Specifications": ["<critical numeric specs with units>"],
  "Operating_Voltage": "<voltage range>",
  "Temperature_Range": "<temperature range>",
  "Power_Consumption": "<active/sleep figures if stated>",
  "Interfaces": ["<communication or I/O interfaces>"],
  "Special_Features": ["<distinguishing features>"],
  "Package_Options": ["<available packages>"],
  "Typical_Applications": ["<typical use cases>"]
}}

Rules:
- Output valid JSON only, no additional text.
- Copy values exactly as stated in the datasheet where possible.
- Omit fields that do not apply; do not invent values.
- Treat any [IMAGE_REF_n] marker as evidence of an associated diagram or
  schematic; if the surrounding text implies its subject (e.g. pinout,
  block diagram), mention it briefly in Special_Features.

Datasheet content:
{datasheet_text}
"""

# Adapted from Appendix B.2.
DETAILED_SUMMARY_PROMPT = """
You are a senior electronics engineer writing a comprehensive technical
summary of a component for engineering colleagues. Write a detailed summary
(250-400 words) covering architecture/type, key specifications, interfaces,
distinguishing features, and ideal application domains. Write in technical
prose, not JSON or bullet points.

Datasheet content:
{datasheet_text}
"""


def _get_llm():
    return ChatOllama(model=SUMMARIZER_MODEL, temperature=0)


def _invoke(llm, prompt_text, image_paths=None):
    """Invoke the LLM with plain text, or with text+images (multimodal
    HumanMessage) when figures were extracted for this datasheet."""
    if image_paths:
        content = build_multimodal_content(prompt_text, image_paths)
        try:
            response = llm.invoke([HumanMessage(content=content)])
            return response.content
        except Exception as e:
            # Text-only model: retry without images instead of failing.
            if "multimodal" not in str(e).lower():
                raise
            print("Model does not support images - falling back to text-only.")
    response = llm.invoke(prompt_text)
    return response.content


def generate_structured_summary(datasheet_text, llm=None, image_paths=None):
    """Structured (JSON) summary strategy, per Appendix B.1. This is the
    summary format used by the paper's best-performing real-data
    configuration (Structured + Rerank).

    image_paths: optional list of figure images extracted from the input
    datasheet (pinouts, block diagrams, etc.) - when provided, Model A
    analyzes them jointly with the text, per Figure 2.
    """
    llm = llm or _get_llm()
    prompt = ChatPromptTemplate.from_template(STRUCTURED_SUMMARY_PROMPT)
    message = prompt.format(datasheet_text=datasheet_text)
    text = _invoke(llm, message, image_paths).strip()

    # Local models don't always return clean JSON; best-effort parse with a
    # graceful fallback so the pipeline doesn't hard-fail on malformed output.
    cleaned = text.strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {"raw_summary": text}


def generate_detailed_summary(datasheet_text, llm=None, image_paths=None):
    """Narrative-form summary strategy, per Appendix B.2."""
    llm = llm or _get_llm()
    prompt = ChatPromptTemplate.from_template(DETAILED_SUMMARY_PROMPT)
    message = prompt.format(datasheet_text=datasheet_text)
    return _invoke(llm, message, image_paths).strip()


def summary_to_text(summary):
    """Flatten a structured summary (dict) or narrative summary (str) into the
    text that gets embedded to form the query vector."""
    if isinstance(summary, str):
        return summary
    return json.dumps(summary, ensure_ascii=False)