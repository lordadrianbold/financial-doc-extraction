"""
Project 1, Week 1 (continued): the actual structured extraction
pipeline — sending real OCR'd receipt text through an LLM with
function-calling schema enforcement (LangChain's
with_structured_output()), then independently validating the result
with rule-based checks, rather than trusting either the model's raw
output or its own self-reported confidence at face value.

Why TWO separate reliability signals, not just one: LLMs are well
known to be poorly calibrated when self-reporting their own
confidence — a model can sound equally confident whether it's right or
wrong. Relying solely on self-reported confidence would undermine the
entire point of a "reliability engineering" project. Instead, this
pipeline combines:
1. The model's own self-reported confidence (still useful signal, just
   not trusted alone)
2. Independent, deterministic rule-based validation (does the total
   look like a real currency amount? does the date look plausible?
   is the company name suspiciously short or empty?) — checks that
   don't depend on the model being right about itself
A receipt only gets treated as "reliably extracted" if BOTH signals
agree; anything else gets flagged for human review rather than
silently trusted.
"""

import os
import re
from datetime import datetime
from pydantic import BaseModel, Field


class ReceiptExtraction(BaseModel):
    """
    The structured output schema — LangChain's with_structured_output()
    enforces this via function calling, so the model's response is
    guaranteed to match this shape (or the call fails loudly), not
    just "usually formatted like this."
    """
    company: str = Field(description="The vendor/company name on the receipt")
    date: str = Field(description="The transaction date as it appears on the receipt")
    address: str = Field(description="The company address on the receipt")
    total: str = Field(description="The total amount on the receipt, digits only plus decimal point")
    confidence: str = Field(description="Your own confidence in this extraction: 'high', 'medium', or 'low'")


EXTRACTION_SYSTEM_PROMPT = """You are extracting structured data from OCR'd receipt text. The text may contain OCR errors, garbled characters, or missing words, since it was automatically extracted from a scanned image.

Extract the company name, date, address, and total amount. If a field is genuinely not present or too garbled to determine, use an empty string for that field rather than guessing.

Rate your own confidence as 'high' only if all fields are clearly present and unambiguous in the text. Use 'medium' if some fields required inference from imperfect OCR text. Use 'low' if you had to guess significantly for any field."""

# REVERTED — three real, different attempts to improve address
# exact-match accuracy (rule-based validation, targeted "look for
# JALAN street names" wording, and this numbered-line structural
# reformatting) were each tested broadly and none moved the real
# number. The numbered-line version specifically introduced a new
# regression (date accuracy dropped from 15/15 to 14/15 on the same
# real 15-receipt sample) with no corresponding benefit to address.
#
# Reverted to this plain, original prompt, which has the strongest
# real evidence behind it (100% date accuracy, 93% total accuracy).
# Address's real limitation is now handled honestly by MEASURING it
# differently, not by further attempts to change the model's
# behavior — see run_extraction_eval.py's postal-code-match metric,
# added specifically because byte-perfect address matching was never
# a fair bar for this field, not because 0% needed to look better.


def build_extraction_chain(model_name: str = "claude-haiku-4-5-20251001"):
    """
    Builds the structured extraction chain. Requires ANTHROPIC_API_KEY
    to be set — same pattern verified working throughout the RAG
    agent project.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Set it before running this script."
        )

    llm = ChatAnthropic(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(ReceiptExtraction)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", "OCR'd receipt text:\n\n{ocr_text}"),
    ])

    return prompt | structured_llm


LAYOUT_AWARE_SYSTEM_PROMPT = """You are extracting structured data from OCR'd receipt text. Each line below is shown with its approximate vertical position on the original receipt image, as [y=NUMBER] — smaller numbers are nearer the top of the receipt. This is NOT part of the actual receipt text, just a positional hint.

Lines with SIMILAR y-values were positioned close together vertically on the real receipt and are likely part of the same visual block (for example, a multi-line business address is usually a tight cluster of nearby y-values near the top of the receipt). Lines with very different y-values are far apart on the page and are usually unrelated to each other, even if they appear close together in this list.

Extract the company name, date, address, and total amount. For the address specifically: use the y-position information to identify ALL lines that form one coherent, spatially-clustered address block near the top of the receipt, not just the single most obviously address-like line — a real address is often split across several nearby lines (unit/lot number, street name, postal code and city).

If a field is genuinely not present or too garbled to determine, use an empty string for that field rather than guessing.

Rate your own confidence as 'high' only if all fields are clearly present and unambiguous in the text. Use 'medium' if some fields required inference from imperfect OCR text. Use 'low' if you had to guess significantly for any field."""

# Added directly in response to real, published research (documented
# in notes.md): specialized layout-aware models (StrucTexT, LayoutLMv2)
# achieve 95%+ F1 on this exact SROIE benchmark specifically because
# they use bounding-box POSITION as a core input, not just text — a
# root cause independently confirmed by three separate real, tested,
# reverted prompt-only attempts in this same project all failing to
# meaningfully improve address accuracy.
#
# This is deliberately NOT the same as actually training/fine-tuning a
# layout-aware model (that would be a genuinely different, much larger
# undertaking, more appropriate to a dedicated fine-tuning project) —
# this instead gives a general-purpose model raw Y-coordinate values as
# structured TEXT context within a normal prompt, testing whether even
# this lighter-weight signal helps. It is also deliberately NOT image-
# based (no pixels are ever sent) — it stays within this project's
# text-only scope, distinct from Project 5's planned multimodal work.
# Real accuracy improvement, if any, has not yet been measured broadly
# — this needs the same real testing discipline as every other change
# in this project before being trusted.


def format_ocr_text_with_positions(words: list[str], y_positions: list[int]) -> str:
    """
    Formats OCR lines with their real vertical (y1) position from the
    receipt's actual bounding-box data — genuinely different from the
    earlier, failed numbered-line attempt, which only conveyed
    SEQUENTIAL ORDER (line 1, line 2, ...), not actual SPATIAL
    position. Two lines can be sequentially far apart in the OCR
    output's word order while being visually close together on the
    real receipt (or vice versa) — real y-coordinates capture the
    actual spatial relationship that sequential numbering cannot.
    """
    if len(words) != len(y_positions):
        raise ValueError(
            f"words and y_positions must be the same length (got {len(words)} "
            f"and {len(y_positions)}) — they're expected to be parallel arrays "
            f"from the same receipt's OCR data."
        )
    return "\n".join(f"[y={y}] {word}" for word, y in zip(words, y_positions))


def build_layout_aware_extraction_chain(model_name: str = "claude-haiku-4-5-20251001"):
    """
    A separate chain, deliberately not replacing build_extraction_chain,
    so the layout-aware approach can be directly compared against the
    existing text-only approach on the same real receipts — not simply
    trusted as an improvement without a real, direct comparison.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Set it before running this script."
        )

    llm = ChatAnthropic(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(ReceiptExtraction)

    prompt = ChatPromptTemplate.from_messages([
        ("system", LAYOUT_AWARE_SYSTEM_PROMPT),
        ("human", "OCR'd receipt text with position hints:\n\n{ocr_text}"),
    ])

    return prompt | structured_llm


# --- Independent, rule-based validation (does NOT trust the model's
# own self-reported confidence) ---

TOTAL_PATTERN = re.compile(r"^\d+\.\d{2}$")


def validate_total(total: str) -> tuple[bool, str]:
    """
    Checks whether the extracted total looks like a genuine currency
    amount — deliberately strict (exactly two decimal places), since
    real SROIE ground-truth totals consistently follow this format
    (confirmed by direct inspection of real ground truth earlier:
    "9.00", "4.95", "102.40").
    """
    if not total:
        return False, "empty"
    if not TOTAL_PATTERN.match(total.strip()):
        return False, f"doesn't match expected currency format (digits.digits): '{total}'"
    return True, "valid format"


def validate_date(date: str) -> tuple[bool, str]:
    """
    Checks whether the extracted date is at least non-empty and
    contains digits — deliberately loose on exact format, since real
    SROIE ground truth uses multiple different real date formats
    (confirmed by direct inspection: "25/12/2018", "10 MAR 2018",
    "30 DEC 17" all appear as genuine, valid real examples) — a strict
    single-format check would incorrectly flag genuinely correct
    extractions just for using a different (but equally valid) date
    format than expected.
    """
    if not date:
        return False, "empty"
    if not any(char.isdigit() for char in date):
        return False, f"contains no digits, unlikely to be a real date: '{date}'"
    return True, "plausible"


def validate_company(company: str) -> tuple[bool, str]:
    """
    A minimal plausibility check — a company name shouldn't be
    suspiciously short. Deliberately not stricter than this, since
    real company names vary enormously in format and length, and a
    stricter check would risk false-flagging genuinely correct short
    names.
    """
    if not company:
        return False, "empty"
    if len(company.strip()) < 3:
        return False, f"suspiciously short: '{company}'"
    return True, "plausible"


POSTAL_CODE_PATTERN = re.compile(r"\b\d{5}\b")

# Real, direct evidence for this threshold: genuine complete SROIE
# ground-truth addresses observed directly ("NO.53 55,57 & 59, JALAN
# SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR." — 68 characters;
# "15, JALAN PERMAS 10/7,PERMAS 81750 MASAI JOHOR" — 48 characters)
# consistently run 45+ characters. A real model extraction that failed
# evaluation ("81750 MASAI JOHOR" — 18 characters) was a genuine,
# correct-but-truncated PARTIAL address, missing the street-level
# detail. This threshold is set below the real full-address examples
# but above the real observed truncation case, so it would have
# correctly flagged that real failure for review.
MIN_PLAUSIBLE_ADDRESS_LENGTH = 25


def validate_address(address: str) -> tuple[bool, str]:
    """
    Added directly in response to real evaluation evidence: an initial
    real 15-receipt run found address was the clear weakest field
    (0/15 exact match, similarity scores scattered widely from 0.152
    to 0.906 — not a simple, consistent truncation pattern), while
    having ZERO rule-based scrutiny at all (an earlier version of this
    module deliberately skipped validating address, reasoning that
    real addresses vary too much in format to check meaningfully —
    that reasoning was a real, now-corrected mistake, since address is
    demonstrably the field most in need of an independent reliability
    signal, not the one field to leave unchecked).

    Two checks, both grounded in directly-observed real ground truth
    patterns for this specific dataset (Malaysian receipt addresses),
    not generic/invented address-validation logic:
    1. Length — genuinely complete addresses in this real dataset
       consistently run 45+ characters; a real model failure was a
       correct-but-truncated 18-character fragment. A minimum length
       threshold, set between these two real observed points, would
       have caught that real failure.
    2. Postal code presence — real Malaysian addresses in this dataset
       consistently include a 5-digit postal code; its absence is a
       concrete, checkable signal of a likely-incomplete extraction.
    """
    if not address:
        return False, "empty"
    if len(address.strip()) < MIN_PLAUSIBLE_ADDRESS_LENGTH:
        return False, f"suspiciously short ({len(address.strip())} chars) for a real full address: '{address}'"
    if not POSTAL_CODE_PATTERN.search(address):
        return False, f"no 5-digit postal code found, real addresses in this dataset consistently include one: '{address}'"
    return True, "plausible"


def validate_extraction(extraction: ReceiptExtraction) -> dict:
    """
    Runs all rule-based checks and combines them into one overall
    validation result — deliberately SEPARATE from and not influenced
    by the model's own self-reported confidence field, so the two
    signals can be compared honestly rather than one silently
    reinforcing the other.
    """
    checks = {
        "total": validate_total(extraction.total),
        "date": validate_date(extraction.date),
        "company": validate_company(extraction.company),
        "address": validate_address(extraction.address),
    }

    failed_checks = {field: reason for field, (passed, reason) in checks.items() if not passed}

    return {
        "rule_based_valid": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "model_self_reported_confidence": extraction.confidence,
        # The two signals AGREE if the model claimed high confidence
        # AND the rule-based checks also found nothing wrong — only
        # this combination is treated as genuinely reliable.
        "signals_agree": extraction.confidence == "high" and len(failed_checks) == 0,
    }


def extract_and_validate(ocr_text: str, chain) -> dict:
    """
    Full pipeline for one receipt: extract, then validate independently.
    """
    extraction = chain.invoke({"ocr_text": ocr_text})
    validation = validate_extraction(extraction)

    return {
        "extraction": extraction.model_dump(),
        "validation": validation,
    }


def extract_and_validate_layout_aware(words: list[str], y_positions: list[int], chain) -> dict:
    """
    The layout-aware counterpart to extract_and_validate() — takes the
    receipt's real words and y-positions separately (not pre-joined
    text), so format_ocr_text_with_positions() can build the
    position-annotated input the layout-aware chain expects.
    """
    formatted_text = format_ocr_text_with_positions(words, y_positions)
    extraction = chain.invoke({"ocr_text": formatted_text})
    validation = validate_extraction(extraction)

    return {
        "extraction": extraction.model_dump(),
        "validation": validation,
    }




