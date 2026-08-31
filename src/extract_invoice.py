"""
Project 1, second half: structured extraction for the richer synthetic
B2B invoice structure — a nested list of line items, not a single flat
field like the receipt pipeline's 4 simple fields.

The key reliability check here is genuinely different from the receipt
pipeline's rule-based checks (currency format, date plausibility):
RECONCILIATION. Because every synthetic invoice's totals are exactly
derivable from its own line items by construction (verified in
generate_synthetic_invoices.py), a correct extraction's line items
should sum to its extracted subtotal, and subtotal + tax should equal
the extracted total — a powerful, genuinely different validation
signal unavailable for SROIE's receipts (which don't expose enough
structure to check this way). A real math mismatch is strong evidence
of an extraction error, independent of the model's own self-reported
confidence, in the same spirit as (but a meaningfully stronger check
than) the receipt pipeline's format-only validators.
"""

import os
import re
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(description="The item or service description")
    gl_code: str = Field(description="The general ledger code shown for this line item, if present")
    quantity: int = Field(description="The quantity for this line item")
    unit_price: float = Field(description="The unit price for this line item")
    line_total: float = Field(description="The line total (quantity times unit price) for this line item")


class InvoiceExtraction(BaseModel):
    """
    A deliberately richer schema than the receipt pipeline's
    ReceiptExtraction — a nested list of structured line items, not a
    single flat field, testing whether LangChain's structured-output
    function calling correctly handles this added complexity (verified
    structurally before writing this module).

    discount_amount added directly in response to a real, confirmed
    finding: a genuinely correct extraction of a discounted invoice was
    being flagged as unreliable by validate_reconciliation(), because
    the schema gave the model no way to report a discount separately
    from the subtotal — the reconciliation check could only compare
    the raw line-item sum against subtotal directly, which will always
    legitimately differ by exactly the discount amount whenever one is
    present, regardless of how correct the extraction actually is.
    """
    vendor: str = Field(description="The vendor/company name issuing the invoice")
    invoice_number: str = Field(description="The invoice number")
    invoice_date: str = Field(description="The invoice date")
    due_date: str = Field(description="The payment due date")
    po_number: str = Field(description="The purchase order number referenced on the invoice")
    line_items: list[LineItem] = Field(description="Every line item listed on the invoice")
    discount_amount: float = Field(default=0.0, description="Any discount amount shown on the invoice, as a positive number. Use 0 if no discount is present.")
    subtotal: float = Field(description="The subtotal before tax (after any discount is applied)")
    tax_amount: float = Field(description="The tax amount")
    total: float = Field(description="The final total amount due")
    confidence: str = Field(description="Your own confidence in this extraction: 'high', 'medium', or 'low'")


INVOICE_EXTRACTION_SYSTEM_PROMPT = """You are extracting structured data from a B2B invoice's text. Extract the vendor name, invoice number, invoice date, due date, PO number, every line item (with its description, GL code if shown, quantity, unit price, and line total), any discount amount shown, the subtotal, tax amount, and final total.

If the invoice shows a discount line (e.g. "Volume Discount" or "Early Payment Discount"), extract that amount separately as discount_amount — the subtotal should be the amount AFTER the discount is applied, not the raw sum of line items. If no discount is shown, use 0 for discount_amount.

If a field is genuinely not present, use an empty string (or 0 for numeric fields) rather than guessing.

Rate your own confidence as 'high' only if all fields including every line item are clearly present and unambiguous. Use 'medium' if some fields required inference. Use 'low' if you had to guess significantly for any field."""


def build_invoice_extraction_chain(model_name: str = "claude-haiku-4-5-20251001"):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Set it before running this script."
        )

    llm = ChatAnthropic(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(InvoiceExtraction)

    prompt = ChatPromptTemplate.from_messages([
        ("system", INVOICE_EXTRACTION_SYSTEM_PROMPT),
        ("human", "Invoice text:\n\n{invoice_text}"),
    ])

    return prompt | structured_llm


def validate_reconciliation(extraction: InvoiceExtraction) -> dict:
    """
    The core reliability check for invoices — genuinely different from
    (and stronger than) the receipt pipeline's format-only validators.
    Because a correctly-extracted invoice's own numbers should be
    internally consistent (line items sum to subtotal; subtotal + tax
    = total), checking this directly catches real extraction errors —
    a wrong quantity, a misread unit price, a dropped line item — that
    a format check alone (e.g. "is this a valid-looking number") could
    never catch, since a wrong value can still be a perfectly
    well-formatted one.
    """
    calculated_raw_line_item_sum = round(sum(item.line_total for item in extraction.line_items), 2)
    # Subtract the extracted discount before comparing against
    # subtotal — this is the actual fix: previously this compared the
    # raw line-item sum DIRECTLY against subtotal with no way to
    # account for a legitimate discount, which meant every correctly-
    # extracted discounted invoice failed reconciliation purely because
    # the schema gave the model nowhere to report the discount
    # separately (confirmed by a real, direct example: a perfectly
    # correct extraction showing a $86.13 discrepancy that was exactly
    # equal to the invoice's real discount amount).
    calculated_subtotal_after_discount = round(calculated_raw_line_item_sum - extraction.discount_amount, 2)
    calculated_total = round(extraction.subtotal + extraction.tax_amount, 2)

    # A small tolerance for legitimate floating-point rounding, not for
    # genuine discrepancies — deliberately tight, since invoice math
    # should be exact, not approximately close.
    #
    # 0.015 (not exactly 0.01) confirmed necessary by direct testing:
    # a genuine 1-cent difference (e.g. $99.99 vs $100.00) does NOT
    # compute to exactly 0.01 in IEEE 754 floating point — Python
    # actually computes abs(100.0 - 99.99) as 0.010000000000005116,
    # a real binary floating-point representation artifact (0.01 and
    # 0.99 can't be represented exactly in binary, the same class of
    # issue as 1/3 not being exact in decimal). A tolerance of exactly
    # 0.01 would incorrectly reject genuine 1-cent rounding due to this
    # tiny excess, while 0.015 reliably tolerates real 1-cent rounding
    # with margin to spare, and is still far too tight to let any
    # genuine multi-cent-or-larger extraction error pass undetected.
    subtotal_reconciles = abs(calculated_subtotal_after_discount - extraction.subtotal) <= 0.015
    total_reconciles = abs(calculated_total - extraction.total) <= 0.015

    return {
        "subtotal_reconciles": subtotal_reconciles,
        "total_reconciles": total_reconciles,
        "fully_reconciles": subtotal_reconciles and total_reconciles,
        "calculated_raw_line_item_sum": calculated_raw_line_item_sum,
        "extracted_discount_amount": extraction.discount_amount,
        "calculated_subtotal_after_discount": calculated_subtotal_after_discount,
        "extracted_subtotal": extraction.subtotal,
        "calculated_total_from_subtotal_plus_tax": calculated_total,
        "extracted_total": extraction.total,
    }


def validate_line_item_fields(extraction: InvoiceExtraction) -> dict:
    """
    A second, INDEPENDENT reliability signal for invoices — added
    directly in response to a real, confirmed structural gap:
    validate_reconciliation() only checks financial math, so it is
    completely blind to errors in non-numeric fields (a real example
    confirmed this directly: every number on an invoice was correct,
    but one line item's description text was wrong, and reconciliation
    still reported fully_reconciles=True regardless).

    This check is deliberately format/plausibility-based, not a
    ground-truth comparison — it has to be, since this runs at real
    inference time with no ground truth available, the same
    constraint every rule-based validator in this whole portfolio has
    worked under (e.g. the receipt pipeline's validate_total() checking
    currency FORMAT, not the actual correct value).

    GL codes in this project's real data consistently follow a
    "NNNN-Description" format (confirmed directly in
    generate_synthetic_invoices.py's own GL_CODES definitions) — a
    genuine, checkable structural pattern, not an arbitrary rule.
    Descriptions get a much looser plausibility check (non-empty,
    reasonable minimum length) since free text can't be format-checked
    the way a structured code can — this is intentionally a weaker
    signal, not a full description-correctness check, since no
    runtime-available signal can fully verify natural-language text
    content without ground truth.
    """
    gl_code_pattern = re.compile(r"^\d{4}-.+$")

    items_with_issues = []
    for i, item in enumerate(extraction.line_items):
        issues = []
        if item.gl_code and not gl_code_pattern.match(item.gl_code):
            issues.append(f"gl_code '{item.gl_code}' doesn't match expected NNNN-Description format")
        if not item.description or len(item.description.strip()) < 3:
            issues.append(f"description is empty or suspiciously short: '{item.description}'")
        if issues:
            items_with_issues.append({"line_item_index": i, "issues": issues})

    return {
        "all_line_items_plausible": len(items_with_issues) == 0,
        "items_with_issues": items_with_issues,
    }


def validate_header_fields(extraction: InvoiceExtraction) -> dict:
    """
    A THIRD, independent reliability signal — extending the same
    format-plausibility technique from validate_line_item_fields() to
    HEADER fields, added directly in response to a real, confirmed gap
    found through testing: an invoice with real OCR noise applied
    (but NOT the missing_po_number factor) had every single line item
    and every financial total exactly correct, yet its po_number was
    wrong — and NOTHING caught it, because validate_line_item_fields()
    only ever checked line-item fields, never header fields. Same
    root cause as the GL-code corruption case (OCR noise corrupting
    character-level text), just manifesting in a different field this
    validation never covered.

    po_number and invoice_number both follow real, checkable formats
    in this project's own generated data — confirmed directly from
    generate_synthetic_invoices.py's own formatting code, not
    guessed: po_number is always "PO-YYYY-NNNNN" (4-digit year,
    5-digit number), invoice_number is always "INV-NNNNNN" (6 digits).
    Both fields can also be legitimately EMPTY (po_number via the
    missing_po_number difficulty factor) — empty is correctly treated
    as "not applicable" here, not a format violation, the same
    distinction already established for GL codes.

    vendor, invoice_date, and due_date are deliberately NOT format-
    checked here — vendor is free text (the same reasoning as
    descriptions: no meaningful format to check without ground truth),
    and dates were already confirmed non-format-checkable in the
    receipt pipeline's own investigation (real SROIE ground truth used
    multiple valid date formats — a lesson directly reapplied here
    rather than re-learned).
    """
    po_number_pattern = re.compile(r"^PO-\d{4}-\d{5}$")
    invoice_number_pattern = re.compile(r"^INV-\d{6}$")

    issues = []
    if extraction.po_number and not po_number_pattern.match(extraction.po_number):
        issues.append(f"po_number '{extraction.po_number}' doesn't match expected PO-YYYY-NNNNN format")
    if extraction.invoice_number and not invoice_number_pattern.match(extraction.invoice_number):
        issues.append(f"invoice_number '{extraction.invoice_number}' doesn't match expected INV-NNNNNN format")

    return {
        "all_header_fields_plausible": len(issues) == 0,
        "issues": issues,
    }


def extract_and_validate_invoice(invoice_text: str, chain) -> dict:
    """
    Full pipeline for one invoice: extract, then independently verify
    the extraction's own internal math, line-item plausibility, AND
    header-field plausibility — three genuinely different,
    complementary signals, not comparing against external ground truth
    (that's what a separate evaluation script does), but checking
    whether the extraction is even SELF-consistent and well-formed at
    every level, none of which needs ground truth to check at all.
    """
    extraction = chain.invoke({"invoice_text": invoice_text})
    reconciliation = validate_reconciliation(extraction)
    field_plausibility = validate_line_item_fields(extraction)
    header_plausibility = validate_header_fields(extraction)

    return {
        "extraction": extraction.model_dump(),
        "reconciliation": reconciliation,
        "field_plausibility": field_plausibility,
        "header_plausibility": header_plausibility,
        "signals_agree": (
            extraction.confidence == "high"
            and reconciliation["fully_reconciles"]
            and field_plausibility["all_line_items_plausible"]
            and header_plausibility["all_header_fields_plausible"]
        ),
    }




