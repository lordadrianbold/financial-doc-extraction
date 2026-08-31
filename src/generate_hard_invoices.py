"""
Project 1, second half (continued): deliberately harder synthetic
invoices — stress-testing the extraction and reliability pipeline
against realistic difficulty factors the first 100 clean invoices
never exercised at all.

Four real, distinct difficulty factors, each independently applied so
their individual effects can be measured separately, not just lumped
into one vague "hard" bucket:

1. A discount line — breaks the naive assumption that subtotal always
   equals the sum of line-item totals, the same category of real
   ambiguity that caused the one genuine SROIE total mismatch back in
   the receipt investigation (subtotal vs. net/final total). Here the
   correct subtotal is sum(line items) MINUS the discount, a real
   pattern many actual invoices have.
2. Missing GL codes on some line items — a genuinely realistic gap
   (not every line item gets coded immediately at invoice time in real
   accounting workflows), not an artificial injected error.
3. Missing PO number — realistic, since not every real business
   transaction references a purchase order.
4. OCR-style character noise — every invoice generated so far has been
   perfectly clean text; this is the first real test of extraction
   robustness against the kind of imperfect input SROIE's real receipts
   already exposed the pipeline to, applied here to the richer
   B2B invoice structure for the first time.
"""

import json
import random
from pathlib import Path

from generate_synthetic_invoices import (
    generate_invoice, format_invoice_as_text, CITIES_BY_STATE, REAL_STATE_TAX_RATES
)

OUTPUT_DIR = Path("data/synthetic")


def apply_discount(rng: random.Random, invoice: dict) -> dict:
    """
    Adds a realistic discount to an already-generated invoice,
    correctly recalculating subtotal/tax/total to still be exactly
    internally consistent — the same discipline as the base generator,
    just with one more real component in the math.
    """
    gt = invoice["ground_truth"]
    raw_subtotal = round(sum(item["line_total"] for item in gt["line_items"]), 2)

    # A realistic discount — either a flat dollar amount or a
    # percentage, matching real invoice conventions, not an arbitrary
    # random deduction.
    if rng.random() < 0.5:
        discount_amount = round(raw_subtotal * rng.uniform(0.02, 0.10), 2)
        discount_label = f"Volume Discount ({round(discount_amount/raw_subtotal*100)}%)"
    else:
        discount_amount = round(rng.uniform(10, 100), 2)
        discount_label = "Early Payment Discount"

    real_subtotal = round(raw_subtotal - discount_amount, 2)
    tax_amount = round(real_subtotal * gt["tax_rate"], 2)
    total = round(real_subtotal + tax_amount, 2)

    gt["discount_label"] = discount_label
    gt["discount_amount"] = discount_amount
    gt["raw_line_item_subtotal"] = raw_subtotal  # kept for reference/debugging, not the "real" subtotal
    gt["subtotal"] = real_subtotal
    gt["tax_amount"] = tax_amount
    gt["total"] = total
    return invoice


def apply_missing_gl_codes(rng: random.Random, invoice: dict, drop_rate: float = 0.3) -> dict:
    """
    Blanks out the GL code on a realistic fraction of line items —
    genuinely missing in the ground truth AND the rendered text (not
    just hidden from the text while still expected in the answer,
    which would be an unfair/impossible target, the same category of
    mistake as the address investigation's OCR-text-never-had-it
    finding, but deliberately introduced here rather than found by
    accident).
    """
    for item in invoice["ground_truth"]["line_items"]:
        if rng.random() < drop_rate:
            item["gl_code"] = ""
    return invoice


def apply_missing_po_number(rng: random.Random, invoice: dict, drop_rate: float = 0.3) -> dict:
    if rng.random() < drop_rate:
        invoice["ground_truth"]["po_number"] = ""
    return invoice


def apply_ocr_noise(rng: random.Random, text: str, noise_rate: float = 0.02) -> str:
    """
    Injects realistic, low-rate character-level noise — character
    substitution (the most common real OCR error type, e.g. 'O'/'0',
    'l'/'1', 'S'/'5' confusions) rather than random deletion, which
    would be less representative of genuine OCR behavior. A LOW rate
    (2% of characters by default) — real OCR on clean, machine-printed
    text (unlike SROIE's photographed/scanned real receipts) has a
    much lower error rate than photographed source material, so this
    should be a mild, realistic amount of noise, not corruption.
    """
    common_ocr_confusions = {
        "O": "0", "0": "O", "l": "1", "1": "l", "S": "5", "5": "S",
        "B": "8", "8": "B", "I": "1",
    }
    chars = list(text)
    for i, char in enumerate(chars):
        if char in common_ocr_confusions and rng.random() < noise_rate:
            chars[i] = common_ocr_confusions[char]
    return "".join(chars)


def format_hard_invoice_as_text(invoice: dict, state: str) -> str:
    """
    Renders the invoice as text, including the discount line if one
    was applied — a real, visible line item on the invoice, not a
    hidden adjustment the model would have no way to see.
    """
    # format_invoice_as_text() expects the GROUND TRUTH dict directly
    # (vendor, invoice_number, etc. at the top level) — invoice here
    # is the outer wrapper dict (invoice_id/invoice_text/ground_truth),
    # a real bug caught immediately on first run: passing the wrapper
    # itself instead of invoice["ground_truth"] raised a KeyError.
    gt = invoice["ground_truth"]
    base_text = format_invoice_as_text(gt, state)

    if "discount_amount" in gt:
        # Insert the discount line and correct subtotal breakdown
        # directly before the existing SUBTOTAL line, matching how a
        # real invoice would actually present this — the discount
        # visibly reducing the raw line-item total down to the real
        # subtotal.
        lines = base_text.split("\n")
        subtotal_line_idx = next(i for i, l in enumerate(lines) if "SUBTOTAL" in l)
        discount_line = f"{gt['discount_label']:>83} -{gt['discount_amount']:>6.2f}"
        raw_subtotal_line = f"{'LINE ITEM SUBTOTAL':>83} {gt['raw_line_item_subtotal']:>7.2f}"
        lines.insert(subtotal_line_idx, raw_subtotal_line)
        lines.insert(subtotal_line_idx + 1, discount_line)
        base_text = "\n".join(lines)

    return base_text


def generate_hard_invoice(rng: random.Random, invoice_number: int) -> dict:
    """
    Generates one invoice with a RANDOM SUBSET of the four difficulty
    factors applied (not all four every time — real invoices don't
    all share every possible complication at once, and applying every
    factor to every invoice would make it impossible to tell which
    factor is actually responsible for any given extraction failure).
    """
    base = generate_invoice(rng, invoice_number)
    base["invoice_id"] = f"HARD-{invoice_number:06d}"

    applied_factors = []
    if rng.random() < 0.4:
        base = apply_discount(rng, base)
        applied_factors.append("discount")
    if rng.random() < 0.5:
        base = apply_missing_gl_codes(rng, base)
        applied_factors.append("missing_gl_codes")
    if rng.random() < 0.3:
        base = apply_missing_po_number(rng, base)
        applied_factors.append("missing_po_number")

    # Determine state again from the vendor's rendered city — needed to
    # re-render the text after any ground-truth changes above.
    state = next(s for s, city in CITIES_BY_STATE.items() if city in base["invoice_text"])
    clean_text = format_hard_invoice_as_text(base, state)

    if rng.random() < 0.4:
        clean_text = apply_ocr_noise(rng, clean_text)
        applied_factors.append("ocr_noise")

    base["invoice_text"] = clean_text
    base["applied_difficulty_factors"] = applied_factors
    return base


def main(num_invoices: int = 30, seed: int = 123):
    rng = random.Random(seed)
    invoices = [generate_hard_invoice(rng, i + 1) for i in range(num_invoices)]

    # The SAME real internal-consistency check as the base generator —
    # even with discounts and other complications, every invoice's own
    # math must still be exactly correct by construction, or this
    # "hard" data would itself be broken, not just difficult.
    inconsistent = []
    for inv in invoices:
        gt = inv["ground_truth"]
        raw_subtotal = round(sum(item["line_total"] for item in gt["line_items"]), 2)
        expected_subtotal = round(raw_subtotal - gt.get("discount_amount", 0), 2)
        expected_tax = round(expected_subtotal * gt["tax_rate"], 2)
        expected_total = round(expected_subtotal + expected_tax, 2)
        if abs(expected_subtotal - gt["subtotal"]) > 0.01 or abs(expected_total - gt["total"]) > 0.01:
            inconsistent.append(inv["invoice_id"])

    if inconsistent:
        raise ValueError(f"Internal consistency check FAILED for {inconsistent[:5]}...")
    print(f"Internal consistency check passed: all {len(invoices)} hard invoices' "
          f"math is exactly correct given their own line items and any applied discount.")

    factor_counts = {}
    for inv in invoices:
        for factor in inv["applied_difficulty_factors"]:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
    print(f"\nDifficulty factors applied across {len(invoices)} invoices:")
    for factor, count in sorted(factor_counts.items()):
        print(f"  {factor}: {count} invoices")

    out_path = OUTPUT_DIR / "hard_synthetic_invoices.json"
    out_path.write_text(json.dumps(invoices, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")

    example_with_discount = next((i for i in invoices if "discount" in i["applied_difficulty_factors"]), None)
    if example_with_discount:
        print(f"\n{'='*72}\nExample with a discount applied:\n{'='*72}")
        print(example_with_discount["invoice_text"])


if __name__ == "__main__":
    main()




    