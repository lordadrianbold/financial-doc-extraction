"""
Project 1, second half of the data plan: synthetic B2B invoices,
supplementing SROIE's real receipt data with the richer structure real
B2B invoices have that SROIE genuinely lacks — line items, GL codes,
tax breakdowns, and purchase order references (SROIE's ground truth
only ever has 4 flat fields: company, date, address, total).

Design discipline matches the GL anomaly detection project's synthetic
data: realistic patterns generated from real business conventions, not
uniform random noise, with genuine internal consistency (line items
that actually sum to the stated subtotal, tax calculated correctly
from a real rate) — so this data can meaningfully test whether an
extraction pipeline can handle realistic invoice complexity, not just
whether it can parse arbitrary text.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path("data/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Real, common US sales tax rates by state — used to generate a
# realistic, varied (not fixed/uniform) tax rate per invoice, since a
# single hardcoded rate would make every invoice's tax math trivially
# predictable rather than genuinely testing extraction of a real,
# varying value.
REAL_STATE_TAX_RATES = {
    "CA": 0.0725, "NY": 0.04, "TX": 0.0625, "WA": 0.065,
    "IL": 0.0625, "PA": 0.06, "FL": 0.06, "OH": 0.0575,
}

# A small, realistic chart-of-accounts-style GL code set, matching real
# common business expense categories — not arbitrary codes, so line
# items can be assigned a GL code that genuinely matches their real
# category (an office-supplies line item gets an office-supplies code,
# not a random one).
GL_CODES = {
    "office_supplies": "6100-Office Supplies",
    "software": "6200-Software & Subscriptions",
    "professional_services": "6300-Professional Services",
    "equipment": "6400-Equipment & Hardware",
    "shipping": "6500-Shipping & Freight",
    "utilities": "6600-Utilities",
    "maintenance": "6700-Maintenance & Repairs",
}

# Real, plausible line-item templates per category — (description
# template, realistic unit price range, realistic quantity range).
# Prices/quantities are ranges grounded in real-world plausibility for
# each category (e.g. software licenses are typically higher unit
# price / lower quantity than office supplies), not uniform across
# every category.
LINE_ITEM_TEMPLATES = {
    "office_supplies": [
        ("Copy Paper, Letter Size, {qty} Reams", (4.50, 8.00), (5, 50)),
        ("Ballpoint Pens, Box of 12", (3.00, 6.00), (2, 20)),
        ("File Folders, Letter Size, Box of 100", (12.00, 22.00), (1, 10)),
        ("Sticky Notes, 3x3 inch, Pack of 12", (8.00, 14.00), (1, 15)),
    ],
    "software": [
        ("Annual Software License - Project Management", (200.00, 600.00), (1, 25)),
        ("Cloud Storage Subscription - Monthly", (15.00, 50.00), (1, 12)),
        ("Antivirus License - Per Seat, Annual", (30.00, 80.00), (5, 100)),
    ],
    "professional_services": [
        ("Consulting Services - Hourly Rate", (95.00, 250.00), (4, 80)),
        ("Legal Review Services", (150.00, 400.00), (2, 20)),
        ("Accounting & Bookkeeping Services - Monthly", (300.00, 900.00), (1, 3)),
    ],
    "equipment": [
        ("Wireless Keyboard and Mouse Combo", (25.00, 60.00), (1, 15)),
        ("27-inch Monitor", (150.00, 400.00), (1, 10)),
        ("Laptop Docking Station", (60.00, 150.00), (1, 10)),
    ],
    "shipping": [
        ("Ground Shipping - Standard", (8.00, 25.00), (1, 20)),
        ("Expedited Freight Delivery", (40.00, 120.00), (1, 5)),
    ],
}

VENDOR_NAMES = [
    "Meridian Office Supply Co.", "Northgate Business Solutions LLC",
    "Cascade Professional Services", "Ironwood Technology Partners",
    "Blue Harbor Logistics Inc.", "Summit Consulting Group",
    "Redstone Equipment Rentals", "Pacific Crest Software Inc.",
]

CITIES_BY_STATE = {
    "CA": "San Diego, CA", "NY": "Albany, NY", "TX": "Austin, TX",
    "WA": "Tacoma, WA", "IL": "Springfield, IL", "PA": "Harrisburg, PA",
    "FL": "Orlando, FL", "OH": "Columbus, OH",
}


def generate_line_items(rng: random.Random, num_items: int) -> list[dict]:
    """
    Generates a realistic, internally-consistent set of line items —
    each with a real GL category, a plausible quantity/price for that
    category (not a uniform random range across all categories), and a
    correctly calculated line total (quantity * unit_price), so the
    generated invoice's own math is genuinely correct, not just
    plausible-looking text.
    """
    categories = rng.sample(list(LINE_ITEM_TEMPLATES.keys()), min(num_items, len(LINE_ITEM_TEMPLATES)))
    # If more items requested than distinct categories available,
    # allow repeats — a single invoice legitimately CAN have multiple
    # items from the same category (e.g. two different office supply
    # line items), so this isn't unrealistic.
    while len(categories) < num_items:
        categories.append(rng.choice(list(LINE_ITEM_TEMPLATES.keys())))

    line_items = []
    for category in categories:
        template, price_range, qty_range = rng.choice(LINE_ITEM_TEMPLATES[category])
        quantity = rng.randint(*qty_range)
        unit_price = round(rng.uniform(*price_range), 2)
        line_total = round(quantity * unit_price, 2)
        description = template.format(qty=quantity) if "{qty}" in template else template

        line_items.append({
            "description": description,
            "gl_code": GL_CODES[category],
            "quantity": quantity,
            "unit_price": unit_price,
            "line_total": line_total,
        })
    return line_items


def generate_invoice(rng: random.Random, invoice_number: int) -> dict:
    """
    Generates one complete, internally-consistent synthetic invoice —
    real subtotal (sum of line totals), real tax (a genuine state rate
    applied to the real subtotal), real total (subtotal + tax) — so
    every generated value is genuinely derivable and checkable, not
    just superficially realistic-looking text.
    """
    state = rng.choice(list(REAL_STATE_TAX_RATES.keys()))
    tax_rate = REAL_STATE_TAX_RATES[state]

    num_items = rng.randint(2, 6)
    line_items = generate_line_items(rng, num_items)

    subtotal = round(sum(item["line_total"] for item in line_items), 2)
    tax_amount = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax_amount, 2)

    invoice_date = datetime(2026, 1, 1) + timedelta(days=rng.randint(0, 240))
    # NET 30 — a real, extremely common real-world payment term
    # convention, not an arbitrary choice.
    due_date = invoice_date + timedelta(days=30)

    vendor = rng.choice(VENDOR_NAMES)
    po_number = f"PO-{invoice_date.year}-{rng.randint(1000, 9999):05d}"

    ground_truth = {
        "vendor": vendor,
        "invoice_number": f"INV-{invoice_number:06d}",
        "invoice_date": invoice_date.strftime("%Y-%m-%d"),
        "due_date": due_date.strftime("%Y-%m-%d"),
        "po_number": po_number,
        "line_items": line_items,
        "subtotal": subtotal,
        "tax_rate": tax_rate,
        "tax_amount": tax_amount,
        "total": total,
    }

    invoice_text = format_invoice_as_text(ground_truth, state)

    return {
        "invoice_id": f"SYNTH-{invoice_number:06d}",
        "invoice_text": invoice_text,
        "ground_truth": ground_truth,
    }


def format_invoice_as_text(invoice: dict, state: str) -> str:
    """
    Renders the structured invoice data as plain text resembling a
    real invoice layout — this is what the extraction pipeline will
    actually receive as input, analogous to SROIE's OCR text.
    """
    lines = [
        invoice["vendor"],
        CITIES_BY_STATE[state],
        "",
        f"INVOICE #: {invoice['invoice_number']}",
        f"DATE: {invoice['invoice_date']}",
        f"DUE DATE: {invoice['due_date']}",
        f"PO NUMBER: {invoice['po_number']}",
        "",
        "DESCRIPTION                                    QTY   UNIT PRICE   TOTAL    GL CODE",
        "-" * 90,
    ]
    for item in invoice["line_items"]:
        lines.append(
            f"{item['description']:<45} {item['quantity']:>5} "
            f"{item['unit_price']:>10.2f} {item['line_total']:>10.2f}   {item['gl_code']}"
        )
    lines.extend([
        "-" * 90,
        f"{'SUBTOTAL':>83} {invoice['subtotal']:>7.2f}",
        f"{'TAX (' + str(round(invoice['tax_rate']*100, 2)) + '%)':>83} {invoice['tax_amount']:>7.2f}",
        f"{'TOTAL':>83} {invoice['total']:>7.2f}",
    ])
    return "\n".join(lines)


def main(num_invoices: int = 100, seed: int = 42):
    rng = random.Random(seed)  # explicit, seeded RNG instance (not the
    # global random module) — makes this generator's output fully
    # reproducible and independent of any other code's random calls
    # elsewhere in a larger pipeline.

    invoices = [generate_invoice(rng, i + 1) for i in range(num_invoices)]

    # A real, worthwhile self-check before saving anything — verify
    # every generated invoice's own math is genuinely internally
    # consistent, the same discipline as validating synthetic data in
    # the GL anomaly detection project earlier in this portfolio.
    inconsistent = []
    for inv in invoices:
        gt = inv["ground_truth"]
        real_subtotal = round(sum(item["line_total"] for item in gt["line_items"]), 2)
        real_tax = round(real_subtotal * gt["tax_rate"], 2)
        real_total = round(real_subtotal + real_tax, 2)
        if abs(real_subtotal - gt["subtotal"]) > 0.01 or abs(real_total - gt["total"]) > 0.01:
            inconsistent.append(inv["invoice_id"])

    if inconsistent:
        raise ValueError(
            f"Internal consistency check FAILED for {len(inconsistent)} generated "
            f"invoices: {inconsistent[:5]}... — this means the generator itself has "
            f"a bug, since every generated invoice's totals should be exactly "
            f"derivable from its own line items by construction."
        )
    print(f"Internal consistency check passed: all {len(invoices)} invoices' "
          f"subtotal/tax/total values are exactly correct given their own line items.")

    out_path = OUTPUT_DIR / "synthetic_invoices.json"
    out_path.write_text(json.dumps(invoices, indent=2), encoding="utf-8")
    print(f"Saved {len(invoices)} synthetic invoices to {out_path}")

    # Print one real example so the actual generated text can be
    # eyeballed directly, not just trusted from the consistency check
    # alone.
    print(f"\n{'='*72}\nExample generated invoice:\n{'='*72}")
    print(invoices[0]["invoice_text"])


if __name__ == "__main__":
    main()




    