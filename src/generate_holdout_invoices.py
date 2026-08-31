"""
Project 1: a genuinely fresh, held-out hard invoice test set —
generated with a DIFFERENT random seed than the original 30
(seed=123), specifically so it was never used, directly or indirectly,
to discover or tune any of the four reliability signals now in
extract_invoice.py.

Why this matters: every one of those four fixes (discount awareness,
line-item field plausibility, semantic similarity, header field
plausibility) was found by investigating real failures on the
original 30-invoice hard set. That's principled, evidence-driven
development, not metric gaming — but it does mean the final reported
67-point reliability gap was measured on the same data that shaped the
system detecting it. This holdout set is the direct, honest test of
whether that result generalizes, or was partly earned by fitting to
this specific sample's particular quirks.

Reuses generate_hard_invoice() directly from generate_hard_invoices.py
rather than duplicating the difficulty-factor logic — the whole point
is testing the SAME reliability system against genuinely new data, not
also testing a second, different data-generation approach.
"""

import json
import random
from pathlib import Path

from generate_hard_invoices import generate_hard_invoice

OUTPUT_DIR = Path("data/synthetic")
HOLDOUT_SEED = 456  # deliberately different from the original hard set's seed=123


def main(num_invoices: int = 30, seed: int = HOLDOUT_SEED):
    rng = random.Random(seed)
    invoices = [generate_hard_invoice(rng, i + 1) for i in range(num_invoices)]

    # Renumber invoice_ids to make it visually obvious these are
    # holdout invoices, not accidentally confusable with the original
    # 30 (which share the same HARD-000001 through HARD-000030 IDs) —
    # a real, easy mistake to make when comparing two result files side
    # by side otherwise.
    for inv in invoices:
        original_num = inv["invoice_id"].split("-")[1]
        inv["invoice_id"] = f"HOLDOUT-{original_num}"

    # The same real internal-consistency check as every other
    # generator in this project — even fresh, never-tuned-against data
    # must still be genuinely correct by construction, not just
    # different.
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
    print(f"Internal consistency check passed: all {len(invoices)} holdout invoices' "
          f"math is exactly correct given their own line items and any applied discount.")

    factor_counts = {}
    for inv in invoices:
        for factor in inv["applied_difficulty_factors"]:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
    print(f"\nDifficulty factors applied across {len(invoices)} holdout invoices:")
    for factor, count in sorted(factor_counts.items()):
        print(f"  {factor}: {count} invoices")

    out_path = OUTPUT_DIR / "holdout_hard_invoices.json"
    out_path.write_text(json.dumps(invoices, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")
    print(f"\nSeed used: {seed} (original hard set used seed=123 — deliberately different, "
          f"confirming this is genuinely new data, not a re-generation of the same invoices)")


if __name__ == "__main__":
    main()




    