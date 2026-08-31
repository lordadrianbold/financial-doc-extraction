"""
Project 1, Week 1 (continued): runs the layout-aware extraction
approach (real bounding-box Y-positions given to the model as
structured text context) against the SAME 15-receipt sample used for
every other evaluation in this project, so results are directly,
fairly comparable — not a different sample that could make an
improvement (or lack of one) look more or less dramatic than it
really is.

Reuses the exact same comparison logic already built and tested in
run_extraction_eval.py (compare_total, compare_text_field,
compare_address_postal_code) rather than duplicating it — the
comparison rules should be identical between the two evaluations for
the comparison itself to be fair.
"""

import json
import random
from pathlib import Path

from run_extraction_eval import (
    compare_total, compare_text_field, compare_address_postal_code, SAMPLE_SIZE
)

RECEIPTS_PATH = Path("data/processed/receipts.json")
RESULTS_PATH = Path("results/layout_aware_extraction_results.json")


def evaluate_one_receipt_layout_aware(receipt: dict, chain) -> dict:
    from extract_receipt import extract_and_validate_layout_aware

    if receipt.get("line_y_positions") is None:
        raise ValueError(
            f"Receipt {receipt['receipt_id']} has no line_y_positions — "
            f"re-run load_sroie_data.py to regenerate receipts.json with "
            f"the updated loader that captures bbox data."
        )

    words = receipt["ocr_text"].split("\n")
    result = extract_and_validate_layout_aware(words, receipt["line_y_positions"], chain)
    extraction = result["extraction"]
    validation = result["validation"]
    ground_truth = receipt["ground_truth"]

    field_results = {
        "company": compare_text_field(extraction["company"], ground_truth["company"]),
        "date": compare_text_field(extraction["date"], ground_truth["date"]),
        "address": compare_text_field(extraction["address"], ground_truth["address"] or ""),
        "total": {"exact_match": compare_total(extraction["total"], ground_truth["total"] or "")},
    }
    postal_code_match = compare_address_postal_code(extraction["address"], ground_truth["address"] or "")

    all_fields_exact = all(fr["exact_match"] for fr in field_results.values())
    practically_correct = (
        field_results["company"]["exact_match"]
        and field_results["date"]["exact_match"]
        and field_results["total"]["exact_match"]
        and bool(postal_code_match)
    )

    return {
        "receipt_id": receipt["receipt_id"],
        "extraction": extraction,
        "ground_truth": ground_truth,
        "field_results": field_results,
        "address_postal_code_match": postal_code_match,
        "all_fields_exact": all_fields_exact,
        "practically_correct": practically_correct,
        "signals_agree": validation["signals_agree"],
    }


def main():
    receipts = json.loads(RECEIPTS_PATH.read_text(encoding="utf-8"))
    complete_receipts = [r for r in receipts if not r["missing_ground_truth_fields"]]

    # SAME seed, SAME sample size as run_extraction_eval.py — this is
    # what makes the comparison fair: the exact same 15 receipts,
    # not a different random draw.
    random.seed(42)
    sample = random.sample(complete_receipts, min(SAMPLE_SIZE, len(complete_receipts)))
    print(f"Evaluating the same {len(sample)}-receipt sample with the layout-aware approach...\n")

    print("Building the layout-aware extraction chain (requires ANTHROPIC_API_KEY)...")
    from extract_receipt import build_layout_aware_extraction_chain
    chain = build_layout_aware_extraction_chain()

    all_results = []
    for i, receipt in enumerate(sample, start=1):
        print(f"[{i}/{len(sample)}] {receipt['receipt_id']}...", end=" ")
        result = evaluate_one_receipt_layout_aware(receipt, chain)
        all_results.append(result)
        status = "ALL CORRECT" if result["all_fields_exact"] else "some mismatch"
        print(f"{status} | signals_agree={result['signals_agree']}")

    print("\n" + "="*60)
    print("Per-field accuracy (layout-aware approach):")
    fields = ["company", "date", "address", "total"]
    for field in fields:
        exact_count = sum(1 for r in all_results if r["field_results"][field]["exact_match"])
        if "close_match" in all_results[0]["field_results"][field]:
            close_count = sum(1 for r in all_results if r["field_results"][field]["close_match"])
            print(f"  {field}: {exact_count}/{len(all_results)} exact, {close_count}/{len(all_results)} close")
        else:
            print(f"  {field}: {exact_count}/{len(all_results)} exact")
        if field == "address":
            postal_results = [r["address_postal_code_match"] for r in all_results if r["address_postal_code_match"] is not None]
            if postal_results:
                postal_match_count = sum(1 for m in postal_results if m)
                print(f"    (of which, correct postal code identified: {postal_match_count}/{len(postal_results)})")

    print(f"\nOverall accuracy (all 4 fields exact): "
          f"{sum(1 for r in all_results if r['all_fields_exact'])}/{len(all_results)}")
    print(f"Overall accuracy (practically correct): "
          f"{sum(1 for r in all_results if r['practically_correct'])}/{len(all_results)}")

    Path("results").mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {RESULTS_PATH}")
    print("\nCompare this directly against results/extraction_results.json "
          "(the original text-only approach) for the real, honest verdict on "
          "whether the layout-aware approach actually helped.")


if __name__ == "__main__":
    main()




    