"""
Project 1, second half (continued): evaluates real extraction against
the deliberately hard synthetic invoices, with results broken down BY
which specific difficulty factor(s) each invoice had applied — the
genuinely new question this script exists to answer, beyond just "is
overall accuracy lower on hard data": WHICH specific real-world
difficulty (a discount line, missing GL codes, a missing PO number,
OCR noise) actually causes real extraction problems, and does the
reliability signal actually discriminate when real difficulty exists,
unlike the base 100 clean invoices where it never got tested against
any real failure at all.

Reuses the exact same comparison logic already tested in
run_invoice_eval.py (compare_numeric, compare_line_items,
normalize_text) rather than duplicating it, so results are directly,
fairly comparable between the clean and hard evaluations.
"""

import json
import random
from pathlib import Path

from run_invoice_eval import compare_numeric, compare_line_items, normalize_text

INVOICES_PATH = Path("data/synthetic/hard_synthetic_invoices.json")
RESULTS_PATH = Path("results/hard_invoice_extraction_results.json")
SAMPLE_SIZE = 30  # evaluating the whole hard set — only 30 exist, and
                  # the whole point is seeing how each factor performs,
                  # not further sampling down from an already-small set


def evaluate_one_hard_invoice(invoice: dict, chain) -> dict:
    from extract_invoice import extract_and_validate_invoice

    result = extract_and_validate_invoice(invoice["invoice_text"], chain)
    extraction = result["extraction"]
    ground_truth = invoice["ground_truth"]

    header_fields = ["vendor", "invoice_number", "invoice_date", "due_date", "po_number"]
    header_results = {
        field: normalize_text(str(extraction[field])) == normalize_text(str(ground_truth[field]))
        for field in header_fields
    }

    line_item_comparison = compare_line_items(extraction["line_items"], ground_truth["line_items"])

    financial_results = {
        "subtotal_exact": compare_numeric(extraction["subtotal"], ground_truth["subtotal"]),
        "tax_amount_exact": compare_numeric(extraction["tax_amount"], ground_truth["tax_amount"]),
        "total_exact": compare_numeric(extraction["total"], ground_truth["total"]),
    }

    all_fields_exact = (
        all(header_results.values())
        and line_item_comparison["all_items_fully_correct"]
        and all(financial_results.values())
    )

    return {
        "invoice_id": invoice["invoice_id"],
        "applied_difficulty_factors": invoice["applied_difficulty_factors"],
        "extraction": extraction,
        "ground_truth": ground_truth,
        "header_results": header_results,
        "line_item_comparison": line_item_comparison,
        "financial_results": financial_results,
        "all_fields_exact": all_fields_exact,
        "signals_agree": result["signals_agree"],
        "reconciliation": result["reconciliation"],
        "field_plausibility": result["field_plausibility"],
        "header_plausibility": result["header_plausibility"],
    }


def report_accuracy_by_factor(all_results: list[dict], factor_name: str):
    """
    Splits results into "had this factor" vs "didn't have this factor"
    and reports accuracy for each group separately — the core new
    analysis this script exists to provide. An invoice can have
    MULTIPLE factors at once (see generate_hard_invoices.py — factors
    are applied independently, not mutually exclusive), so these
    groups are not a clean partition of the whole sample; a given
    invoice can appear in multiple "had this factor" groups
    simultaneously. That's intentional: the question being asked is
    "does having X specifically correlate with lower accuracy",
    regardless of what else that invoice also had applied.
    """
    with_factor = [r for r in all_results if factor_name in r["applied_difficulty_factors"]]
    without_factor = [r for r in all_results if factor_name not in r["applied_difficulty_factors"]]

    with_accuracy = sum(1 for r in with_factor if r["all_fields_exact"]) / len(with_factor) if with_factor else None
    without_accuracy = sum(1 for r in without_factor if r["all_fields_exact"]) / len(without_factor) if without_factor else None

    print(f"\n  {factor_name}:")
    if with_accuracy is not None:
        print(f"    WITH this factor    ({len(with_factor)} invoices): {with_accuracy:.0%} fully correct")
    else:
        print(f"    WITH this factor: no invoices had it in this sample.")
    if without_accuracy is not None:
        print(f"    WITHOUT this factor ({len(without_factor)} invoices): {without_accuracy:.0%} fully correct")
    else:
        print(f"    WITHOUT this factor: no invoices lacked it in this sample.")

    if with_accuracy is not None and without_accuracy is not None:
        diff = (without_accuracy - with_accuracy) * 100
        if diff > 5:
            print(f"    -> This factor appears to genuinely hurt accuracy ({diff:.0f} point drop).")
        elif diff < -5:
            print(f"    -> Unexpected: accuracy was HIGHER with this factor present ({-diff:.0f} points) "
                  f"— worth investigating directly rather than assuming this is meaningful on a small sample.")
        else:
            print(f"    -> No clear effect from this factor alone in this sample.")


def main():
    invoices = json.loads(INVOICES_PATH.read_text(encoding="utf-8"))

    random.seed(42)
    sample = invoices if len(invoices) <= SAMPLE_SIZE else random.sample(invoices, SAMPLE_SIZE)
    print(f"Evaluating all {len(sample)} hard synthetic invoices...\n")

    print("Building the invoice extraction chain (requires ANTHROPIC_API_KEY)...")
    from extract_invoice import build_invoice_extraction_chain
    chain = build_invoice_extraction_chain()

    all_results = []
    for i, invoice in enumerate(sample, start=1):
        print(f"[{i}/{len(sample)}] {invoice['invoice_id']} "
              f"(factors: {invoice['applied_difficulty_factors'] or 'none'})...", end=" ")
        result = evaluate_one_hard_invoice(invoice, chain)
        all_results.append(result)
        status = "ALL CORRECT" if result["all_fields_exact"] else "some mismatch"
        print(f"{status} | signals_agree={result['signals_agree']}")

    print("\n" + "="*60)
    print(f"Overall accuracy on hard invoices: "
          f"{sum(1 for r in all_results if r['all_fields_exact'])}/{len(all_results)}")

    # Description comparison: lexical (substring) vs. semantic
    # (all-mpnet-base-v2) side by side — the real, direct test of
    # whether the semantic check actually recognizes cases the old
    # substring check couldn't, including the specific real paraphrase
    # case (HARD-000009) that motivated adding this signal in the
    # first place.
    all_item_results = [ir for r in all_results for ir in r["line_item_comparison"]["item_results"]]
    if all_item_results:
        print(f"\nDescription matching, across {len(all_item_results)} individual line items:")
        lexical_correct = sum(1 for ir in all_item_results if ir["description_close"])
        semantic_correct = sum(1 for ir in all_item_results if ir["description_semantically_close"])
        print(f"  description_close (lexical/substring): {lexical_correct}/{len(all_item_results)}")
        print(f"  description_semantically_close (semantic, all-mpnet-base-v2): {semantic_correct}/{len(all_item_results)}")

        # The specific, real, interesting cases: where the two signals
        # DISAGREE — this is where semantic similarity is either
        # correctly rescuing a real paraphrase the lexical check missed,
        # or (per the honest research limitation already documented)
        # potentially over-crediting something that isn't genuinely
        # equivalent. Worth surfacing directly, not just the aggregate.
        disagreements = [ir for ir in all_item_results if ir["description_close"] != ir["description_semantically_close"]]
        if disagreements:
            print(f"\n  {len(disagreements)} case(s) where the two signals disagreed:")
            for ir in disagreements:
                print(f"    lexical={ir['description_close']}, semantic={ir['description_semantically_close']} "
                      f"(similarity={ir['description_semantic_similarity']})")

    print("\nAccuracy broken down by difficulty factor:")
    for factor in ["discount", "missing_gl_codes", "missing_po_number", "ocr_noise"]:
        report_accuracy_by_factor(all_results, factor)

    # The real question this whole dataset was built to test — does
    # the reliability signal actually discriminate when real failures
    # exist, unlike the clean 100-invoice set where it never got a
    # real chance to.
    agreed = [r for r in all_results if r["signals_agree"]]
    disagreed = [r for r in all_results if not r["signals_agree"]]
    agreed_accuracy = sum(1 for r in agreed if r["all_fields_exact"]) / len(agreed) if agreed else None
    disagreed_accuracy = sum(1 for r in disagreed if r["all_fields_exact"]) / len(disagreed) if disagreed else None

    print(f"\nReliability signal check on hard invoices:")
    print(f"  AGREED ({len(agreed)} invoices): "
          f"accuracy = {agreed_accuracy:.0%}" if agreed else "  No invoices had agreeing signals.")
    print(f"  DISAGREED ({len(disagreed)} invoices): "
          f"accuracy = {disagreed_accuracy:.0%}" if disagreed else "  No invoices had disagreeing signals.")
    if agreed_accuracy is not None and disagreed_accuracy is not None:
        if agreed_accuracy > disagreed_accuracy:
            print(f"  -> The reliability signal IS informative here: "
                  f"{(agreed_accuracy - disagreed_accuracy)*100:.0f} point real gap.")
        else:
            print(f"  -> Not informative on this sample — worth real investigation, not just noting.")

    Path("results").mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()



    