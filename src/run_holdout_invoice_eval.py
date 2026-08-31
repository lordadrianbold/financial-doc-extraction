"""
Project 1: evaluates the exact same, unmodified reliability system
(extract_invoice.py's four signals — reconciliation, line-item field
plausibility, semantic similarity, header field plausibility) against
the fresh holdout set generated in generate_holdout_invoices.py.

Reuses evaluate_one_hard_invoice() and report_accuracy_by_factor()
directly from run_hard_invoice_eval.py, unchanged — the entire point
of this script is testing whether the SAME reliability code
generalizes to genuinely new data, not testing a different evaluation
approach.
"""

import json
from pathlib import Path

from run_hard_invoice_eval import evaluate_one_hard_invoice, report_accuracy_by_factor

INVOICES_PATH = Path("data/synthetic/holdout_hard_invoices.json")
RESULTS_PATH = Path("results/holdout_invoice_extraction_results.json")


def main():
    invoices = json.loads(INVOICES_PATH.read_text(encoding="utf-8"))
    print(f"Evaluating all {len(invoices)} HOLDOUT invoices — data never used to "
          f"develop or tune any of the four reliability signals being tested here.\n")

    print("Building the invoice extraction chain (requires ANTHROPIC_API_KEY)...")
    from extract_invoice import build_invoice_extraction_chain
    chain = build_invoice_extraction_chain()

    all_results = []
    for i, invoice in enumerate(invoices, start=1):
        print(f"[{i}/{len(invoices)}] {invoice['invoice_id']} "
              f"(factors: {invoice['applied_difficulty_factors'] or 'none'})...", end=" ")
        result = evaluate_one_hard_invoice(invoice, chain)
        all_results.append(result)
        status = "ALL CORRECT" if result["all_fields_exact"] else "some mismatch"
        print(f"{status} | signals_agree={result['signals_agree']}")

    print("\n" + "="*60)
    print(f"Overall accuracy on HOLDOUT invoices: "
          f"{sum(1 for r in all_results if r['all_fields_exact'])}/{len(all_results)}")

    all_item_results = [ir for r in all_results for ir in r["line_item_comparison"]["item_results"]]
    if all_item_results:
        print(f"\nDescription matching, across {len(all_item_results)} individual line items:")
        lexical_correct = sum(1 for ir in all_item_results if ir["description_close"])
        semantic_correct = sum(1 for ir in all_item_results if ir["description_semantically_close"])
        print(f"  description_close (lexical/substring): {lexical_correct}/{len(all_item_results)}")
        print(f"  description_semantically_close (semantic): {semantic_correct}/{len(all_item_results)}")

    print("\nAccuracy broken down by difficulty factor (HOLDOUT set):")
    for factor in ["discount", "missing_gl_codes", "missing_po_number", "ocr_noise"]:
        report_accuracy_by_factor(all_results, factor)

    # THE REAL QUESTION this whole script exists to answer: does the
    # reliability gap found on the original 30 invoices — the same
    # data used to develop all four signals — actually hold up on
    # genuinely fresh data, or was it partly an artifact of fitting to
    # that specific sample.
    agreed = [r for r in all_results if r["signals_agree"]]
    disagreed = [r for r in all_results if not r["signals_agree"]]
    agreed_accuracy = sum(1 for r in agreed if r["all_fields_exact"]) / len(agreed) if agreed else None
    disagreed_accuracy = sum(1 for r in disagreed if r["all_fields_exact"]) / len(disagreed) if disagreed else None

    print(f"\n" + "="*60)
    print(f"THE REAL TEST — reliability signal on genuinely unseen data:")
    print(f"  AGREED ({len(agreed)} invoices): "
          f"accuracy = {agreed_accuracy:.0%}" if agreed else "  No invoices had agreeing signals.")
    print(f"  DISAGREED ({len(disagreed)} invoices): "
          f"accuracy = {disagreed_accuracy:.0%}" if disagreed else "  No invoices had disagreeing signals.")

    if agreed_accuracy is not None and disagreed_accuracy is not None:
        gap = (agreed_accuracy - disagreed_accuracy) * 100
        print(f"\n  Original 30-invoice (tuning) set showed a 67-point gap.")
        print(f"  This holdout set shows a {gap:.0f}-point gap.")
        if gap >= 40:
            print(f"  -> The reliability system genuinely generalizes — this is real, "
                  f"not an artifact of fitting to the original sample.")
        elif gap >= 15:
            print(f"  -> The signal still works on new data, though somewhat weaker than "
                  f"on the original tuning set — a real, honest, worth-reporting result, "
                  f"not a failure.")
        else:
            print(f"  -> WARNING: the gap shrank substantially on genuinely new data — "
                  f"real evidence the original 67-point result may have been partly "
                  f"fitted to that specific sample's quirks. Worth reporting honestly, "
                  f"not hidden.")

    Path("results").mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()




    