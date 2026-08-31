"""
Project 1, second half: runs real invoice extraction against a sample
of the synthetic invoices and compares results to their known, exactly
correct ground truth — genuinely different from the receipt pipeline's
evaluation in one important way: since these invoices were generated
by this project itself (see generate_synthetic_invoices.py), the
ground truth here is guaranteed complete and exactly correct by
construction, unlike SROIE's real, occasionally-imperfect human
annotations. Any mismatch found here is unambiguously an extraction
error, not a possible ground-truth quality issue — a genuinely cleaner
signal than the receipt evaluation could ever have.

Line-item comparison is the real new complexity this script has to
handle: matching a LIST of extracted line items against a LIST of
ground-truth line items, not just comparing two single scalar values.

SEMANTIC SIMILARITY, added directly in response to a real, confirmed
finding: the original bidirectional-substring `description_close`
check correctly identifies exact/truncated/expanded matches, but
cannot distinguish a legitimate paraphrase from a genuinely wrong
answer — confirmed as a real, industry-wide, actively-researched
limitation (not unique to this project) via direct research: 2026
industry sources report exact-match comparison showing "false-fail
rates above 30% on perfectly good answers" once paraphrasing is
involved, and embedding-based semantic similarity is the standard
first-line fix, recovering "60-80% of correct paraphrases that
exact-match silently rejects."

Model and threshold choice are BOTH grounded directly in real,
published research, not guessed: `all-mpnet-base-v2` was found (in a
real, cited comparison study on the MRPC paraphrase-detection
benchmark) to outperform BERT and RoBERTa-based sentence embedding
models for this exact task (75.6% accuracy, F1 0.836) — a different,
deliberate choice from the smaller `all-MiniLM-L6-v2` used elsewhere
in this portfolio's RAG project, made specifically because MPNet was
the real winner for THIS task in published research, not simply reused
for convenience. The similarity threshold (0.671) is that same study's
own reported optimal threshold for cosine similarity on this task —
not picked arbitrarily.

Honest limitation, also confirmed by the same research: semantic
similarity alone is vulnerable to negation ("we allow refunds" vs "we
do not allow refunds" can score as similar despite opposite meaning) —
a real, acknowledged blind spot of this approach, not claimed to be
fully solved here.
"""

import json
import random
import re
from pathlib import Path

INVOICES_PATH = Path("data/synthetic/synthetic_invoices.json")
RESULTS_PATH = Path("results/invoice_extraction_results.json")
SAMPLE_SIZE = 15

# The real, published optimal threshold for MPNet-based cosine
# similarity on paraphrase detection (MRPC benchmark) — not an
# arbitrary choice.
SEMANTIC_SIMILARITY_THRESHOLD = 0.671

# Lazy-loaded, module-level singleton — loading a ~400MB sentence
# embedding model on every single comparison call would be extremely
# wasteful; this is loaded once and reused across an entire evaluation
# run, the same pattern used for the embedding model in this
# portfolio's RAG project and live service.
_semantic_model = None


def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading semantic similarity model (all-mpnet-base-v2, ~420MB, one-time download)...")
        _semantic_model = SentenceTransformer("all-mpnet-base-v2")
    return _semantic_model


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Returns cosine similarity between two texts' sentence embeddings —
    a genuine semantic comparison, not a lexical/character-level one.
    Empty strings are handled explicitly rather than passed to the
    model, since embedding an empty string isn't meaningful and two
    empty strings should be treated as a clear non-match here (an
    empty extracted description is a real extraction failure, not a
    "semantically similar to nothing" edge case worth a high score).
    """
    if not text_a or not text_b:
        return 0.0
    model = get_semantic_model()
    embeddings = model.encode([text_a, text_b])
    import numpy as np
    a, b = embeddings[0], embeddings[1]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def compare_numeric(extracted, ground_truth, tolerance: float = 0.015) -> bool:
    """
    Same widened tolerance as extract_invoice.py's reconciliation
    check, for the same real reason — a genuine 1-cent difference
    doesn't compute to exactly 0.01 in floating point, confirmed by
    direct testing there.
    """
    try:
        return abs(float(extracted) - float(ground_truth)) <= tolerance
    except (ValueError, TypeError):
        return False


def compare_line_items(extracted_items: list[dict], truth_items: list[dict]) -> dict:
    """
    Compares two LISTS of line items — matched by POSITION (index),
    not by any other pairing strategy. This is a deliberate, simple
    choice: since extraction should naturally proceed in document
    order (the model reads top to bottom), positional matching is a
    reasonable default — but it has a real, acknowledged limitation:
    if the model extracts items in a genuinely different ORDER than
    the source document (or drops/duplicates one), positional matching
    could misattribute which extracted item corresponds to which real
    ground-truth item. The item COUNT mismatch check below at least
    surfaces this failure mode directly rather than silently
    mismatching items against each other.
    """
    count_matches = len(extracted_items) == len(truth_items)

    item_results = []
    # zip() stops at the shorter list — if counts genuinely differ,
    # this compares as many pairs as it can and the missing/extra
    # items are reflected in count_matches being False above, not
    # silently ignored.
    for extracted_item, truth_item in zip(extracted_items, truth_items):
        norm_extracted_desc = normalize_text(extracted_item["description"])
        norm_truth_desc = normalize_text(truth_item["description"])
        # Checks BOTH directions — a real bug caught by testing: the
        # original version only checked whether the (longer) ground-
        # truth description was a substring of the (typically shorter)
        # extracted one, which can never be true when the model
        # extracts a shortened version of the real description (e.g.
        # "Consulting Services" instead of the full "Consulting
        # Services - Hourly Rate"). Checking both directions correctly
        # handles a model extracting either a truncated OR an expanded
        # version of the real description.
        description_close = (
            norm_extracted_desc == norm_truth_desc
            or norm_truth_desc in norm_extracted_desc
            or norm_extracted_desc in norm_truth_desc
        )

        # A SEPARATE, genuinely different signal — semantic similarity
        # via sentence embeddings, added specifically because
        # description_close (a lexical/substring check) cannot detect
        # a legitimate paraphrase that shares no meaningful substring
        # with ground truth. Kept as an ADDITIONAL field, not a
        # replacement, so both can be directly compared on real data —
        # the same "add, don't silently swap" discipline used
        # throughout this whole project (e.g. postal_code_match added
        # alongside exact_match/close_match in the receipt pipeline,
        # not replacing either).
        semantic_similarity = compute_semantic_similarity(
            extracted_item["description"], truth_item["description"]
        )
        description_semantically_close = semantic_similarity >= SEMANTIC_SIMILARITY_THRESHOLD

        item_results.append({
            "description_close": description_close,
            "description_semantic_similarity": round(semantic_similarity, 3),
            "description_semantically_close": description_semantically_close,
            "gl_code_exact": extracted_item["gl_code"] == truth_item["gl_code"],
            "quantity_exact": extracted_item["quantity"] == truth_item["quantity"],
            "unit_price_exact": compare_numeric(extracted_item["unit_price"], truth_item["unit_price"]),
            "line_total_exact": compare_numeric(extracted_item["line_total"], truth_item["line_total"]),
        })

    # Explicitly list which fields must be True for an item to count
    # as fully correct — NOT a blind `all(field_result.values())`,
    # which would be a real bug given `description_semantic_similarity`
    # is a raw float (not boolean): any non-zero float is "truthy" in
    # Python, so a blind all() check would let even a terrible
    # similarity score (e.g. 0.01) silently pass, defeating the
    # threshold check entirely. Uses description_semantically_close
    # (not the older, cruder description_close) as the field that
    # determines correctness — the research-grounded signal is the
    # one that should decide the headline metric, while both remain
    # visible in item_results for direct comparison.
    boolean_fields = ["description_semantically_close", "gl_code_exact", "quantity_exact", "unit_price_exact", "line_total_exact"]
    all_items_fully_correct = (
        count_matches
        and len(item_results) > 0
        and all(all(field_result[f] for f in boolean_fields) for field_result in item_results)
    )

    return {
        "count_matches": count_matches,
        "extracted_count": len(extracted_items),
        "truth_count": len(truth_items),
        "item_results": item_results,
        "all_items_fully_correct": all_items_fully_correct,
    }


def evaluate_one_invoice(invoice: dict, chain) -> dict:
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


def main():
    invoices = json.loads(INVOICES_PATH.read_text(encoding="utf-8"))

    random.seed(42)
    sample = random.sample(invoices, min(SAMPLE_SIZE, len(invoices)))
    print(f"Evaluating a sample of {len(sample)} synthetic invoices...\n")

    print("Building the invoice extraction chain (requires ANTHROPIC_API_KEY)...")
    from extract_invoice import build_invoice_extraction_chain
    chain = build_invoice_extraction_chain()

    all_results = []
    for i, invoice in enumerate(sample, start=1):
        print(f"[{i}/{len(sample)}] {invoice['invoice_id']}...", end=" ")
        result = evaluate_one_invoice(invoice, chain)
        all_results.append(result)
        status = "ALL CORRECT" if result["all_fields_exact"] else "some mismatch"
        print(f"{status} | signals_agree={result['signals_agree']}")

    print("\n" + "="*60)
    print("Header field accuracy:")
    header_fields = ["vendor", "invoice_number", "invoice_date", "due_date", "po_number"]
    for field in header_fields:
        exact_count = sum(1 for r in all_results if r["header_results"][field])
        print(f"  {field}: {exact_count}/{len(all_results)} exact")

    print("\nLine item accuracy:")
    count_match_count = sum(1 for r in all_results if r["line_item_comparison"]["count_matches"])
    print(f"  Correct item count: {count_match_count}/{len(all_results)}")
    fully_correct_count = sum(1 for r in all_results if r["line_item_comparison"]["all_items_fully_correct"])
    print(f"  All items fully correct: {fully_correct_count}/{len(all_results)}")

    # A per-field breakdown ACROSS all individual line items in the
    # sample (not per-invoice) — a real, more granular signal, since
    # an invoice can have some correct and some incorrect items within
    # it, which the per-invoice "all items fully correct" bar alone
    # would hide.
    all_item_results = [ir for r in all_results for ir in r["line_item_comparison"]["item_results"]]
    if all_item_results:
        print(f"  (across {len(all_item_results)} individual line items compared)")
        for field in ["description_close", "description_semantically_close", "gl_code_exact", "quantity_exact", "unit_price_exact", "line_total_exact"]:
            correct = sum(1 for ir in all_item_results if ir[field])
            label = f"{field} (lexical/substring)" if field == "description_close" else \
                    f"{field} (semantic, all-mpnet-base-v2)" if field == "description_semantically_close" else field
            print(f"    {label}: {correct}/{len(all_item_results)}")

    print("\nFinancial totals accuracy:")
    for field in ["subtotal_exact", "tax_amount_exact", "total_exact"]:
        exact_count = sum(1 for r in all_results if r["financial_results"][field])
        print(f"  {field}: {exact_count}/{len(all_results)}")

    print(f"\nOverall accuracy (everything exactly correct): "
          f"{sum(1 for r in all_results if r['all_fields_exact'])}/{len(all_results)}")

    # The reconciliation-signal correlation check — mirroring the
    # receipt pipeline's real investigation into whether its
    # reliability signal actually predicts real accuracy, applied here
    # to see whether the SAME question holds for this genuinely
    # different, reconciliation-based signal.
    agreed = [r for r in all_results if r["signals_agree"]]
    disagreed = [r for r in all_results if not r["signals_agree"]]
    agreed_accuracy = sum(1 for r in agreed if r["all_fields_exact"]) / len(agreed) if agreed else None
    disagreed_accuracy = sum(1 for r in disagreed if r["all_fields_exact"]) / len(disagreed) if disagreed else None

    print(f"\nReliability signal check (confidence='high' AND invoice's own math reconciles):")
    print(f"  Receipts where signals AGREED: {len(agreed)}, "
          f"accuracy = {agreed_accuracy:.0%}" if agreed else "  No invoices had agreeing signals.")
    print(f"  Receipts where signals DISAGREED: {len(disagreed)}, "
          f"accuracy = {disagreed_accuracy:.0%}" if disagreed else "  No invoices had disagreeing signals.")

    Path("results").mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()



    