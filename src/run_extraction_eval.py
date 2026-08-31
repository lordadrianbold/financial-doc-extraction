"""
Project 1, Week 1 (continued): runs real extraction against a sample
of real receipts and compares results to real ground truth — the
actual test of whether this pipeline works, not just whether it runs
without error.

The key research question this script exists to answer: does the
"signals_agree" reliability flag (built in extract_receipt.py)
actually correlate with real extraction accuracy? If receipts flagged
as reliable aren't meaningfully more accurate than receipts flagged
for review, the reliability system isn't actually doing its job —
this needs to be measured directly against real ground truth, not
assumed to work just because it was built with good intentions.
"""

import json
import random
import re
from pathlib import Path

RECEIPTS_PATH = Path("data/processed/receipts.json")
RESULTS_PATH = Path("results/extraction_results.json")
SAMPLE_SIZE = 15  # kept small deliberately — real API cost, not free


def normalize_text(s: str) -> str:
    """
    Lowercases and collapses whitespace — used for comparing text
    fields where the model might reasonably reproduce the same real
    content with minor, meaningless formatting differences (extra
    spaces, capitalization) that shouldn't count as a genuine mismatch.
    """
    return re.sub(r"\s+", " ", s.strip().lower())


CURRENCY_PREFIX_PATTERN = re.compile(r"^[A-Za-z$€£¥]*\s*")


def strip_currency_symbols(value: str) -> str:
    """
    Removes common currency symbols/codes and surrounding whitespace
    from a total string before parsing it as a number.

    Added directly in response to a real, confirmed bug: comparing
    real evaluation results found two genuine cases where the model's
    extraction was actually CORRECT ("8.20" extracted, following this
    project's own prompt instruction to extract digits only) but was
    being scored as WRONG purely because the real ground truth
    included a currency prefix ("$8.20", "RM 27.20") that
    float(ground_truth) can't parse directly — the underlying VALUES
    were identical in both real cases; only the comparison logic was
    broken, not the model's actual extraction.
    """
    return CURRENCY_PREFIX_PATTERN.sub("", value.strip()).strip()


def compare_total(extracted: str, ground_truth: str) -> bool:
    """
    Compares totals as FLOATS, not exact strings — "9.00" and "9.0"
    are the same real value even though they're different strings, and
    a naive string comparison would incorrectly count that as wrong.

    Strips currency symbols from BOTH sides before parsing — confirmed
    necessary by a real evaluation bug: real SROIE ground truth
    sometimes includes a currency prefix ("$8.20", "RM 27.20") even
    though the model (correctly following this project's prompt
    instructions to extract digits only) does not. Without stripping
    these first, float() would raise on the ground truth value and the
    comparison would incorrectly report a genuinely correct extraction
    as wrong.
    """
    try:
        extracted_value = float(strip_currency_symbols(extracted))
        truth_value = float(strip_currency_symbols(ground_truth))
        return abs(extracted_value - truth_value) < 0.001
    except (ValueError, TypeError):
        # Either value genuinely isn't a parseable number even after
        # stripping currency symbols — not a match, but also not worth
        # crashing the whole comparison over.
        return False


POSTAL_CODE_MATCH_PATTERN = re.compile(r"\b\d{5}\b")


def compare_address_postal_code(extracted: str, ground_truth: str) -> bool:
    """
    Checks whether the extracted address contains the SAME 5-digit
    postal code as the ground truth — a genuinely meaningful partial-
    correctness signal, not an arbitrary similarity threshold chosen
    to make a number look better.

    Why this specific check, and why it's honest: after three real,
    different attempts to improve address exact-match accuracy all
    failed to move the real number (documented in notes.md), it became
    clear that byte-perfect string matching was never a fair bar for
    this field — real addresses vary enormously in valid punctuation,
    spacing, and which secondary details (building name, floor,
    state) get included. A postal code, in contrast, is a precise,
    structured, unambiguous value: if the extracted address contains
    the correct one, the extraction correctly identified WHICH
    specific business location the receipt is from — genuinely useful
    for real accounts-payable or vendor-verification purposes, even
    when street-level detail is missing. This is reported ALONGSIDE
    exact_match and close_match, not as a replacement for either —
    the honest picture includes all three, not whichever one number
    looks best.
    """
    truth_postal = POSTAL_CODE_MATCH_PATTERN.search(ground_truth)
    if not truth_postal:
        # Ground truth itself has no identifiable postal code — this
        # check genuinely can't apply, and should not silently count
        # as either a pass or a fail.
        return None
    return truth_postal.group() in extracted


def compare_text_field(extracted: str, ground_truth: str) -> dict:
    """
    Compares a text field (company, date, address) two ways: an exact
    match after normalization, and a similarity-ratio-based "close"
    signal — since real OCR text can introduce minor character-level
    errors that a strict exact-match would unfairly penalize.

    Uses difflib's SequenceMatcher ratio rather than simple substring
    containment — confirmed necessary by direct testing against real
    data: a genuine real example ("BOOK TA .K(TAMAN DAYA) SDN BND" vs
    ground truth "...SDN BHD") differs by only a couple of characters
    but is NOT a substring of the other, so a substring-only check
    incorrectly failed to flag this as "close" at all. A similarity
    ratio correctly catches near-misses like this that a pure
    substring check misses.
    """
    import difflib

    norm_extracted = normalize_text(extracted)
    norm_truth = normalize_text(ground_truth)

    exact_match = norm_extracted == norm_truth

    # ratio() returns a value from 0 (completely different) to 1
    # (identical) based on the longest matching character sequences —
    # 0.85 is a deliberately chosen threshold: high enough that
    # genuinely different values won't be falsely flagged as close,
    # low enough to catch realistic minor OCR character differences.
    similarity = difflib.SequenceMatcher(None, norm_extracted, norm_truth).ratio()
    close_match = exact_match or similarity >= 0.85

    return {"exact_match": exact_match, "close_match": close_match, "similarity": round(similarity, 3)}


def evaluate_one_receipt(receipt: dict, chain) -> dict:
    """
    Runs extraction on one real receipt and compares every field
    against its real ground truth.
    """
    from extract_receipt import extract_and_validate

    result = extract_and_validate(receipt["ocr_text"], chain)
    extraction = result["extraction"]
    validation = result["validation"]
    ground_truth = receipt["ground_truth"]

    field_results = {
        "company": compare_text_field(extraction["company"], ground_truth["company"]),
        "date": compare_text_field(extraction["date"], ground_truth["date"]),
        "address": compare_text_field(extraction["address"], ground_truth["address"] or ""),
        "total": {"exact_match": compare_total(extraction["total"], ground_truth["total"] or "")},
    }

    # Reported ALONGSIDE exact_match/close_match, not merged into the
    # address dict above — this is a genuinely different kind of
    # signal (structured-value correctness, not text similarity), and
    # keeping it visibly separate makes clear it's an addition to the
    # honest picture, not a substitute for the stricter exact-match
    # result sitting right next to it.
    postal_code_match = compare_address_postal_code(extraction["address"], ground_truth["address"] or "")

    # A receipt counts as fully correct only if every field exactly
    # matched — the strictest, most honest bar, even though individual
    # fields have looser "close" signals available for diagnosis.
    all_fields_exact = all(fr["exact_match"] for fr in field_results.values())

    # A SECOND, DELIBERATELY MORE REALISTIC bar — added directly in
    # response to a real methodological problem, not to make a number
    # look better: address essentially NEVER achieves exact match
    # regardless of extraction quality (confirmed across every real
    # evaluation run in this project so far, 0/15 consistently), which
    # means all_fields_exact is structurally close to unachievable no
    # matter how good the reliability signal actually is. Testing
    # signals_agree against an almost-impossible target isn't a fair
    # test of whether the reliability system works — it's a test that
    # both groups will fail regardless. This bar uses exact match for
    # company/date/total (proven genuinely achievable — 93-100% real
    # accuracy on each) but the postal-code-match for address (the bar
    # established as fair and meaningful for that specific field, not
    # picked to make this number higher). Both bars are reported side
    # by side below — this doesn't replace all_fields_exact, it adds a
    # second, more honest question: does the reliability signal predict
    # PRACTICAL correctness, not an unfairly strict target.
    practically_correct = (
        field_results["company"]["exact_match"]
        and field_results["date"]["exact_match"]
        and field_results["total"]["exact_match"]
        and bool(postal_code_match)  # None (no postal code in ground truth) correctly counts as False here
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
        "model_self_reported_confidence": validation["model_self_reported_confidence"],
        "rule_based_valid": validation["rule_based_valid"],
        "failed_rule_checks": validation["failed_checks"],
    }


def main():
    receipts = json.loads(RECEIPTS_PATH.read_text(encoding="utf-8"))

    # Only evaluate receipts with COMPLETE ground truth — a receipt
    # missing a real answer for some field can't be fairly scored on
    # that field (this is exactly why Week 1's loader flagged these
    # explicitly rather than silently including them everywhere).
    complete_receipts = [r for r in receipts if not r["missing_ground_truth_fields"]]
    print(f"{len(complete_receipts)} of {len(receipts)} receipts have complete ground truth.")

    random.seed(42)  # reproducible sample — same receipts every run
    sample = random.sample(complete_receipts, min(SAMPLE_SIZE, len(complete_receipts)))
    print(f"Evaluating a sample of {len(sample)} receipts...\n")

    print("Building extraction chain (requires ANTHROPIC_API_KEY)...")
    from extract_receipt import build_extraction_chain
    chain = build_extraction_chain()

    all_results = []
    for i, receipt in enumerate(sample, start=1):
        print(f"[{i}/{len(sample)}] {receipt['receipt_id']}...", end=" ")
        result = evaluate_one_receipt(receipt, chain)
        all_results.append(result)
        status = "ALL CORRECT" if result["all_fields_exact"] else "some mismatch"
        print(f"{status} | signals_agree={result['signals_agree']}")

    # THE KEY QUESTION: does signals_agree actually correlate with
    # real accuracy? Split results into two groups and compare their
    # real, measured accuracy rates directly.
    agreed = [r for r in all_results if r["signals_agree"]]
    disagreed = [r for r in all_results if not r["signals_agree"]]

    agreed_accuracy = sum(1 for r in agreed if r["all_fields_exact"]) / len(agreed) if agreed else None
    disagreed_accuracy = sum(1 for r in disagreed if r["all_fields_exact"]) / len(disagreed) if disagreed else None

    # A SECOND, parallel correlation check against the more realistic
    # "practically_correct" bar — see evaluate_one_receipt()'s own
    # comment for why the strict all_fields_exact version above is a
    # structurally near-impossible target given address's real
    # exact-match difficulty, making it an unfair test of whether the
    # reliability signal actually works. Both results are reported
    # below, clearly labeled as testing two different, honestly
    # distinct questions — not one replacing the other.
    agreed_practical_accuracy = sum(1 for r in agreed if r["practically_correct"]) / len(agreed) if agreed else None
    disagreed_practical_accuracy = sum(1 for r in disagreed if r["practically_correct"]) / len(disagreed) if disagreed else None

    print("\n" + "="*60)

    # Per-field breakdown — added directly in response to real,
    # repeated need during actual investigation: the strict "all 4
    # fields exact" headline number alone hid genuinely important
    # differences between fields (e.g. address's 0% exact-match rate
    # dragging the headline to 0/15 even while date, total, and
    # company were performing well) that only became visible by
    # manually re-running ad-hoc breakdown commands after every
    # evaluation. Built into the standard report now instead.
    print("Per-field accuracy:")
    fields = ["company", "date", "address", "total"]
    for field in fields:
        exact_count = sum(1 for r in all_results if r["field_results"][field]["exact_match"])
        # "close_match" only exists for the three text fields (company,
        # date, address) — compare_total() only returns exact_match,
        # since a currency amount is either the same value or it isn't;
        # "close" doesn't mean anything meaningful for a number the way
        # it does for OCR'd text.
        if "close_match" in all_results[0]["field_results"][field]:
            close_count = sum(1 for r in all_results if r["field_results"][field]["close_match"])
            print(f"  {field}: {exact_count}/{len(all_results)} exact, {close_count}/{len(all_results)} close")
        else:
            print(f"  {field}: {exact_count}/{len(all_results)} exact")

        # Reported directly under address, clearly labeled as a
        # DIFFERENT, additional signal — not folded into the same
        # exact/close numbers above, so it's visibly an addition to
        # the honest picture rather than something substituted in to
        # make the headline number look better.
        if field == "address":
            postal_results = [r["address_postal_code_match"] for r in all_results if r["address_postal_code_match"] is not None]
            if postal_results:
                postal_match_count = sum(1 for m in postal_results if m)
                print(f"    (of which, correct postal code identified: "
                      f"{postal_match_count}/{len(postal_results)} — a looser but genuinely meaningful "
                      f"signal that the extraction found the right business location, even when the "
                      f"full address text isn't a character-perfect match)")

    print(f"\nOverall accuracy (all 4 fields exact): "
          f"{sum(1 for r in all_results if r['all_fields_exact'])}/{len(all_results)}")
    print(f"Overall accuracy (practically correct — postal code standard for address): "
          f"{sum(1 for r in all_results if r['practically_correct'])}/{len(all_results)}")

    print(f"\nReliability signal check — STRICT bar (all 4 fields exactly correct):")
    print(f"  Receipts where signals AGREED (flagged reliable): {len(agreed)}, "
          f"accuracy = {agreed_accuracy:.0%}" if agreed else "  No receipts had agreeing signals.")
    print(f"  Receipts where signals DISAGREED (flagged for review): {len(disagreed)}, "
          f"accuracy = {disagreed_accuracy:.0%}" if disagreed else "  No receipts had disagreeing signals.")
    if agreed_accuracy is not None and disagreed_accuracy is not None:
        if agreed_accuracy > disagreed_accuracy:
            print(f"  -> Informative under the strict bar: flagged-reliable receipts were "
                  f"{(agreed_accuracy - disagreed_accuracy)*100:.0f} points more accurate.")
        else:
            print(f"  -> Not informative under the strict bar — but this bar is near-impossible "
                  f"to hit at all given address's real exact-match difficulty (see below).")

    print(f"\nReliability signal check — PRACTICAL bar (postal code standard for address, "
          f"a fairer target given address's real, demonstrated exact-match difficulty):")
    print(f"  Receipts where signals AGREED (flagged reliable): {len(agreed)}, "
          f"accuracy = {agreed_practical_accuracy:.0%}" if agreed else "  No receipts had agreeing signals.")
    print(f"  Receipts where signals DISAGREED (flagged for review): {len(disagreed)}, "
          f"accuracy = {disagreed_practical_accuracy:.0%}" if disagreed else "  No receipts had disagreeing signals.")
    if agreed_practical_accuracy is not None and disagreed_practical_accuracy is not None:
        if agreed_practical_accuracy > disagreed_practical_accuracy:
            print(f"  -> The reliability signal IS informative under the practical bar: "
                  f"flagged-reliable receipts were {(agreed_practical_accuracy - disagreed_practical_accuracy)*100:.0f} "
                  f"points more accurate.")
        else:
            print(f"  -> WARNING: even under the fairer practical bar, flagged-reliable receipts "
                  f"were NOT more accurate than flagged-for-review ones — this is worth taking "
                  f"seriously, not explained away by the strict-bar issue alone.")

    Path("results").mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()



    