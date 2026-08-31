"""
Project 1, Week 1: loading and parsing the SROIE dataset (ICDAR 2019
Robust Reading Challenge on Scanned Receipts OCR and Information
Extraction).

REAL FORMAT, confirmed by directly inspecting the actual downloaded
data (this repackaged Hugging Face mirror, jsdnrs/ICDAR2019-SROIE) —
NOT the raw-folder-of-.txt-files format the original SROIE release
documentation describes, and not what an earlier version of this
loader was built around before the real file was ever inspected. The
real data ships as two Parquet files (train/test splits), each row
containing:
  - key: the receipt ID (e.g. "X00016469612")
  - words: a list of individual OCR'd text fragments — ALREADY parsed
    out of the raw bounding-box format, no manual parsing needed
  - entities: a dict with exactly 4 ground-truth fields — company,
    date, address, total — confirmed by direct inspection to match
    the field names in the original SROIE documentation
  - image / image_size: raw JPEG bytes and dimensions — present in
    this same file, but NOT used by this project (this project
    deliberately works from OCR text, not images, to stay distinct
    from Project 5's multimodal focus — see README). Worth noting for
    later: Project 5 could potentially reuse this exact same
    downloaded file rather than needing a separate download, since the
    images are already sitting right here.

This project uses SROIE's OCR TEXT (the `words` list) as input, not
the images — simulating the real, common production pattern where OCR
and structured extraction are separate pipeline stages, and keeping
this project's focus on RELIABLE STRUCTURED EXTRACTION from already-
messy text, distinct from Project 5's direct image-based approach.
"""

import json
from pathlib import Path

# Relative paths — this project follows the same convention as every
# other project in this portfolio: run scripts from the project root,
# not from inside src/, so these paths resolve correctly.
RAW_DIR = Path("data/raw/sroie/data")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# The 4 ground-truth fields SROIE's documentation says every receipt
# should have — used below to check real data against this expectation
# rather than assuming it's always true.
EXPECTED_ENTITY_FIELDS = {"company", "date", "address", "total"}


def parse_row(row: dict) -> dict:
    """
    Converts one raw parquet row into this project's working format:
    joined OCR text (simulating what a real upstream OCR stage would
    hand off) plus the ground-truth entities for later evaluation.

    `words` arrives as a numpy array (a normal artifact of reading
    Parquet through pandas) rather than a plain Python list — converted
    explicitly here, since a numpy array doesn't JSON-serialize the
    same way a plain list does, and this project needs to eventually
    save parsed receipts to a JSON file.
    """
    # list(...) converts numpy's array type into a plain Python list —
    # confirmed necessary by direct testing: json.dumps() fails on a
    # raw numpy array but works fine on a plain list.
    words = list(row["words"])

    # bboxes was present in the real parquet schema (confirmed by
    # direct inspection back in Week 1) but deliberately not loaded
    # until now — Project 1 originally worked from flat OCR text only,
    # to stay distinct from Project 5's planned multimodal scope. Added
    # here specifically to test a real, published research finding:
    # layout-aware systems (StrucTexT, LayoutLMv2) achieve 95%+ F1 on
    # this exact benchmark, while text-only approaches (three different
    # attempts, all tested broadly in this project) could not close
    # address's real accuracy gap. This uses the RAW COORDINATE VALUES
    # as structured text context in a prompt — a genuinely different,
    # lighter-weight approach than actually training a layout-aware
    # model, and still fully text-based (no image pixels involved),
    # keeping this within Project 1's scope rather than crossing into
    # Project 5's planned image-based approach.
    #
    # Each bbox is [x1, y1, x2, y2] (top-left and bottom-right corners,
    # confirmed by direct inspection of the real data). Only y1 (the
    # top edge) is kept here — a simple, sufficient proxy for a line's
    # vertical position on the page, which is what matters for
    # identifying which lines are spatially clustered close together
    # (like a multi-line address block) versus scattered elsewhere on
    # the receipt.
    bboxes = row.get("bboxes")
    line_y_positions = [int(box[1]) for box in bboxes] if bboxes is not None else None

    # dict(...) ensures a genuine plain Python dict, not whatever
    # parquet-specific mapping-like object pandas might hand back —
    # keeps downstream code (like .get() below) behaving predictably.
    entities = dict(row["entities"])

    # A real, worthwhile check rather than assuming every row is
    # perfectly well-formed: flag (don't silently drop) any row whose
    # ground truth is missing an expected field.
    #
    # IMPORTANT, confirmed by direct testing against a real parquet
    # round-trip: Parquet's columnar format requires a CONSISTENT
    # SCHEMA across all rows for a dict/struct column. If different
    # receipts originally had different sets of ground-truth keys,
    # pyarrow infers the union of all possible keys across every row
    # and fills genuinely-missing ones with None — so a "missing"
    # field still EXISTS as a key in the dict, just holding None
    # instead of being absent entirely. Checking for missing KEYS
    # (an earlier version of this check) silently failed to catch
    # this real case — confirmed by a real test failure before this
    # fix, not assumed in advance. The correct check is for missing or
    # empty VALUES, not missing keys.
    missing_fields = sorted(
        field for field in EXPECTED_ENTITY_FIELDS
        # .get(field) returns None if the key is genuinely absent, or
        # the stored value (possibly None) if the key exists — `not`
        # catches both cases uniformly, since `not None` and
        # `not ""` are both True.
        if not entities.get(field)
    )

    return {
        "receipt_id": row["key"],
        # "\n".join(words) simulates handing the extraction pipeline
        # already-messy, already-OCR'd text — one line per detected
        # text fragment, exactly the shape a real upstream OCR stage
        # would produce.
        "ocr_text": "\n".join(words),
        "ocr_word_count": len(words),
        "line_y_positions": line_y_positions,
        "ground_truth": entities,
        "missing_ground_truth_fields": missing_fields,
    }


def load_split(parquet_path: Path) -> list[dict]:
    """
    Loads one full parquet split (train or test) and parses every row.
    """
    # Imported inside the function rather than at module level — keeps
    # this module importable (e.g. for testing parse_row in isolation)
    # even in a context where pandas isn't installed, since only THIS
    # function actually needs it.
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    # df.iloc[i] retrieves row i as a Series (dict-like), which
    # parse_row expects — iterating by integer position rather than
    # df.iterrows() is a minor but deliberate choice for clarity here,
    # since the row order itself doesn't carry meaning to preserve.
    return [parse_row(df.iloc[i]) for i in range(len(df))]


def main():
    train_path = RAW_DIR / "train-00000-of-00001.parquet"
    test_path = RAW_DIR / "test-00000-of-00001.parquet"

    if not train_path.exists() or not test_path.exists():
        # Fail loudly with a clear, actionable message rather than
        # letting pandas raise its own less-helpful file-not-found
        # error deep inside load_split() — this is the FIRST thing
        # main() checks, before any real work happens.
        raise FileNotFoundError(
            f"Expected real SROIE parquet files at {train_path} and "
            f"{test_path} — see README.md for the real download step "
            f"(this sandbox cannot fetch huggingface.co directly)."
        )

    print("Loading train split...")
    train_receipts = load_split(train_path)
    print(f"  {len(train_receipts)} receipts")

    print("Loading test split...")
    test_receipts = load_split(test_path)
    print(f"  {len(test_receipts)} receipts")

    # Combined into one flat list — downstream steps (Week 2's
    # extraction pipeline) don't need to know or care which original
    # split a given receipt came from.
    all_receipts = train_receipts + test_receipts
    print(f"\nTotal: {len(all_receipts)} receipts")

    word_counts = [r["ocr_word_count"] for r in all_receipts]
    print(f"OCR word counts: avg {sum(word_counts)/len(word_counts):.1f}, "
          f"min {min(word_counts)}, max {max(word_counts)}")

    # A real, worthwhile summary check: how many receipts have
    # genuinely incomplete ground truth, given the real parquet
    # behavior confirmed above (missing fields show up as None values,
    # not absent keys). This matters directly for Week 2's evaluation
    # step — a receipt with incomplete ground truth can't be fairly
    # scored against a field it never had a real answer for.
    incomplete = [r for r in all_receipts if r["missing_ground_truth_fields"]]
    if incomplete:
        print(f"\nWARNING: {len(incomplete)} receipts have incomplete ground truth "
              f"(missing one or more of {sorted(EXPECTED_ENTITY_FIELDS)}). "
              f"These are still included in the output but should be excluded "
              f"from accuracy evaluation later, since there's nothing complete "
              f"to score them against.")
    else:
        print("\nAll receipts have complete ground truth (company, date, address, total).")

    # Saved as JSON (not re-saved as parquet) specifically so it's
    # directly readable/inspectable by a human or a simple script,
    # matching the same "save an inspectable intermediate artifact"
    # pattern used throughout this whole portfolio (e.g. the RAG
    # project's chunks.json).
    out_path = PROCESSED_DIR / "receipts.json"
    out_path.write_text(json.dumps(all_receipts, indent=2), encoding="utf-8")
    print(f"\nSaved parsed receipts to {out_path}")


if __name__ == "__main__":
    main()



    