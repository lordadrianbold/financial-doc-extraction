# Financial Document Extraction Pipeline

The first of five new AI Engineering-focused projects in this
portfolio, targeting a specific, current market gap: structured
outputs, function-calling schema enforcement, and — the real focus of
this project — genuine **output reliability engineering**, not just
extraction accuracy.

**Status: complete.** Full week-by-week findings, every real bug
encountered, and the complete reliability investigation (seven
distinct blind spots found and fixed across two datasets, plus
holdout validation) are in [`notes.md`](notes.md) — this README is the
front-door summary.

## Result

A four-signal reliability system that achieves **100% precision on
every extraction it recommends trusting** — verified not once, but
twice, including on a completely fresh holdout dataset the system was
never tuned against.

**Real, measured results:**
- **987 real SROIE receipts** (ICDAR 2019 benchmark) loaded and parsed
  correctly, after finding and fixing a real data-format mismatch and
  a real Parquet schema bug.
- **Receipt extraction**: 100% date accuracy, 93% total accuracy
  (after fixing a real currency-comparison bug that was under-reporting
  the model's actual performance), 73% company accuracy — and a
  definitively closed investigation into address extraction's real
  ceiling, independently confirmed against real published research
  (SROIE's own documented OCR quality issues, and real benchmark
  results from layout-aware models like StrucTexT/LayoutLMv2).
- **160 synthetic B2B invoices** (100 clean + 30 deliberately hard + 30
  held-out) — line items, GL codes, tax, discounts, and purchase order
  numbers — generated with genuine internal mathematical consistency,
  verified across 785+ individual line items.
- **Invoice reliability system, final result**: 100% accuracy on every
  invoice flagged as reliable, 33% on invoices flagged for review — a
  real 67-point gap on the tuning set, and **confirmed to hold on a
  genuinely fresh holdout set never used in development** (100%
  precision transferred exactly; overall gap 50 points).

## Why reliability engineering, not just accuracy

Most extraction demos report a single accuracy number. This project
asks a harder, more production-relevant question: **can the system
tell you, at inference time with no ground truth available, when to
trust its own output?** That's the real, differentiating skill this
project set out to demonstrate — and building it correctly took
finding and fixing seven genuinely distinct blind spots, not one clean
implementation.

## The real investigation, condensed

Full detail in [`notes.md`](notes.md). Summary:

- **A real data-format mismatch**, caught by direct inspection: the
  real downloaded SROIE data shipped as Parquet, not the raw-text-file
  format documented — the loader was rewritten to match reality, not
  documentation.
- **A real Parquet schema bug**: missing ground-truth fields appear as
  present keys holding `None`, not absent keys — an initial version's
  missing-field check silently failed to catch this, caught by testing
  against a real Parquet round-trip.
- **A definitively closed address investigation**: three different,
  genuinely distinct technical attempts (rule-based validation, targeted
  prompt wording, real spatial position data given to the model) each
  tested broadly and found not to help — the real root cause was
  confirmed by reading raw OCR text directly: some ground-truth
  addresses require text that was never captured by OCR at all,
  independently confirmed against real published research on this
  exact benchmark.
- **A real currency-comparison bug** that was under-reporting genuine
  model accuracy by treating `"$8.20"` and `"8.20"` as different values.
- **A discount-blind reconciliation check**: a perfectly correct
  extraction was flagged unreliable because the schema gave the model
  nowhere to report a legitimate discount — fixed by extending the
  schema and the reconciliation math, tested against adversarial cases
  to confirm the fix didn't loosen real error detection.
- **A structural blind spot in reconciliation itself**: financial math
  checks can never detect errors in non-numeric fields by definition —
  confirmed by a real example where every number was correct but one
  description was wrong, with `fully_reconciles=True` regardless. Fixed
  with an independent field-plausibility signal.
- **A real, industry-documented paraphrase-detection limitation**,
  addressed with real research: `all-mpnet-base-v2` and its published
  optimal threshold (0.671) — not guessed values — added as a genuinely
  new semantic similarity signal, confirmed on live data to correctly
  recognize the exact real paraphrase case that motivated it.
- **A header-field blind spot**, mirroring the line-item one: OCR noise
  corrupting a PO number went uncaught because no signal checked header
  fields — closed with a fourth, symmetric validator.
- **Honest holdout validation**: rather than trust a result measured on
  the same data used to develop it, a fresh 30-invoice set (different
  random seed, confirmed genuinely different data) was evaluated with
  the unmodified system. The result that matters most — precision on
  "trust this" — held exactly. A specific per-factor finding reversed
  direction between samples, correctly flagged as real sampling noise
  rather than hidden or overclaimed.

## Project structure

```
financial-doc-extraction/
├── data/
│   ├── raw/sroie/              # real downloaded SROIE parquet data (not tracked in git)
│   ├── processed/               # parsed receipts.json
│   └── synthetic/               # 100 clean + 30 hard + 30 holdout invoices
├── src/
│   ├── load_sroie_data.py               # real SROIE data loading
│   ├── extract_receipt.py               # receipt extraction + rule-based validation
│   ├── run_extraction_eval.py           # receipt evaluation
│   ├── run_layout_aware_eval.py         # the real-position-data address attempt
│   ├── generate_synthetic_invoices.py   # 100 clean synthetic invoices
│   ├── generate_hard_invoices.py        # 30 deliberately hard invoices
│   ├── generate_holdout_invoices.py     # 30 fresh holdout invoices
│   ├── extract_invoice.py               # invoice extraction + 4-signal reliability system
│   ├── run_invoice_eval.py              # clean invoice evaluation + semantic similarity
│   ├── run_hard_invoice_eval.py         # hard invoice evaluation, per-factor breakdown
│   └── run_holdout_invoice_eval.py      # the real holdout validation test
├── results/
├── requirements.txt
├── notes.md                     # full week-by-week technical write-up
└── README.md
```

## Setup

Real SROIE data must be downloaded on your own machine (this project's
build environment has network restrictions blocking huggingface.co
directly):

```
pip install -r requirements.txt
hf download jsdnrs/ICDAR2019-SROIE --repo-type dataset --local-dir data/raw/sroie
python src\load_sroie_data.py
python src\generate_synthetic_invoices.py
python src\generate_hard_invoices.py
python src\generate_holdout_invoices.py
```

Then, with `ANTHROPIC_API_KEY` set:

```
python src\run_extraction_eval.py
python src\run_invoice_eval.py
python src\run_hard_invoice_eval.py
python src\run_holdout_invoice_eval.py
```

## Notes

Full week-by-week reasoning, every real bug found and fixed, and the
complete reliability investigation — including the real research
citations and the exact failing cases that motivated each fix — is in
[`notes.md`](notes.md).




