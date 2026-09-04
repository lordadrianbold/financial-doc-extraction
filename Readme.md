# Financial Document Extraction Pipeline

This is the first of five AI Engineering projects in this portfolio.
It targets a real market gap: structured outputs and function-calling
schema enforcement. The main focus is output reliability engineering.
Extraction accuracy alone is not enough.

**Status: complete.** The system is deployed live on AWS. Full
details are in [`notes.md`](notes.md). This includes every bug found,
the complete reliability investigation, and holdout validation
results. This README is a summary.

## Live demo

A live demo runs on AWS ECS Fargate. It is scaled to zero by default
to avoid ongoing cost. Restart it with this command:

```
aws ecs update-service --cluster ts-forecast-cluster --service financial-doc-extraction-service --desired-count 1 --region us-east-1
```

Then get its public IP. The IP changes every time the service
restarts.

```
aws ecs list-tasks --cluster ts-forecast-cluster --service-name financial-doc-extraction-service --desired-status RUNNING --region us-east-1
aws ecs describe-tasks --cluster ts-forecast-cluster --tasks <task-id> --region us-east-1
aws ec2 describe-network-interfaces --network-interface-ids <eni-id> --region us-east-1
```

Visit `http://<public-ip>:8000`. Click "Run Invoice Extraction Demo."
This runs the real extraction pipeline on a real sample invoice. It
shows the actual reliability verdict.

The service also has two API endpoints for real documents:
`POST /extract-receipt` and `POST /extract-invoice`.

Scale back to zero when done (`--desired-count 0`).

## Result

The reliability system uses four independent signals. When it says
trust an extraction, it is right 100% of the time.

This was checked twice. First on the data used to build the system.
Then on a completely fresh dataset the system had never seen.

**Real, measured results:**
- **987 real SROIE receipts** (ICDAR 2019 benchmark) were loaded and
  parsed correctly. This required finding and fixing a real data-
  format mismatch and a Parquet schema bug.
- **Receipt extraction results**: 100% date accuracy, 93% total
  accuracy, 73% company accuracy. The total accuracy number improved
  after fixing a currency-comparison bug that had been under-reporting
  the model's actual performance. Address extraction was investigated
  closely, and its real ceiling was found. This ceiling was confirmed
  against real published research — SROIE's own documented OCR quality
  issues, and benchmark results from layout-aware models like
  StrucTexT and LayoutLMv2.
- **160 synthetic B2B invoices** were generated: 100 clean, 30
  deliberately hard, and 30 held out for testing. Each invoice
  includes line items, GL codes, tax, discounts, and purchase order
  numbers. The math in every invoice is internally consistent. This
  was verified across more than 785 individual line items.
- **Invoice reliability system, final result**: 100% accuracy on
  every invoice flagged as reliable. Invoices flagged for review were
  only 33% accurate. That is a real 67-point gap on the tuning data.
  This gap was tested again on a completely fresh holdout set the
  system had never seen. The 100% precision held exactly. The overall
  gap was 50 points.

## Why reliability engineering, not just accuracy

Most extraction demos report a single accuracy number. This project
asks a different question: can the system tell you when to trust its
own output? This has to work at inference time, with no ground truth
available.

That is the real, differentiating skill this project set out to
demonstrate. Building it correctly required finding and fixing seven
distinct blind spots. It did not happen in one clean implementation.

## The real investigation, condensed

Full detail is in [`notes.md`](notes.md). Summary below.

- **A real data-format mismatch** was found by direct inspection. The
  documentation described a raw-text-file format. The actual
  downloaded SROIE data was shipped as Parquet instead. The loader was
  rewritten to match reality, not the documentation.
- **A real Parquet schema bug** was found. Missing ground-truth fields
  appear as present keys holding `None`. They do not appear as absent
  keys. An early version of the missing-field check silently failed to
  catch this. The bug was caught by testing against a real Parquet
  round-trip.
- **The address investigation is now definitively closed.** Three
  separate technical attempts were tried: rule-based validation,
  targeted prompt wording, and real spatial position data given to the
  model. Each was tested broadly. None helped. The real root cause was
  found by reading raw OCR text directly: some ground-truth addresses
  require text that was never captured by OCR at all. This was
  independently confirmed against real published research on this
  exact benchmark.
- **A real currency-comparison bug** treated `"$8.20"` and `"8.20"` as
  different values. This was under-reporting the model's genuine
  accuracy.
- **The reconciliation check was blind to discounts.** A perfectly
  correct extraction was flagged as unreliable, because the schema
  gave the model nowhere to report a legitimate discount. This was
  fixed by extending the schema and the reconciliation math. The fix
  was tested against adversarial cases to confirm it did not weaken
  real error detection.
- **Reconciliation itself had a structural blind spot.** Financial
  math checks can never detect errors in non-numeric fields. This was
  confirmed by a real example: every number was correct, but one
  description was wrong. The system still reported
  `fully_reconciles=True`. This was fixed with an independent field-
  plausibility signal.
- **A real, industry-documented paraphrase-detection limitation** was
  addressed with real research, not guesswork. The model
  `all-mpnet-base-v2` and its published optimal threshold (0.671) were
  added as a new semantic similarity signal. On live data, this signal
  correctly recognized the exact real paraphrase case that motivated
  it.
- **A header-field blind spot mirrored the line-item one.** OCR noise
  had corrupted a PO number, and no signal caught it, because no
  signal checked header fields. This was closed with a fourth,
  symmetric validator.
- **A holdout validation was run to check the results honestly.** A
  result measured on the same data used to build a system cannot be
  fully trusted. A fresh 30-invoice set was generated with a different
  random seed, and confirmed to be genuinely different data. The
  unmodified system was evaluated on this new set. The result that
  matters most — precision on "trust this" — held exactly. One
  specific per-factor finding reversed direction between the two
  samples. This was correctly flagged as real sampling noise, not
  hidden or overclaimed.

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
│   ├── run_holdout_invoice_eval.py      # the real holdout validation test
│   └── service.py                       # FastAPI wrapper + live demo page
├── results/
├── Dockerfile
├── task-definition.json         # the real ECS Fargate task definition
├── requirements.txt
├── requirements-service.txt     # lighter deps for the deployed service specifically
├── notes.md                     # full week-by-week technical write-up
└── README.md
```

## Setup

Real SROIE data must be downloaded on your own machine. The build
environment used for this project has network restrictions and cannot
reach huggingface.co directly.

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

To run the live service locally:

```
docker build -t financial-doc-extraction-service .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=%ANTHROPIC_API_KEY% financial-doc-extraction-service
```

Then visit `http://localhost:8000`.

## Notes

Full week-by-week reasoning is in [`notes.md`](notes.md). This
includes every real bug found and fixed, the complete reliability
investigation, the real research citations, and the exact failing
cases that motivated each fix.



