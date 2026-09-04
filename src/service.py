"""
Project 1: FastAPI service wrapping both real extraction pipelines
(receipts with rule-based validation, invoices with the 4-signal
reliability system) — following the exact proven pattern from the
financial-rag-agent and month-end-close-assistant projects' own live
services (lifespan startup handler building both chains once, not
per-request).

Two genuinely separate extraction endpoints, matching the two
genuinely different real pipelines and schemas — not one generic
endpoint pretending they're the same thing. Only the invoice pipeline
gets a bundled "/run-demo" endpoint, since real generated invoice data
is available to bundle; genuine SROIE receipt data requires a real,
separate local download (network-restricted in the build environment)
and is not faked here for demo convenience.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

_receipt_chain = None
_invoice_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Builds both real extraction chains ONCE at startup — the same
    proven pattern as every other live service in this portfolio.
    """
    global _receipt_chain, _invoice_chain
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from extract_receipt import build_extraction_chain
    from extract_invoice import build_invoice_extraction_chain

    print("Building the receipt extraction chain...")
    _receipt_chain = build_extraction_chain()
    print("Building the invoice extraction chain...")
    _invoice_chain = build_invoice_extraction_chain()
    print("Both chains ready.")
    yield
    _receipt_chain = None
    _invoice_chain = None


app = FastAPI(title="Financial Document Extraction", lifespan=lifespan)


class ReceiptRequest(BaseModel):
    ocr_text: str


class InvoiceRequest(BaseModel):
    invoice_text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "receipt_chain_ready": _receipt_chain is not None,
        "invoice_chain_ready": _invoice_chain is not None,
    }


@app.post("/extract-receipt")
def extract_receipt_endpoint(request: ReceiptRequest):
    """
    Real API for real caller-supplied receipt OCR text — the same
    shape SROIE's real OCR text takes (one line per detected text
    fragment). No bundled demo here, deliberately: genuine SROIE data
    requires its own real, separate local download, and this project's
    entire discipline has been about not faking real data for
    convenience.
    """
    if _receipt_chain is None:
        raise HTTPException(status_code=503, detail="Receipt chain not ready yet — service is still starting up.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on this service.")

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from extract_receipt import extract_and_validate

    try:
        return extract_and_validate(request.ocr_text, _receipt_chain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")


@app.post("/extract-invoice")
def extract_invoice_endpoint(request: InvoiceRequest):
    """
    Real API for real caller-supplied invoice text.
    """
    if _invoice_chain is None:
        raise HTTPException(status_code=503, detail="Invoice chain not ready yet — service is still starting up.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on this service.")

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from extract_invoice import extract_and_validate_invoice

    try:
        return extract_and_validate_invoice(request.invoice_text, _invoice_chain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")


@app.get("/run-invoice-demo")
def run_invoice_demo():
    """
    Runs the real invoice pipeline against one real, bundled synthetic
    invoice — no input needed, specifically for the live demo page.
    """
    import json
    from pathlib import Path

    demo_data_path = Path(__file__).parent.parent / "data" / "synthetic" / "synthetic_invoices.json"
    if not demo_data_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Demo data not found at {demo_data_path} — check the Docker image includes data/synthetic/."
        )
    invoices = json.loads(demo_data_path.read_text(encoding="utf-8"))
    sample_invoice = invoices[0]

    if _invoice_chain is None:
        raise HTTPException(status_code=503, detail="Invoice chain not ready yet — service is still starting up.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on this service.")

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from extract_invoice import extract_and_validate_invoice

    try:
        result = extract_and_validate_invoice(sample_invoice["invoice_text"], _invoice_chain)
        result["invoice_text"] = sample_invoice["invoice_text"]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")


DEMO_PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Financial Document Extraction — Live Demo</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 780px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }
  h1 { font-size: 22px; }
  p.subtitle { color: #555; }
  button { background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-size: 15px; cursor: pointer; }
  button:disabled { background: #93b4ec; cursor: default; }
  #status { margin: 12px 0; color: #555; }
  .section { margin-top: 28px; }
  .section h2 { font-size: 16px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
  pre { white-space: pre-wrap; background: #f8f8f8; padding: 14px; border-radius: 6px; font-size: 13px; line-height: 1.5; }
  .trust { color: #15803d; font-weight: 600; }
  .review { color: #b91c1c; font-weight: 600; }
  code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }
</style>
</head>
<body>
  <h1>Financial Document Extraction</h1>
  <p class="subtitle">A live pipeline demonstrating LLM structured extraction with an independent,
  multi-signal reliability system — not just extraction accuracy, but whether the system can tell
  you when to trust its own output.</p>

  <button id="runBtn" onclick="runDemo()">Run Invoice Extraction Demo</button>
  <div id="status"></div>

  <div id="results" style="display:none;">
    <div class="section">
      <h2>Source Invoice Text</h2>
      <pre id="sourceText"></pre>
    </div>
    <div class="section">
      <h2>Extraction</h2>
      <pre id="extraction"></pre>
    </div>
    <div class="section">
      <h2>Reliability Signal</h2>
      <div id="reliability"></div>
    </div>
  </div>

  <div class="section">
    <h2>Real API endpoints</h2>
    <p>This service also exposes real, programmatic endpoints:
    <code>POST /extract-receipt</code> (real SROIE-format OCR text) and
    <code>POST /extract-invoice</code> (real invoice text) — both requiring genuine caller-supplied
    document text, not bundled demo data.</p>
  </div>

<script>
async function runDemo() {
  const btn = document.getElementById('runBtn');
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  btn.disabled = true;
  status.textContent = 'Running the pipeline (real LLM calls in progress)...';
  results.style.display = 'none';

  try {
    const resp = await fetch('/run-invoice-demo');
    if (!resp.ok) {
      const err = await resp.json();
      status.textContent = 'Error: ' + (err.detail || resp.statusText);
      btn.disabled = false;
      return;
    }
    const data = await resp.json();

    document.getElementById('sourceText').textContent = data.invoice_text;
    document.getElementById('extraction').textContent = JSON.stringify(data.extraction, null, 2);

    const reliabilityDiv = document.getElementById('reliability');
    const trusted = data.signals_agree;
    reliabilityDiv.innerHTML = trusted
      ? '<span class="trust">✓ TRUST THIS — reconciliation, field format, and confidence all agree</span>'
      : '<span class="review">⚠ FLAGGED FOR REVIEW — one or more reliability signals disagree</span>';

    results.style.display = 'block';
    status.textContent = '';
  } catch (e) {
    status.textContent = 'Request failed: ' + e;
  }
  btn.disabled = false;
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return DEMO_PAGE_HTML





