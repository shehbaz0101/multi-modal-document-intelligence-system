# Installation Guide — CPU-only (Dell 5310, i5-10th gen, 16 GB RAM)

This guide gets DocIntel running on a machine with no GPU. Everything here
has been tuned for your hardware. Expected cold-start time from fresh OS:
about 20–30 minutes (mostly download time).

---

## What will run locally vs in the cloud

| Component | Runs where | Notes |
|---|---|---|
| Qdrant (vector DB) | Local (Docker) | ~300 MB RAM |
| OpenSearch (BM25) | Local (Docker) | ~1 GB RAM |
| Postgres | Local (Docker) | ~100 MB RAM |
| Parsing (pymupdf4llm) | Local | 1–5 ms/page, no GPU |
| Embedding (MiniLM ONNX) | Local | ~1500 sentences/sec on i5 |
| Figure captioning | Claude API | remote, no GPU |
| Answer generation | Claude API | remote, no GPU |

Total RAM in use when running: ~2.5 GB out of your 16 GB. Comfortable.

---

## Step 1 — Install Docker Desktop

Docker runs Qdrant + OpenSearch + Postgres so you don't have to install
or manage them directly.

**Windows:**
https://docs.docker.com/desktop/install/windows-install/

After install, open Docker Desktop → Settings → Resources → set RAM to at
least 4 GB (the default 2 GB is too low for OpenSearch).

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in, then:
docker --version   # should print 26.x or newer
```

---

## Step 2 — Install Python 3.11+

**Windows:**
Download from https://www.python.org/downloads/ — tick "Add to PATH".

Or via the Microsoft Store: search "Python 3.11".

**Linux:**
```bash
sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip -y
```

Verify:
```bash
python --version   # Python 3.11.x or 3.12.x
```

---

## Step 3 — Install Tesseract (OCR for scanned PDFs)

Only needed if your documents are scans. Skip if you only have native PDFs
(born-digital, not photographed pages).

**Windows:**
Download the installer from:
https://github.com/UB-Mannheim/tesseract/wiki
→ tick "Add to PATH" during install.

**Linux:**
```bash
sudo apt install tesseract-ocr -y
```

Verify:
```bash
tesseract --version
```

---

## Step 4 — Clone the repo and create a virtual environment

```bash
git clone <repo-url> docintel
cd docintel

# Create a venv (keep project deps isolated from system Python)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

---

## Step 5 — Install CPU-only PyTorch first

Do this **before** `pip install -r requirements.txt`. If you let pip resolve
torch on its own it might pull the 2.5 GB CUDA wheel.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

This installs the ~250 MB CPU-only wheel. Takes 2–3 minutes on a good
connection.

---

## Step 6 — Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

This will take 5–10 minutes. The heavy downloads are:
- `sentence-transformers` + model weights (~22 MB for MiniLM, downloaded
  on first use)
- `pymupdf4llm` (fast, ~20 MB)
- `pdfplumber` (fast, ~5 MB)

---

## Step 7 — Configure your API key

```bash
cp .env.example .env
```

Open `.env` in any editor and set:
```
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

Everything else in `.env` is already tuned for your hardware. You don't
need to change anything else to get started.

---

## Step 8 — Start the infrastructure

```bash
docker compose up -d
```

First run downloads the images (~1.5 GB total). Subsequent starts are
instant. Wait about 30 seconds for OpenSearch to fully initialise, then:

```bash
# Verify everything is healthy:
docker compose ps
```

You should see `healthy` next to qdrant, opensearch, and postgres.

If OpenSearch shows `unhealthy` after 60 seconds, it usually means Docker
doesn't have enough RAM. Go to Docker Desktop → Settings → Resources and
raise to 4 GB.

---

## Step 9 — Run the quickstart

```bash
python scripts/quickstart.py \
    --pdf path/to/any_document.pdf \
    --ask "summarise the main findings"
```

You'll see:
```
[ingested]  doc_id=abc123...  pages=12
[ask]       retrieved 8 evidence items in 340ms
============================================================
ANSWER
============================================================
The document covers... [e0] ... [e1] ...

CITATIONS
[e0] annual_report.pdf  p.3  (text)   ...
[e1] annual_report.pdf  p.7  (table)  | Revenue | ...
```

---

## Performance expectations on your hardware

| Operation | Typical time |
|---|---|
| Ingest a 10-page PDF (native, no scans) | 2–5 seconds |
| Ingest a 50-page PDF | 10–25 seconds |
| Ingest a scanned page (OCR) | 3–8 seconds per page |
| Embed 100 text blocks | ~0.1 seconds (ONNX) |
| BM25 + dense retrieval + RRF | 50–200 ms |
| Claude API answer generation | 2–6 seconds |
| **Full query end-to-end** | **~3–8 seconds** |

For context: a GPU machine would do the same thing in 1–2 seconds. Totally
usable for development, demos, and small production workloads (<100 docs).

---

## RAM usage guide

Your 16 GB is enough, but be mindful:

| Thing | RAM |
|---|---|
| Docker (Qdrant + OpenSearch + Postgres) | ~1.8 GB |
| Python process (parser + embedder loaded) | ~600 MB |
| sentence-transformers model (MiniLM) | ~50 MB |
| Chrome / VS Code / other apps | varies |
| **Safe headroom** | ~10+ GB free |

If you switch to `BAAI/bge-base-en-v1.5` (better quality, 768-dim), the
model footprint rises to ~250 MB — still fine.

---

## Troubleshooting

**`torch` imports fail or says "no CUDA device"**
That's fine — CUDA not found is expected on a CPU-only machine. Torch
will fall back to CPU automatically. The warning is harmless.

**OpenSearch container exits immediately**
Docker RAM is too low. In Docker Desktop → Resources, set Memory to 4 GB.

**`pytesseract.pytesseract.TesseractNotFoundError`**
Tesseract binary is not on your PATH. Re-run the tesseract install step,
or set the path manually in your `.env`:
```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Embedding is slow (~40 sentences/sec instead of ~1500)**
The ONNX backend isn't loading. Check for `optimum[onnxruntime]` in your
venv: `pip show optimum`. If it's missing, run:
```bash
pip install "optimum[onnxruntime]"
```

**`ModuleNotFoundError: No module named 'docling'`**
That's expected — Docling is not in the CPU requirements. The parser is
set to `PARSER_PRIMARY=cpu` in your `.env`, so Docling is never imported.

---

## Want to run tests?

```bash
pytest tests/ -v
```

The fusion tests run without any infra (no Docker needed). All 5 should
pass in under a second.
