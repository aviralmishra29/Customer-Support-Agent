# Ingesting other data formats into RAG (beyond what we built)

Our project only ever touched plain-text-friendly sources — JSON records and Markdown docs. Real-world RAG systems ingest far more format variety. This note catalogs the common ones, how each gets converted into something embeddable, and how chunking strategy changes per format. The end-to-end shape never changes (chunk → embed → store → retrieve → generate) — what changes is *how you get to a chunkable, embeddable representation* in the first place.

---

## Text-based documents

### Plain text / Markdown / HTML source snippets (what we already built)
Already covered in [[01-rag-pipeline-and-chunking]] — structural chunking on natural boundaries (headers, list items), converting tables/bold-markup to plain sentences before embedding. Baseline case everything else compares against.

### PDF
PDFs are a rendering format, not a structured-text format — "text" and "layout" are two different problems.
- **Text-layer PDFs** (exported from Word/Google Docs, most "born-digital" PDFs): a library like `PyMuPDF` (`fitz`), `pdfplumber`, or the heavier `unstructured` package extracts the actual text per page. Chunk by detected headings if the PDF has a recoverable outline/bookmark structure, otherwise fall back to paragraph or fixed-size splitting per page.
- **Scanned/image PDFs** (no real text layer, e.g. a photographed contract): there is no text to extract — the page is pixels. Requires **OCR** (Optical Character Recognition — Tesseract is the standard open-source engine, cloud alternatives like Azure Document Intelligence / Google Document AI are more accurate but not local) to convert the image into text *before* anything else can happen. OCR quality directly caps everything downstream — bad OCR means every later step (chunk, embed, retrieve) is working from corrupted source text. This is the same "garbage in, garbage out" problem from our very first RAG lesson, just relocated one step earlier in the pipeline.
- **Tables and multi-column layouts** inside PDFs are a known hard problem — naive text extraction often interleaves columns or loses table structure entirely (a 3-column table might extract as one garbled row). Production systems use layout-aware parsers specifically built for this (`unstructured`'s layout models, or commercial document-intelligence APIs) rather than plain text extraction.

### Word documents (DOCX)
Easier than PDF because DOCX keeps real, machine-readable structure. `python-docx` exposes heading levels/styles, paragraphs, and tables directly — no OCR or layout-guessing needed. Chunk by walking the document's actual heading hierarchy (Heading 1 → Heading 2 → paragraphs under it), the same structural-boundary idea we used for our Markdown policies, just reading DOCX's own style metadata instead of `##` syntax.

### PowerPoint (PPTX)
Structurally simpler than DOCX in one way: **one slide is usually one natural chunk** — a self-contained unit of a single idea, similar to how one product JSON record was one natural chunk for us. `python-pptx` extracts slide text, titles, and speaker notes. Complication: slides often lean on embedded images/charts/diagrams that carry meaning no text extraction will capture — for those, see the image-captioning approach below, applied per-slide.

### Email (.eml/.msg)
Has clean structured metadata (sender, recipients, subject, date) plus a body — extract both, since the metadata is often exactly what a user would filter/search by (similar to our chunk `metadata` fields). The real complication is **thread noise**: a reply chain often includes the entire quoted history of previous messages appended below the new content. Naive extraction chunks the same quoted text repeatedly across every message in a thread. Production email ingestion specifically strips quoted-reply blocks and signatures before chunking.

### Web pages (HTML)
Raw HTML is full of markup noise irrelevant to meaning — `<nav>`, `<script>`, `<footer>`, ads, cookie banners. A parser like `BeautifulSoup` strips tags and non-content boilerplate first, ideally targeting just the main content region (`<article>`, `<main>`, or heuristic "biggest text block" detection). Once you have clean text, chunk on the page's own heading tags (`<h1>`, `<h2>`, ...) — same structural-boundary principle again, just reading HTML headings instead of Markdown ones.

### Code files
Different enough to call out separately: chunking on line-count or character-count regularly slices a function in half, destroying meaning (a chunk with only the first three lines of a function is nearly useless). Real code-RAG systems chunk on **syntactic boundaries** using an AST (Abstract Syntax Tree) parser — one chunk per function or class, so each chunk is a complete, coherent unit. General-purpose text embedding models work adequately on code; dedicated code-embedding models (e.g. `CodeBERT`-family) exist and do measurably better by being trained specifically on code syntax and semantics.

---

## Structured / tabular data

### CSV / Excel (XLSX)
Mechanically identical to our `products.json` case: each **row** is one record, each **column** is one field. Convert each row into a natural-language sentence using column headers as field names — exactly what `product_to_text()` did, just reading a CSV/XLSX row instead of a JSON object. XLSX adds complications JSON doesn't have: multiple sheets (treat each meaningfully, don't blindly concatenate), merged cells, and formulas (extract the *computed value*, not the formula text — the formula itself has no semantic meaning to embed).

### JSON / API responses (what we already built)
Covered in [[01-rag-pipeline-and-chunking]] — one object = one chunk, converted to natural language before embedding (never embed raw JSON syntax).

### Relational/NoSQL databases as a live source
Not a file format, but a common real ingestion source: rather than a one-time export, production systems run a periodic (or event-triggered) job that pulls rows/documents and converts them the same way as CSV/JSON above. The important judgment call, same one from [[01-rag-pipeline-and-chunking]]: **not every column/field belongs in the vector DB** — narrative/descriptive fields (a product description) are RAG-worthy; transactional/exact-match fields (price, stock count, order status) stay queried directly from the source database via tool calls, never embedded.

---

## Multi-modal data (not text at all)

### Images
The one genuinely different case, not just "another parser": a text embedding model like `nomic-embed-text` cannot accept an image as input at all — there's no text to extract or feed it. Two real approaches:
1. **Multi-modal embedding model** (CLIP and open variants like OpenCLIP) — trained on millions of (image, caption) pairs so images and text share one vector space; a photo and its matching description land close together, and two visually similar photos land close together too. Embed product photos at ingestion, embed the user's uploaded photo at query time, do nearest-neighbor search exactly like our text pipeline — same mechanism, different embedding model and different input type.
2. **Caption-then-embed** — use a vision-capable model (a VLM) to describe the image in plain text first, then embed that description with a normal text embedding model. Simpler to bolt onto an existing text-only pipeline, but lossy — visual detail that doesn't make it into the caption is gone before embedding ever happens.

Locally, on our own stack: Ollama supports some vision-capable models (e.g. `llava`), making caption-then-embed buildable without violating the local-only constraint. True CLIP-scale multi-modal embedding is also runnable locally, just heavier on CPU-only hardware than anything we've built so far.

### Audio
Two different goals, two different techniques:
- **Search by spoken content** (e.g. "find the part of this podcast where they discuss X"): transcribe first with an ASR (Automatic Speech Recognition) model like Whisper, then treat the transcript as plain text — chunk and embed exactly like any other text source. Chunking is usually by timestamp segments (e.g. every 30 seconds, or per detected speaker turn) so each chunk can link back to a specific moment in the audio.
- **Search by sound/acoustic similarity** (e.g. "find similar-sounding music," speaker identification): needs an audio-specific embedding model trained directly on waveforms/spectrograms, not a transcript — analogous to how images need a vision-specific embedding model, not a caption.

### Video
Combines both of the above, plus a visual channel: extract the audio track and transcribe it (as above), extract representative frames and treat them as images (as above), and chunk the result by scene or time segment so a "chunk" corresponds to a coherent moment in the video rather than an arbitrary time slice. The most complete video-RAG systems fuse all three signals (transcript text, frame visuals, scene boundaries) rather than relying on just one.

---

## Quick reference

| Format | Extraction tool | Natural chunk unit | Embedding approach |
|---|---|---|---|
| Markdown/plain text | none needed | section (heading-bounded) | text embedding |
| JSON/API records | none needed | one object/record | text embedding (after → sentence) |
| PDF (text layer) | PyMuPDF/pdfplumber | page or detected section | text embedding |
| PDF (scanned) | OCR (Tesseract) → text | page or detected section | text embedding |
| DOCX | python-docx | heading-bounded section | text embedding |
| PPTX | python-pptx | one slide | text embedding |
| Email | email parser | one message (dequoted) | text embedding |
| HTML/web | BeautifulSoup | heading-bounded section | text embedding |
| Code | AST parser | one function/class | text or code-specific embedding |
| CSV/XLSX | pandas/openpyxl | one row | text embedding (after → sentence) |
| SQL/NoSQL DB | periodic export job | one row/document | text embedding (after → sentence) |
| Images | — | one image | multi-modal embedding (CLIP) or caption→text embedding |
| Audio (content search) | ASR (Whisper) → text | time/speaker segment | text embedding |
| Audio (sound search) | — | time segment | audio-specific embedding |
| Video | ASR + frame extraction | scene/time segment | fused text + image (+ audio) embeddings |
