"""Zero-framework chunking: turn raw seed data into natural-language chunks ready for embedding."""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

CATEGORY_EXPANSIONS = {
    "tws": "true wireless earbud (tws)",
}


def expand_category(category: str) -> str:
    return CATEGORY_EXPANSIONS.get(category, category)


def product_to_text(product: dict) -> str:
    specs = product["specs"]
    spec_bits = []
    if "battery_hours" in specs:
        spec_bits.append(f"{specs['battery_hours']}h battery life")
    if "anc" in specs:
        spec_bits.append("has ANC (active noise cancellation)" if specs["anc"] else "no ANC")
    if "ip_rating" in specs:
        spec_bits.append(f"{specs['ip_rating']} water/dust resistance")
    if "bluetooth" in specs:
        spec_bits.append(f"Bluetooth {specs['bluetooth']}")

    return (
        f"{product['name']} is a {expand_category(product['category'])} priced at Rs. {product['price']}. "
        f"{product['description']} "
        f"Specs: {', '.join(spec_bits)}."
    )


def faq_to_text(faq: dict) -> str:
    return f"Q: {faq['question']} A: {faq['answer']}"


def markdown_table_to_sentences(table_lines: list[str]) -> str:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in table_lines
        if line.strip() and not set(line.strip().strip("|").replace("|", "").strip()) <= set("-: ")
    ]
    header, *data_rows = rows
    sentences = []
    for row in data_rows:
        subject = row[0]
        details = ", ".join(f"{h.lower()}: {v}" for h, v in zip(header[1:], row[1:]))
        sentences.append(f"{subject} — {details}.")
    return " ".join(sentences)


BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
BULLET_RE = re.compile(r"^(?:[-*]|\d+\.)\s+")


def clean_line(line: str) -> str:
    line = BOLD_RE.sub(r"\1", line.strip())
    return BULLET_RE.sub("", line).strip()


def section_body_to_text(body_lines: list[str]) -> str:
    table_lines = [l for l in body_lines if l.strip().startswith("|")]
    prose_lines = [clean_line(l) for l in body_lines if l.strip() and not l.strip().startswith("|")]

    prose_text = ". ".join(l.rstrip(".:") for l in prose_lines if l)
    if prose_text and not prose_text.endswith("."):
        prose_text += "."

    parts = [prose_text] if prose_text else []
    if table_lines:
        parts.append(markdown_table_to_sentences(table_lines))
    return " ".join(parts)


def chunk_policy_markdown(text: str) -> list[str]:
    lines = text.strip().splitlines()
    title = lines[0].lstrip("#").strip()

    body = "\n".join(lines[1:])
    raw_sections = re.split(r"\n##\s+", body)[1:]  # skip pre-header intro (tagline)

    chunks = []
    for section in raw_sections:
        section_lines = section.splitlines()
        header = section_lines[0].strip()
        text_body = section_body_to_text(section_lines[1:])
        chunks.append(f"{title} — {header}: {text_body}")
    return chunks


def build_all_chunks() -> list[dict]:
    chunks = []

    products = json.loads((DATA_DIR / "seed" / "products.json").read_text())
    for p in products:
        chunks.append({
            "id": f"product-{p['sku']}",
            "text": product_to_text(p),
            "metadata": {"type": "product", "sku": p["sku"], "name": p["name"], "category": p["category"]},
        })

    faqs = json.loads((DATA_DIR / "seed" / "faqs.json").read_text())
    for f in faqs:
        chunks.append({
            "id": f"faq-{f['faq_key']}",
            "text": faq_to_text(f),
            "metadata": {"type": "faq", "faq_key": f["faq_key"], "topic": f["topic"]},
        })

    for filename in ["warranty-policy.md", "safety-sop.md", "return-policy.md"]:
        policy_text = (DATA_DIR / "policies" / filename).read_text()
        doc_title = policy_text.strip().splitlines()[0].lstrip("#").strip()
        for i, section_text in enumerate(chunk_policy_markdown(policy_text)):
            chunks.append({
                "id": f"policy-{filename}-{i}",
                "text": section_text,
                "metadata": {"type": "policy", "doc": doc_title, "source_file": filename},
            })

    return chunks


if __name__ == "__main__":
    chunks = build_all_chunks()
    print(f"Built {len(chunks)} chunks total.")

    out_path = DATA_DIR / "processed" / "chunks.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"Wrote chunks to {out_path}")
