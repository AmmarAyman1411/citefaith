import io

def extract_text(filename: str, raw: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        import pymupdf
        doc = pymupdf.open(stream=raw, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in d.paragraphs)
    if name.endswith(".txt") or name.endswith(".md") or name.endswith(".rst"):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")
    # fallback
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")
