from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from .checker import CitationChecker
from .files import extract_text

app = FastAPI(title="Citation Faithfulness Checker")
_checker = None

def get_checker():
    global _checker
    if _checker is None:
        _checker = CitationChecker()
    return _checker

class CheckRequest(BaseModel):
    text: str
    sources: list[str]

@app.get("/")
def root():
    return {"name": "Citation Faithfulness Checker", "docs": "/docs", "endpoints": ["/check", "/check/files"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/check")
def check(req: CheckRequest):
    return get_checker().check(req.text, req.sources)

@app.post("/check/files")
async def check_files(
    text_file: UploadFile = File(..., description="File whose claims will be checked"),
    source_file: UploadFile = File(..., description="Source file to check against"),
):
    text = extract_text(text_file.filename, await text_file.read())
    source = extract_text(source_file.filename, await source_file.read())
    if not text.strip():
        raise HTTPException(400, "text_file produced no text")
    if not source.strip():
        raise HTTPException(400, "source_file produced no text")
    return get_checker().check(text, [source])
