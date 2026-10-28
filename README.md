# Citation Faithfulness Checker

Open-source citation faithfulness checker: extracts claims, retrieves evidence, verifies via NLI. Fully open-source models and data (RAGBench, HoVer, SciFact).

## Quick Start (Streamlit UI)

```powershell
pip install -r requirements.txt
$env:PYTHONPATH="src"
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

### Docker (Optional)

```powershell
docker build -t citefaith -f docker/Dockerfile .
docker run -p 8000:8000 citefaith
```

---

## API (FastAPI)

```powershell
$env:PYTHONPATH="src"
uvicorn citefaith.api:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
- `POST /check` — JSON body `{"text": "...", "sources": ["..."]}`
- `POST /check/files` — multipart: `text_file` + `source_file`

---

## Deploy to Streamlit Cloud (Free, No Docker)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo → main branch → `streamlit_app.py`
4. Deploy (3 minutes)

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | Cross-encoder model |
| `RETRIEVER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `ENT_THRESHOLD` | `0.5` | Entailment threshold |
| `CONTRA_THRESHOLD` | `0.5` | Contradiction threshold |
| `CONTRA_MIN_SIM` | `0.5` | Min similarity for contradiction |
| `SIM_THRESHOLD` | `0.7` | Similarity fallback threshold |

---

## Performance (RAGBench, 12 domains)

| Metric | Value |
|--------|-------|
| Accuracy | 0.70 |
| Precision | 0.90 |
| Recall | 0.73 |
| F1 | 0.81 |

---

## Project Structure

```
src/citefaith/     # Core package (api, checker, nli, retriever, data, files)
scripts/           # eval, training, diagnostics
tests/             # smoke tests
data/              # RAGBench datasets (gitignored)
docker/            # Dockerfile, docker-compose.yml
.streamlit/        # Streamlit config
```

---

## License

MIT — see [LICENSE](LICENSE).