# Citation Faithfulness Checker

Open-source citation faithfulness checker: extracts claims, retrieves evidence, verifies via NLI. Fully open-source models and data (RAGBench, HoVer, SciFact).

## Quick Start

```powershell
# Python 3.11+
pip install -r requirements.txt
$env:PYTHONPATH="src"
uvicorn citefaith.api:app --host 0.0.0.0 --port 8000
```

Test at `http://127.0.0.1:8000/docs`

### Docker

```powershell
# Build from repo root
docker build -t citefaith -f docker/Dockerfile .

# Run
docker run -p 8000:8000 citefaith

# Or with Compose
docker-compose up --build
```

---

## API

### POST /check (JSON)
```bash
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "sources": ["..."]}'
```

### POST /check/files (Multipart)
```bash
curl -X POST http://localhost:8000/check/files \
  -F "text_file=@claims.txt" -F "source_file=@sources.txt"
```

**Verdicts:** `supported` | `contradicted` | `neutral`

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

## Deploy to Hugging Face Spaces

1. Move Dockerfile to root: `Move-Item "docker\Dockerfile" "Dockerfile"`
2. Push to GitHub: `gh repo create citefaith --public --source=. --push`
3. Create HF Space → Docker → connect repo

HF builds in cloud (~10-15 min first build).

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
```

---

## License

MIT — see [LICENSE](LICENSE).