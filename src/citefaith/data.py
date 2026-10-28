from datasets import load_from_disk
from pathlib import Path

DATA_ROOT = Path("data/ragbench")

def load_ragbench(config: str, split: str = "train"):
    return load_from_disk(str(DATA_ROOT / config))[split]

def iter_examples(configs=None, split="train"):
    if configs is None:
        configs = [p.name for p in DATA_ROOT.iterdir() if p.is_dir()]
    for cfg in configs:
        ds = load_from_disk(str(DATA_ROOT / cfg))[split]
        for ex in ds:
            ex["_config"] = cfg
            yield ex

def doc_sentences(ex):
    out = []
    for doc_sents in (ex.get("documents_sentences") or []):
        for key, sent in doc_sents:
            out.append((key, sent))
    return out

def response_sentences(ex):
    return [(k, s) for k, s in ex["response_sentences"]]

def gold_labels(ex):
    # map response sentence key -> fully_supported bool
    labels = {}
    for info in ex["sentence_support_information"]:
        labels[info["response_sentence_key"]] = bool(info.get("fully_supported", False))
    return labels
