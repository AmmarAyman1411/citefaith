"""Cache NLI scores + retrieval sims for RAGBench so decision rules can be tuned offline.

Example:
  python scripts/cache_scores.py --split train --limit 4000 --out data/cache_train.jsonl
  python scripts/cache_scores.py --split test  --limit 2000 --out data/cache_test.jsonl
"""
import argparse, json
from tqdm import tqdm
from citefaith.checker import CitationChecker
from citefaith.data import iter_examples

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=1000)
ap.add_argument("--split", default=None)
ap.add_argument("--configs", nargs="*", default=None)
ap.add_argument("--out", default="data/score_cache.jsonl")
args = ap.parse_args()

c = CitationChecker()
out = open(args.out, "w", encoding="utf-8")
n = 0
for ex in tqdm(iter_examples(args.configs, args.split) if args.split else iter_examples(args.configs),
               desc="cache"):
    d_sents = []
    for doc in ex["documents_sentences"]:
        d_sents.extend(doc)
    c.retriever.build(d_sents)
    gold = {i["response_sentence_key"]: bool(i.get("fully_supported", False))
            for i in ex["sentence_support_information"]}
    for key, sent in ex["response_sentences"]:
        ev = c.retriever.query(sent, k=c.k)
        # NLI expects (premise=evidence, hypothesis=claim)
        probs = c.nli.scores([(e[1], sent) for e in ev])
        out.write(json.dumps({
            "id": ex["id"], "config": ex.get("_config"), "split": args.split, "key": key,
            "claim": sent, "gold": gold.get(key),
            "ev": [[e[0], e[1], round(float(e[2]), 5)] for e in ev],
            "probs": [[round(float(x), 5) for x in row] for row in probs],
            "claim_nums": sorted(c.nli._nums_with_units(sent)),
        }, ensure_ascii=False) + "\n")
        n += 1
        if n >= args.limit:
            break
    if n >= args.limit:
        break
out.close()
print(f"cached {n} claims -> {args.out}")
