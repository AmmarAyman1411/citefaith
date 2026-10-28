"""Diagnose which verdicts cause false negatives / false positives."""
import argparse, json
from collections import Counter
from tqdm import tqdm
from citefaith.checker import CitationChecker
from citefaith.data import iter_examples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()
    c = CitationChecker()
    fn_verdicts = Counter()
    fp_verdicts = Counter()
    fn_scores = []
    n_ex = 0
    for ex in tqdm(iter_examples(), desc="diag"):
        res = c.check_example(ex)
        n_ex += 1
        for cl in res["claims"]:
            pred = cl["verdict"] == "supported"
            gold = bool(cl["gold_supported"])
            if pred and not gold:
                fp_verdicts[cl["verdict"]] += 1
            if (not pred) and gold:
                fn_verdicts[cl["verdict"]] += 1
                fn_scores.append(cl["score"])
        if n_ex >= args.limit:
            break
    print(json.dumps({
        "false_negative_verdicts": dict(fn_verdicts),
        "false_positive_verdicts": dict(fp_verdicts),
        "fn_mean_score": round(sum(fn_scores)/len(fn_scores), 4) if fn_scores else None,
        "fn_max_score": round(max(fn_scores), 4) if fn_scores else None,
    }, indent=2))

if __name__ == "__main__":
    main()
