"""Train a classifier on cached NLI features; supports cross-domain (holdout config) evaluation.

Examples:
  python scripts/train_classifier.py --train data/cache_train.jsonl --test data/cache_test.jsonl
  python scripts/train_classifier.py --train data/cache_train.jsonl --holdout-configs finqa tatqa
"""
import argparse, json, os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from citefaith.features import features, FEATURE_NAMES

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def featurize(recs, ent_idx=1, contra_idx=0):
    X, y, cfg, ids = [], [], [], []
    for r in recs:
        ev = [(e[0], e[1], e[2]) for e in r["ev"]]
        X.append(features(r["claim"], ev, np.array(r["probs"], dtype=float), ent_idx, contra_idx))
        y.append(bool(r["gold"]))
        cfg.append(r.get("config"))
        ids.append(r["id"])
    return np.stack(X), np.array(y), cfg, ids

def report(name, y, pred):
    print(f"\n=== {name} ===")
    print(f"acc {accuracy_score(y, pred):.4f}  f1 {f1_score(y, pred):.4f}  n={len(y)}")
    print(classification_report(y, pred, target_names=["unsupported", "supported"]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/cache_train.jsonl")
    ap.add_argument("--test", default="data/cache_test.jsonl")
    ap.add_argument("--holdout-configs", nargs="*", default=[])
    ap.add_argument("--out", default="models/clf.joblib")
    args = ap.parse_args()

    tr = load(args.train)
    te = load(args.test) if os.path.exists(args.test) else []
    Xtr, ytr, ctr, _ = featurize(tr)
    print(f"train {len(tr)} claims from {len(set(ctr))} configs | test {len(te)} claims")

    if args.holdout_configs:
        mask = np.array([c not in args.holdout_configs for c in ctr])
        Xtr, ytr = Xtr[mask], ytr[mask]
        print(f"holding out configs: {args.holdout_configs} -> train on {len(ytr)} claims")

    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(Xtr, ytr)
    print("train acc:", round(accuracy_score(ytr, clf.predict(Xtr)), 4))

    if te:
        Xte, yte, cte, _ = featurize(te)
        pred = clf.predict(Xte)
        report("ALL TEST CONFIGS", yte, pred)
        for cfg in sorted(set(cte)):
            m = np.array([c == cfg for c in cte])
            print(f"  {cfg:12s} acc {accuracy_score(yte[m], pred[m]):.3f} f1 {f1_score(yte[m], pred[m]):.3f} n={m.sum()}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(clf, args.out)
    print("\nsaved", args.out)
    for n, w in sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {n:18s} {w:+.3f}")
