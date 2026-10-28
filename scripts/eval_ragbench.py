import argparse, json
from tqdm import tqdm
from citefaith.checker import CitationChecker
from citefaith.data import iter_examples

def doc_sentences(ex):
    out = []
    for doc_sents in (ex.get("documents_sentences") or []):
        out.extend(doc_sents)
    return out

def gold_labels(ex):
    labels = {}
    for info in ex.get("sentence_support_information", []):
        labels[info["response_sentence_key"]] = bool(info.get("fully_supported", False))
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-config", action="store_true")
    args = ap.parse_args()

    checker = CitationChecker()
    tp=fp=tn=fn=0
    by_cfg = {}
    examples=0
    for ex in tqdm(iter_examples(args.configs, args.split), desc="eval"):
        text = ex["response"]
        sources = ex["documents"]
        res = checker.check(text, sources)
        examples += 1
        if args.limit and examples >= args.limit * (len(args.configs) if args.configs else 12):
            break
        cfg = ex.get("_config")
        by_cfg.setdefault(cfg, [0,0,0,0])
        gold = gold_labels(ex)
        rs = ex["response_sentences"]
        for i, cl in enumerate(res["claims"]):
            if i >= len(rs): break
            pred = cl["verdict"] == "supported"
            gold_val = gold.get(rs[i][0], False)
            if pred and gold_val: tp+=1; by_cfg[cfg][0]+=1
            elif pred and not gold_val: fp+=1; by_cfg[cfg][1]+=1
            elif (not pred) and (not gold_val): tn+=1; by_cfg[cfg][2]+=1
            else: fn+=1; by_cfg[cfg][3]+=1

    tot = tp+tn+fp+fn
    acc = (tp+tn)/tot if tot else 0
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    out = {"examples": examples, "claim_acc": round(acc,4), "precision": round(prec,4),
           "recall": round(rec,4), "f1": round(f1,4), "tp":tp,"fp":fp,"tn":tn,"fn":fn}
    if args.per_config and by_cfg:
        out["per_config"] = {}
        for cfg,(t1,t2,t3,t4) in sorted(by_cfg.items()):
            n=t1+t2+t3+t4
            p = t1/(t1+t2) if (t1+t2) else 0
            r = t1/(t1+t4) if (t1+t4) else 0
            out["per_config"][cfg] = {"n": n, "acc": round((t1+t3)/n,4), "precision": round(p,4), "recall": round(r,4)}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
