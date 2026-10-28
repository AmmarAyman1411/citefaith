"""Evaluate the faithfulness checker on your own labeled examples.

Input: JSONL file where each line is:
  {"text": "...", "sources": ["...", "..."], "gold": [true, false, ...]}
`gold` is a per-sentence label matching the order of sentences in `text`.

Usage:
  $env:PYTHONPATH="src"; python scripts/eval_custom.py --file data/my_examples.jsonl
"""
import argparse, json
from citefaith.checker import CitationChecker
from citefaith.data import split_sentences

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    checker = CitationChecker()
    tp=fp=tn=fn=0
    faithful_exact=0
    n=0
    examples=[]
    with open(args.file, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            ex=json.loads(line)
            sents=split_sentences(ex["text"])
            gold=ex["gold"]
            if len(gold)!=len(sents):
                print(f"skip {ex['id'] if 'id' in ex else n}: gold length {len(gold)} != sentences {len(sents)}")
                continue
            res=checker.check(ex["text"], ex["sources"])
            examples.append({"id": ex.get("id", n), "claims": res["claims"], "gold": gold})
            for cl,g in zip(res["claims"], gold):
                pred=cl["verdict"]=="supported"
                if pred and g: tp+=1
                elif pred and not g: fp+=1
                elif (not pred) and (not g): tn+=1
                else: fn+=1
            if all((cl["verdict"]=="supported")==g for cl,g in zip(res["claims"], gold)):
                faithful_exact+=1
            n+=1
            if args.limit and n>=args.limit: break

    tot=tp+tn+fp+fn
    acc=(tp+tn)/tot if tot else 0
    prec=tp/(tp+fp) if (tp+fp) else 0
    rec=tp/(tp+fn) if (tp+fn) else 0
    f1=2*prec*rec/(prec+rec) if (prec+rec) else 0
    print(json.dumps({
        "examples": n, "claims": tot,
        "accuracy": round(acc,4), "precision": round(prec,4),
        "recall": round(rec,4), "f1": round(f1,4),
        "tp":tp,"fp":fp,"tn":tn,"fn":fn,
        "faithful_exact_match": round(faithful_exact/n,4) if n else 0,
    }, indent=2))

    print("\nMisclassified claims:")
    for ex in examples:
        for cl,g in zip(ex["claims"], ex["gold"]):
            pred=cl["verdict"]=="supported"
            if pred!=g:
                print(f"  id={ex['id']} pred={cl['verdict']} gold={'supported' if g else 'unsupported'} "
                      f"score={cl['score']:.3f} claim={cl['claim'][:90]}")

if __name__=="__main__":
    main()
