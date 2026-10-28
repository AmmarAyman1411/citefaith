"""Sweep decision-rule thresholds against cached NLI scores (fast, no model needed)."""
import argparse, json, re
import numpy as np

UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(kg|kgs|g|milligrams?|mg|cm|mm|km|m|meters?|centimeters?|millimeters?|"
    r"kilometers?|miles?|mile\b|ft|feet|foot|inches|inch|in\b|yards?|seconds?|minutes?|hours?|"
    r"days?|weeks?|months?|years?|yr\b|%|percent|dollars?|usd|euros?|eur|pounds?|lbs?)\b",
    re.IGNORECASE,
)

def ev_nums(text):
    return {(m.group(1), m.group(2).lower()) for m in UNIT_RE.finditer(text)}

def decide(rec, sim_thr, ent_thr, contra_thr, contra_min_sim):
    ev = rec["ev"]
    probs = np.array(rec["probs"], dtype=float)
    sims = np.array([e[2] for e in ev], dtype=float)
    claim = rec["claim"]
    cn = ev_nums(claim)
    num_mismatch = np.array([any(cu == eu and c != e for c, cu in cn for e, eu in ev_nums(x[1])) for x in ev])
    ent = probs[:, 1]
    contra = probs[:, 0]
    soft = (sims >= sim_thr) & (contra < 0.3) & (~num_mismatch)
    support_idx = np.where((ent >= ent_thr) | soft)[0]
    contra_ok = (contra >= contra_thr) & (sims >= contra_min_sim)
    contra_idx = np.where(contra_ok | (num_mismatch & (sims >= contra_min_sim)))[0]
    if len(support_idx) and (not len(contra_idx) or ent[support_idx].max() >= contra[contra_idx].max()):
        return "supported"
    if len(contra_idx):
        return "contradicted"
    return "neutral"

def evaluate(recs, **kw):
    tp=fp=tn=fn=0
    by_id={}
    for r in recs:
        pred = decide(r, **kw) == "supported"
        gold = bool(r["gold"])
        by_id.setdefault(r["id"], []).append((pred, gold))
        if pred and gold: tp+=1
        elif pred and not gold: fp+=1
        elif (not pred) and (not gold): tn+=1
        else: fn+=1
    ex_ok = sum(1 for v in by_id.values() if all(p == g for p, g in v))
    acc = (tp+tn)/max(1,tp+tn+fp+fn)
    prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
    f1 = 2*prec*rec/max(1e-9, prec+rec)
    return {"acc":round(acc,4),"prec":round(prec,4),"rec":round(rec,4),"f1":round(f1,4),
            "tp":tp,"fp":fp,"tn":tn,"fn":fn,"ex_acc":round(ex_ok/max(1,len(by_id)),4)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/score_cache.jsonl")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.cache, encoding="utf-8")]
    print(f"{len(recs)} claims\n")
    base = dict(sim_thr=0.7, ent_thr=0.5, contra_thr=0.5, contra_min_sim=0.5)
    print("BASELINE:", json.dumps(evaluate(recs, **base)))
    grid = []
    for sim in [0.6,0.65,0.7,0.75]:
        for cms in [0.0,0.4,0.5,0.6]:
            for ct in [0.5,0.6,0.7]:
                r = evaluate(recs, sim_thr=sim, ent_thr=0.5, contra_thr=ct, contra_min_sim=cms)
                grid.append(((sim,0.5,ct,cms), r))
    grid.sort(key=lambda x: -x[1]["f1"])
    print("\nTOP 8 BY F1:")
    for params, r in grid[:8]:
        print(f"  sim_thr={params[0]} ent_thr={params[1]} contra_thr={params[2]} contra_min_sim={params[3]}")
        print(f"    {json.dumps(r)}")
