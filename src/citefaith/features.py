"""Feature extraction for the learned faithfulness classifier."""
import re
import numpy as np

NUM_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(kg|kgs|g|mg|cm|mm|km|m|meters?|centimeters?|millimeters?|miles?|mile\b|"
    r"ft|feet|inches|inch|yards?|seconds?|minutes?|hours?|days?|weeks?|months?|years?|%)\b",
    re.IGNORECASE,
)

def ev_nums(text):
    return {(m.group(1), m.group(2).lower()) for m in NUM_RE.finditer(text)}

def features(claim, evidences, probs, ent_idx=1, contra_idx=0):
    """evidences: list of (key, text[, sim]); probs: (k,3) nli probabilities."""
    probs = np.asarray(probs, dtype=float)
    sims = np.array([e[2] if len(e) > 2 else 0.0 for e in evidences], dtype=float)
    if len(probs) == 0:
        return np.zeros(18, dtype=float)
    ent, contra, neu = probs[:, ent_idx], probs[:, contra_idx], probs[:, 2 - ent_idx - contra_idx]
    cn = ev_nums(claim)
    mism = np.array([any(cu == eu and c != e for c, cu in cn for e, eu in ev_nums(x[1]))
                     for x in evidences], dtype=float)
    return np.array([
        ent.max(), ent.mean(), ent[0],
        contra.max(), contra.mean(), contra[0],
        neu.max(), neu.mean(),
        sims.max(), sims.mean(), sims[0],
        mism.max(),
        len(claim.split()), len(cn),
        float(np.argmax(ent) == 0),
        float(np.argmax(contra) == 0),
        ent.max() - contra.max(),
        sims.max() - contra.max(),
    ], dtype=float)
