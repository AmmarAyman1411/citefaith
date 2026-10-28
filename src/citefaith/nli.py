import os
import re
import numpy as np
from sentence_transformers import CrossEncoder

# units we trust for numeric mismatch detection
UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(kg|cm|m|%|years?)\b", re.IGNORECASE
)

class NLIModel:
    def __init__(self, model_name=None):
        model_name = model_name or os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
        self.model = CrossEncoder(model_name)
        self.id2label = self.model.model.config.id2label
        self.sim_threshold = float(os.getenv("SIM_THRESHOLD", "0.7"))
        self.ent_threshold = float(os.getenv("ENT_THRESHOLD", "0.5"))
        self.contra_threshold = float(os.getenv("CONTRA_THRESHOLD", "0.5"))
        # a contradiction only counts if the contradicting sentence is about the same thing
        self.contra_min_sim = float(os.getenv("CONTRA_MIN_SIM", "0.5"))
        self.ent_idx, self.contra_idx, self.neu_idx = 1, 0, 2
        for idx, lab in self.id2label.items():
            l = str(lab).lower()
            if "entail" in l:
                self.ent_idx = int(idx)
            elif "contrad" in l:
                self.contra_idx = int(idx)
            elif "neutral" in l:
                self.neu_idx = int(idx)

    def scores(self, pairs):
        raw = self.model.predict(pairs)
        raw = np.atleast_2d(raw)
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _nums_with_units(text):
        return {(m.group(1), m.group(2).lower()) for m in UNIT_RE.finditer(text)}

    def verify(self, claim, evidences):
        """evidences: list of (key, text) or (key, text, sim)."""
        if not evidences:
            return "neutral", 0.0, None
        # NLI expects (premise, hypothesis): premise = evidence text, hypothesis = claim
        pairs = [(ev[1], claim) for ev in evidences]
        probs = self.scores(pairs)
        ent = probs[:, self.ent_idx]
        contra = probs[:, self.contra_idx]
        sims = np.array([ev[2] if len(ev) > 2 else 0.0 for ev in evidences])
        claim_nu = self._nums_with_units(claim)
        num_mismatch = np.zeros(len(evidences), dtype=bool)
        for i, ev in enumerate(evidences):
            ev_nu = self._nums_with_units(ev[1])
            num_mismatch[i] = any(cu == eu and cn != en
                                  for cn, cu in claim_nu for en, eu in ev_nu)

        # support: strong entailment OR topical paraphrase via similarity
        soft_support = (sims >= self.sim_threshold) & (contra < 0.3) & (~num_mismatch)
        support_idx = np.where((ent >= self.ent_threshold) | soft_support)[0]
        # contradiction: topical NLI contradiction OR numeric mismatch on a topical sentence
        contra_ok = (contra >= self.contra_threshold) & (sims >= self.contra_min_sim)
        contra_idx = np.where(contra_ok | (num_mismatch & (sims >= self.contra_min_sim)))[0]

        if len(support_idx) and (not len(contra_idx) or ent[support_idx].max() >= contra[contra_idx].max()):
            best = int(support_idx[np.argmax(ent[support_idx])])
            score = max(float(ent[best]), float(sims[best]))
            return "supported", score, evidences[best]
        if len(contra_idx):
            b = int(contra_idx[np.argmax(contra[contra_idx] + num_mismatch[contra_idx])])
            return "contradicted", float(contra[b]), evidences[b]

        best = int(np.argmax(sims)) if len(sims) else 0
        return "neutral", float(sims[best]), evidences[best]
