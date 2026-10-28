import re
from .retriever import Retriever
from .nli import NLIModel

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

class CitationChecker:
    def __init__(self, k=3):
        self.retriever = Retriever()
        self.nli = NLIModel()
        self.k = k

    def check(self, text, sources):
        sents = split_sentences(text)
        corpus = []
        for si, src in enumerate(sources):
            for sj, s in enumerate(split_sentences(src)):
                corpus.append((f"s{si}_{sj}", s))
        self.retriever.build(corpus)
        claims = []
        unsupported = 0
        for sent in sents:
            ev = self.retriever.query(sent, k=self.k)
            verdict, score, best = self.nli.verify(sent, ev)
            if verdict != "supported":
                unsupported += 1
            claims.append({
                "claim": sent,
                "verdict": verdict,
                "score": score,
                "evidence": best[1] if best else None,
                "evidence_key": best[0] if best else None,
            })
        return {
            "claims": claims,
            "faithful": unsupported == 0,
            "unsupported_count": unsupported,
            "total_claims": len(claims),
        }
