import os
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, model_name=None):
        model_name = model_name or os.getenv("RETRIEVER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)
        self.corpus = []  # list of (key, text)
        self.emb = None

    def build(self, corpus):
        self.corpus = list(corpus)
        texts = [t for _, t in self.corpus]
        self.emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def query(self, text, k=3):
        if not self.corpus:
            return []
        q = self.model.encode([text], normalize_embeddings=True)[0]
        sims = self.emb @ q
        idx = np.argsort(sims)[::-1][:k]
        return [(self.corpus[i][0], self.corpus[i][1], float(sims[i])) for i in idx]
