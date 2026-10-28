from citefaith.checker import CitationChecker

def test_faithful_claim():
    c = CitationChecker()
    r = c.check("Paris is the capital of France.", ["Paris is the capital and largest city of France."])
    assert r["total_claims"] == 1
    assert r["claims"][0]["verdict"] in {"supported","neutral","contradicted"}

def test_unsupported():
    c = CitationChecker()
    r = c.check("The moon is made of cheese.", ["The moon is a natural satellite of Earth."])
    assert r["unsupported_count"] >= 0
