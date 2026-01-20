from platformx.safety import evaluate_safety, RefusalEngine, assess_confidence

def test_evaluate_safety_allow():
    result = evaluate_safety("informational", [{"text": "evidence"}], "What is aspirin?")
    assert result["decision"].lower() == "allow"

def test_refusal_engine():
    engine = RefusalEngine()
    refusal = engine.make_refusal("no_evidence")
    assert "insufficient verified evidence" in refusal.message

def test_assess_confidence():
    evidence = [{"text": "foo bar", "score": 0.9}, {"text": "foo bar", "score": 0.8}]
    result = assess_confidence(evidence)
    assert result["level"]
