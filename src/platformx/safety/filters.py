from enum import Enum
from typing import List, Dict, Any, Tuple


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


class ReasonCode(str, Enum):
    NON_MEDICAL = "non_medical"
    OUT_OF_SCOPE_CHEMISTRY = "out_of_scope_chemistry"
    PROCEDURAL_LAB = "procedural_lab"
    NO_EVIDENCE = "no_evidence"
    UNSAFE_INTENT = "unsafe_intent"
    ALLOWED = "allowed"


def _normalize_intent(intent: Any) -> str:
    # Accept either Enum or string-like intents; normalize to lowercase string
    try:
        return intent.value.lower()  # when intent is Enum-like
    except Exception:
        return str(intent).lower()


def evaluate_safety(query_intent: Any, retrieved_evidence: List[Dict[str, Any]], query_text: str) -> Dict[str, Any]:
    """Evaluate safety for a query + retrieved evidence deterministically.

    Returns a structured decision dict:
      { decision: Decision, reason: ReasonCode, details: { ... } }

    This gate is usable without any model and makes conservative, deterministic
    decisions. It favors explicit blocking over permissive behavior.
    """
    intent = _normalize_intent(query_intent)

    # 1) Basic intent policy: only allow informational/regulatory/safety intents
    allowed_intents = {"informational", "regulatory", "safety"}
    if intent not in allowed_intents:
        return {"decision": Decision.BLOCK, "reason": ReasonCode.NON_MEDICAL, "details": {"intent": intent}}

    # 2) Procedural lab safety: deterministic keyword check
    procedural_keywords = {"incubate", "mix", "pipette", "sterilize", "autoclave", "centrifuge", "culture", "prepare solution", "add reagent"}
    lowered = query_text.lower()
    for kw in procedural_keywords:
        if kw in lowered:
            return {"decision": Decision.BLOCK, "reason": ReasonCode.PROCEDURAL_LAB, "details": {"matched_keyword": kw}}

    # 3) Out-of-scope chemistry: block requests that clearly seek synthesis or illicit instructions
    out_of_scope_keywords = {"synthesize", "how to make", "how to synthesize", "manufacture", "produce"}
    for kw in out_of_scope_keywords:
        if kw in lowered:
            return {"decision": Decision.BLOCK, "reason": ReasonCode.OUT_OF_SCOPE_CHEMISTRY, "details": {"matched_keyword": kw}}

    # 4) Evidence presence: require at least one retrieved item with non-empty text
    if not retrieved_evidence or all(not (item.get("text") or "") for item in retrieved_evidence):
        return {"decision": Decision.BLOCK, "reason": ReasonCode.NO_EVIDENCE, "details": {"retrieved_count": len(retrieved_evidence)}}

    # 5) If all checks pass, allow
    return {"decision": Decision.ALLOW, "reason": ReasonCode.ALLOWED, "details": {"retrieved_count": len(retrieved_evidence)}}
