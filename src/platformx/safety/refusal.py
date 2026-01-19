from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Refusal:
    message: str
    reason: str
    details: Dict[str, Any]


class RefusalEngine:
    """Produce calm, professional, deterministic refusals.

    Refusals are considered a successful safety outcome. They must never
    speculate or provide procedural alternatives that bypass safety.
    """

    def __init__(self) -> None:
        pass

    def make_refusal(self, reason_code: str, details: Dict[str, Any] | None = None) -> Refusal:
        details = details or {}
        # Standardized, non-suggestive refusal language
        message_map = {
            "no_evidence": "I cannot provide an answer because there is insufficient verified evidence in the available sources.",
            "procedural_lab": "I cannot assist with procedural laboratory instructions.",
            "out_of_scope_chemistry": "I cannot assist with requests to synthesize or manufacture chemical substances.",
            "non_medical": "This request is outside the permitted medical or regulatory informational scope.",
            "unsafe_intent": "I cannot assist with that request because it is unsafe.",
        }

        msg = message_map.get(reason_code, "I cannot comply with this request.")
        return Refusal(message=msg, reason=reason_code, details=details)
