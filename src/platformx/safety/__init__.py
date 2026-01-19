"""Safety package for PlatformX.

Expose the high-level safety APIs: `evaluate_safety`, `RefusalEngine`, and
`assess_confidence`. Internals remain private to prevent accidental misuse.

Safety philosophy: Wrong answers can cause real harm. Silence is safer than
speculation. Refusal preserves trust. Traceability matters more than fluency.
Accuracy > coverage > speed.
"""

from .filters import evaluate_safety, Decision, ReasonCode
from .refusal import RefusalEngine
from .confidence import assess_confidence, ConfidenceLevel

__all__ = ["evaluate_safety", "Decision", "ReasonCode", "RefusalEngine", "assess_confidence", "ConfidenceLevel"]
