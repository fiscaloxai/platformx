
"""
Safety filters for PlatformX: configurable, extensible, and domain-agnostic.

Provides:
- FilterRule: Configurable rule dataclass
- SafetyFilter, KeywordFilter, RegexFilter, PIIFilter, IntentFilter: Extensible filter classes
- SafetyFilterChain: Priority-ordered filter chain
- create_default_filter_chain: Factory for domain-specific defaults
- evaluate_safety: Backward-compatible API using the new system

How to use:
    # Create a filter chain with custom rules
    chain = SafetyFilterChain()
    chain.add_rule(FilterRule(rule_id="toxicity", name="Toxicity", description="Block toxic", rule_type="keyword", patterns=["kill"], action=Decision.BLOCK))
    result = chain.check("kill all", {})

    # Load/save rules from JSON
    chain.save_rules_to_file("rules.json")
    chain.load_rules_from_file("rules.json")

    # Use domain-specific defaults
    chain = create_default_filter_chain("pharma")

    # Short-circuit: check() returns on first block; check_all() returns all block decisions
"""
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
import logging
import json

logger = logging.getLogger("platformx.safety.filters")


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
    PII_DETECTED = "pii_detected"
    CUSTOM_RULE = "custom_rule"
    TOXICITY = "toxicity"
    OFF_TOPIC = "off_topic"


def _normalize_intent(intent: Any) -> str:
    # Accept either Enum or string-like intents; normalize to lowercase string
    try:
        return intent.value.lower()  # when intent is Enum-like
    except Exception:
        return str(intent).lower()

@dataclass
class FilterRule:
    """A single configurable filter rule."""
    rule_id: str
    name: str
    description: str
    rule_type: str  # e.g., "keyword", "regex", "intent", "custom"
    pattern: Optional[str] = None
    patterns: List[str] = field(default_factory=list)
    action: Decision = Decision.BLOCK
    reason_code: ReasonCode = ReasonCode.CUSTOM_RULE
    enabled: bool = True
    case_sensitive: bool = False
    priority: int = 100

class SafetyFilter(ABC):
    """Abstract base class for safety filters."""
    @abstractmethod
    def check(self, query_text: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if content should be filtered.
        Returns None if passed, or dict with decision/reason/details if blocked.
        """
        pass

    @abstractmethod
    def filter_id(self) -> str:
        """Return unique identifier for this filter."""
        pass

class KeywordFilter(SafetyFilter):
    """Filter based on keyword matching."""
    def __init__(self, rule: FilterRule):
        self.rule = rule
        self.patterns = list(rule.patterns)
        if rule.pattern:
            self.patterns.append(rule.pattern)
        if not rule.case_sensitive:
            self.patterns = [p.lower() for p in self.patterns]

    def check(self, query_text: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = query_text if self.rule.case_sensitive else query_text.lower()
        for pattern in self.patterns:
            if pattern and pattern in text:
                return {
                    "decision": self.rule.action,
                    "reason": self.rule.reason_code,
                    "details": {"matched_pattern": pattern, "rule_id": self.rule.rule_id},
                }
        return None

    def filter_id(self) -> str:
        return f"keyword:{self.rule.rule_id}"

class RegexFilter(SafetyFilter):
    """Filter based on regex pattern matching."""
    def __init__(self, rule: FilterRule):
        self.rule = rule
        self.compiled_patterns = []
        flags = 0 if rule.case_sensitive else re.IGNORECASE
        patterns = list(rule.patterns)
        if rule.pattern:
            patterns.append(rule.pattern)
        for pat in patterns:
            try:
                self.compiled_patterns.append(re.compile(pat, flags=flags))
            except Exception as e:
                logger.warning(f"Regex compilation failed for pattern '{pat}': {e}")

    def check(self, query_text: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for regex in self.compiled_patterns:
            m = regex.search(query_text)
            if m:
                return {
                    "decision": self.rule.action,
                    "reason": self.rule.reason_code,
                    "details": {"matched_pattern": regex.pattern, "groups": m.groups(), "rule_id": self.rule.rule_id},
                }
        return None

    def filter_id(self) -> str:
        return f"regex:{self.rule.rule_id}"

class PIIFilter(SafetyFilter):
    """Filter for detecting Personally Identifiable Information."""
    def __init__(self, detect_emails=True, detect_phones=True, detect_ssn=True, detect_credit_cards=True, custom_patterns: Dict[str, str] = None):
        self._patterns = {}
        if detect_emails:
            self._patterns["email"] = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        if detect_phones:
            self._patterns["phone"] = re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        if detect_ssn:
            self._patterns["ssn"] = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        if detect_credit_cards:
            self._patterns["credit_card"] = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
        if custom_patterns:
            for k, v in custom_patterns.items():
                try:
                    self._patterns[k] = re.compile(v)
                except Exception as e:
                    logger.warning(f"Custom PII regex failed for {k}: {e}")

    def check(self, query_text: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for typ, regex in self._patterns.items():
            if regex.search(query_text):
                return {
                    "decision": Decision.BLOCK,
                    "reason": ReasonCode.PII_DETECTED,
                    "details": {"pii_type": typ, "pattern": regex.pattern},
                }
        return None

    def filter_id(self) -> str:
        return "pii_detector"

class IntentFilter(SafetyFilter):
    """Filter based on query intent classification."""
    def __init__(self, allowed_intents: Set[str], block_reason: ReasonCode = ReasonCode.OFF_TOPIC):
        self.allowed_intents = {i.lower() for i in allowed_intents}
        self.block_reason = block_reason

    def check(self, query_text: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        intent = context.get("intent")
        if intent is None:
            return None
        norm_intent = str(intent).lower()
        if norm_intent not in self.allowed_intents:
            return {
                "decision": Decision.BLOCK,
                "reason": self.block_reason,
                "details": {"intent": norm_intent},
            }
        return None

    def filter_id(self) -> str:
        return "intent_filter"

class SafetyFilterChain:
    """Chain of safety filters executed in priority order.

    Filters are executed in order of ascending priority (lower = higher priority).
    Short-circuits on first block decision. Use check_all for full analysis.
    """
    def __init__(self):
        self._filters: List[Tuple[int, SafetyFilter]] = []
        self._logger = logger

    def add_filter(self, filter: SafetyFilter, priority: int = 100) -> "SafetyFilterChain":
        self._filters.append((priority, filter))
        self._filters.sort(key=lambda x: x[0])
        return self

    def remove_filter(self, filter_id: str) -> bool:
        for i, (priority, f) in enumerate(self._filters):
            if f.filter_id() == filter_id:
                del self._filters[i]
                return True
        return False

    def add_rule(self, rule: FilterRule) -> "SafetyFilterChain":
        if rule.rule_type == "keyword":
            self.add_filter(KeywordFilter(rule), rule.priority)
        elif rule.rule_type == "regex":
            self.add_filter(RegexFilter(rule), rule.priority)
        # Extendable: add more rule types here
        else:
            self._logger.warning(f"Unknown rule_type: {rule.rule_type}")
        return self

    def check(self, query_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if context is None:
            context = {}
        for priority, f in self._filters:
            result = f.check(query_text, context)
            if result:
                return result
        return {"decision": Decision.ALLOW, "reason": ReasonCode.ALLOWED, "details": {"filters_passed": len(self._filters)}}

    def check_all(self, query_text: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if context is None:
            context = {}
        results = []
        for priority, f in self._filters:
            result = f.check(query_text, context)
            if result:
                results.append(result)
        return results

    def load_rules_from_file(self, path: str) -> int:
        with open(path, "r") as f:
            rules = json.load(f)
        count = 0
        for rule_dict in rules:
            rule = FilterRule(**rule_dict)
            self.add_rule(rule)
            count += 1
        return count

    def save_rules_to_file(self, path: str) -> None:
        rules = []
        for priority, f in self._filters:
            if hasattr(f, "rule"):
                rules.append(f.rule.__dict__)
        with open(path, "w") as f:
            json.dump(rules, f, indent=2)

def evaluate_safety(query_intent: Any, retrieved_evidence: List[Dict[str, Any]], query_text: str) -> Dict[str, Any]:
    """Evaluate safety for a query + retrieved evidence deterministically.

    Returns a structured decision dict:
      { decision: Decision, reason: ReasonCode, details: { ... } }

    This gate is usable without any model and makes conservative, deterministic
    decisions. It favors explicit blocking over permissive behavior.
    """
    # For backward compatibility, use a filter chain with default rules
    chain = SafetyFilterChain()
    # Intent filter
    allowed_intents = {"informational", "regulatory", "safety"}
    chain.add_filter(IntentFilter(allowed_intents), priority=10)
    # Procedural lab filter
    chain.add_filter(KeywordFilter(FilterRule(
        rule_id="procedural_lab",
        name="Procedural Lab",
        description="Block procedural lab actions",
        rule_type="keyword",
        patterns=["incubate", "mix", "pipette", "sterilize", "autoclave", "centrifuge", "culture", "prepare solution", "add reagent"],
        action=Decision.BLOCK,
        reason_code=ReasonCode.PROCEDURAL_LAB,
        priority=20,
    )), priority=20)
    # Out-of-scope chemistry
    chain.add_filter(KeywordFilter(FilterRule(
        rule_id="out_of_scope_chemistry",
        name="Out of Scope Chemistry",
        description="Block synthesis/illicit instructions",
        rule_type="keyword",
        patterns=["synthesize", "how to make", "how to synthesize", "manufacture", "produce"],
        action=Decision.BLOCK,
        reason_code=ReasonCode.OUT_OF_SCOPE_CHEMISTRY,
        priority=30,
    )), priority=30)
    # Evidence presence: custom filter
    class EvidenceFilter(SafetyFilter):
        def check(self, query_text, context):
            retrieved_evidence = context.get("retrieved_evidence", [])
            if not retrieved_evidence or all(not (item.get("text") or "") for item in retrieved_evidence):
                return {"decision": Decision.BLOCK, "reason": ReasonCode.NO_EVIDENCE, "details": {"retrieved_count": len(retrieved_evidence)}}
            return None
        def filter_id(self):
            return "evidence_filter"
    chain.add_filter(EvidenceFilter(), priority=40)
    # Compose context
    context = {"intent": _normalize_intent(query_intent), "retrieved_evidence": retrieved_evidence}
    result = chain.check(query_text, context)
    if result["decision"] == Decision.ALLOW:
        result["details"]["retrieved_count"] = len(retrieved_evidence)
    return result

def create_default_filter_chain(domain: str = "general") -> SafetyFilterChain:
    """
    Factory for a default filter chain. Always includes PIIFilter and domain-specific keyword filters.
    Example:
        chain = create_default_filter_chain("pharma")
        result = chain.check("mix solution", {"intent": "informational"})
    """
    chain = SafetyFilterChain()
    chain.add_filter(PIIFilter(), priority=5)
    if domain == "general":
        chain.add_filter(KeywordFilter(FilterRule(
            rule_id="basic_safety",
            name="Basic Safety",
            description="Block basic unsafe keywords",
            rule_type="keyword",
            patterns=["kill", "suicide", "bomb", "attack"],
            action=Decision.BLOCK,
            reason_code=ReasonCode.TOXICITY,
            priority=10,
        )), priority=10)
    elif domain == "pharma":
        chain.add_filter(KeywordFilter(FilterRule(
            rule_id="lab_procedure",
            name="Lab Procedure",
            description="Block procedural lab actions",
            rule_type="keyword",
            patterns=["incubate", "pipette", "autoclave", "centrifuge", "synthesize", "prepare solution"],
            action=Decision.BLOCK,
            reason_code=ReasonCode.PROCEDURAL_LAB,
            priority=10,
        )), priority=10)
    elif domain == "finance":
        chain.add_filter(KeywordFilter(FilterRule(
            rule_id="financial_advice",
            name="Financial Advice",
            description="Block financial advice",
            rule_type="keyword",
            patterns=["buy stock", "sell stock", "investment advice", "guaranteed returns"],
            action=Decision.BLOCK,
            reason_code=ReasonCode.CUSTOM_RULE,
            priority=10,
        )), priority=10)
    elif domain == "legal":
        chain.add_filter(KeywordFilter(FilterRule(
            rule_id="legal_advice",
            name="Legal Advice",
            description="Block legal advice",
            rule_type="keyword",
            patterns=["legal advice", "represent you in court", "file a lawsuit"],
            action=Decision.BLOCK,
            reason_code=ReasonCode.CUSTOM_RULE,
            priority=10,
        )), priority=10)
    return chain

__all__ = [
    "Decision",
    "ReasonCode",
    "evaluate_safety",
    "FilterRule",
    "SafetyFilter",
    "KeywordFilter",
    "RegexFilter",
    "PIIFilter",
    "IntentFilter",
    "SafetyFilterChain",
    "create_default_filter_chain",
]


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
