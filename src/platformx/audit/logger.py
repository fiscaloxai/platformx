from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import json


@dataclass
class AuditEntry:
    timestamp: str
    event: str
    payload: Dict[str, Any]


class AuditLogger:
    """Structured, deterministic in-memory audit logger.

    Responsibilities:
    - Log queries, retrieved sources, safety decisions, refusals, and confidence levels
    - Produce human-inspectable structured logs
    - No external communication; logs remain in-memory until exported
    """

    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []

    def _now(self) -> str:
        # ISO-8601 UTC timestamp
        return datetime.utcnow().isoformat() + "Z"

    def log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        payload = payload or {}
        entry = AuditEntry(timestamp=self._now(), event=event, payload=payload)
        self._entries.append(entry)

    def log_query(self, query_text: str, intent: str) -> None:
        self.log("query", {"text": query_text, "intent": intent})

    def log_retrieval(self, results: List[Dict[str, Any]]) -> None:
        # store only deterministic, inspectable fields
        items = [
            {"dataset_id": r.get("dataset_id"), "chunk_id": r.get("chunk_id"), "score": float(r.get("score", 0.0))}
            for r in results
        ]
        self.log("retrieval", {"results": items})

    def log_safety_decision(self, decision: str, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.log("safety", {"decision": decision, "reason": reason, "details": details or {}})

    def log_refusal(self, refusal_message: str, reason: str) -> None:
        self.log("refusal", {"message": refusal_message, "reason": reason})

    def log_confidence(self, level: str, rationale: Dict[str, Any]) -> None:
        self.log("confidence", {"level": level, "rationale": rationale})

    def get_entries(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self._entries]

    def export_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.get_entries(), fh, indent=2, ensure_ascii=False)
