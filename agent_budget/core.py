"""
agent-budget core: Token/cost budget enforcement for LLM agents.

Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Exceptions / Warnings
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    """Raised when a hard budget limit is breached."""

    def __init__(self, message: str, scope: str, limit_type: str, used: float, limit: float):
        super().__init__(message)
        self.scope = scope
        self.limit_type = limit_type
        self.used = used
        self.limit = limit


class BudgetWarning(UserWarning):
    """Emitted when usage crosses a soft alert threshold."""


# ---------------------------------------------------------------------------
# Alert level
# ---------------------------------------------------------------------------

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Model rates (cost per 1 000 tokens, USD)
# ---------------------------------------------------------------------------

@dataclass
class ModelRates:
    """
    Per-model token pricing in USD per 1 000 tokens.

    Defaults reflect approximate mid-2025 pricing; override as needed.
    """
    input_per_1k: float = 0.002
    output_per_1k: float = 0.002

    # Predefined rate tables
    PRESETS: Dict[str, "ModelRates"] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def for_model(cls, model: str) -> "ModelRates":
        """Return known rates for common models, falling back to defaults."""
        _known = {
            "gpt-4o":              cls(input_per_1k=0.005,  output_per_1k=0.015),
            "gpt-4o-mini":         cls(input_per_1k=0.00015, output_per_1k=0.0006),
            "gpt-4-turbo":         cls(input_per_1k=0.01,   output_per_1k=0.03),
            "gpt-3.5-turbo":       cls(input_per_1k=0.0005, output_per_1k=0.0015),
            "claude-3-opus":       cls(input_per_1k=0.015,  output_per_1k=0.075),
            "claude-3-sonnet":     cls(input_per_1k=0.003,  output_per_1k=0.015),
            "claude-3-haiku":      cls(input_per_1k=0.00025, output_per_1k=0.00125),
            "gemini-1.5-pro":      cls(input_per_1k=0.00125, output_per_1k=0.005),
            "gemini-1.5-flash":    cls(input_per_1k=0.000075, output_per_1k=0.0003),
        }
        return _known.get(model, cls())

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_1k + output_tokens * self.output_per_1k) / 1000.0


# ---------------------------------------------------------------------------
# Usage record
# ---------------------------------------------------------------------------

@dataclass
class UsageRecord:
    """A single token-usage event."""
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Budget alert
# ---------------------------------------------------------------------------

@dataclass
class BudgetAlert:
    scope: str
    level: AlertLevel
    message: str
    used: float
    limit: float
    timestamp: float = field(default_factory=time.time)

    @property
    def pct(self) -> float:
        return self.used / self.limit if self.limit else 0.0


# ---------------------------------------------------------------------------
# Budget scope definition
# ---------------------------------------------------------------------------

@dataclass
class BudgetScope:
    """
    Defines budget limits for a named scope (e.g. a session, user, or global).

    Args:
        name: Scope identifier.
        max_tokens_per_session: Hard limit on total tokens for one session.
        max_cost_per_session_usd: Hard limit on cost for one session.
        max_tokens_per_day: Hard daily token limit.
        max_cost_per_day_usd: Hard daily cost limit.
        max_tokens_per_model: Dict of {model: max_tokens} limits.
        max_cost_per_model_usd: Dict of {model: max_cost_usd} limits.
        alert_at_pct: Fraction (0–1) at which to fire soft alerts (default 0.8).
        hard_stop: If True, raise BudgetExceededError when limit hit; otherwise just alert.
    """
    name: str
    max_tokens_per_session: Optional[int] = None
    max_cost_per_session_usd: Optional[float] = None
    max_tokens_per_day: Optional[int] = None
    max_cost_per_day_usd: Optional[float] = None
    max_tokens_per_model: Dict[str, int] = field(default_factory=dict)
    max_cost_per_model_usd: Dict[str, float] = field(default_factory=dict)
    alert_at_pct: float = 0.8
    hard_stop: bool = True


# ---------------------------------------------------------------------------
# Budget (stateful tracker for one scope)
# ---------------------------------------------------------------------------

class Budget:
    """
    Stateful budget tracker for a single BudgetScope.

    Tracks session totals and a rolling daily window.
    """

    def __init__(self, scope: BudgetScope, rates: Optional[Dict[str, ModelRates]] = None) -> None:
        self._scope = scope
        self._rates: Dict[str, ModelRates] = rates or {}
        self._lock = threading.Lock()

        # Session accumulators
        self._session_tokens = 0
        self._session_cost = 0.0

        # Daily accumulators (reset after >24h)
        self._day_tokens = 0
        self._day_cost = 0.0
        self._day_start: float = time.time()

        # Per-model accumulators
        self._model_tokens: Dict[str, int] = {}
        self._model_cost: Dict[str, float] = {}

        # History
        self._records: List[UsageRecord] = []
        self._alerts: List[BudgetAlert] = []
        self._alert_callbacks: List[Callable[[BudgetAlert], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def scope(self) -> BudgetScope:
        return self._scope

    def on_alert(self, callback: Callable[[BudgetAlert], None]) -> None:
        """Register a callback to be invoked on any budget alert."""
        self._alert_callbacks.append(callback)

    def estimate(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        """Estimate the cost of a call without recording it."""
        rates = self._get_rates(model)
        return rates.estimate_cost(input_tokens, output_tokens)

    def check(self, model: str, input_tokens: int, output_tokens: int = 0) -> None:
        """
        Check whether a proposed call would breach any budget limit.

        Raises BudgetExceededError if hard_stop=True and a limit would be exceeded.
        Does NOT record usage — call record() afterwards.
        """
        cost = self.estimate(model, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens
        with self._lock:
            self._maybe_reset_day()
            self._check_limits(model, total_tokens, cost, prospective=True)

    def record(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict] = None,
    ) -> UsageRecord:
        """
        Record actual token usage and enforce budget limits.

        Returns the UsageRecord created.
        Raises BudgetExceededError if a hard limit is breached after recording.
        """
        rates = self._get_rates(model)
        cost = rates.estimate_cost(input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens

        record = UsageRecord(
            session_id=session_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            metadata=metadata or {},
        )

        with self._lock:
            self._maybe_reset_day()

            self._session_tokens += total_tokens
            self._session_cost += cost
            self._day_tokens += total_tokens
            self._day_cost += cost
            self._model_tokens[model] = self._model_tokens.get(model, 0) + total_tokens
            self._model_cost[model] = self._model_cost.get(model, 0.0) + cost
            self._records.append(record)

            self._check_limits(model, total_tokens, cost, prospective=False)

        return record

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        with self._lock:
            self._maybe_reset_day()
            return {
                "scope": self._scope.name,
                "session": {
                    "tokens": self._session_tokens,
                    "cost_usd": round(self._session_cost, 6),
                },
                "day": {
                    "tokens": self._day_tokens,
                    "cost_usd": round(self._day_cost, 6),
                },
                "per_model": {
                    m: {
                        "tokens": self._model_tokens[m],
                        "cost_usd": round(self._model_cost[m], 6),
                    }
                    for m in self._model_tokens
                },
                "alert_count": len(self._alerts),
                "record_count": len(self._records),
            }

    @property
    def alerts(self) -> List[BudgetAlert]:
        with self._lock:
            return list(self._alerts)

    @property
    def records(self) -> List[UsageRecord]:
        with self._lock:
            return list(self._records)

    def reset_session(self) -> None:
        with self._lock:
            self._session_tokens = 0
            self._session_cost = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_rates(self, model: str) -> ModelRates:
        if model in self._rates:
            return self._rates[model]
        return ModelRates.for_model(model)

    def _maybe_reset_day(self) -> None:
        """Reset daily counters if more than 24 hours have elapsed."""
        if time.time() - self._day_start >= 86400:
            self._day_tokens = 0
            self._day_cost = 0.0
            self._day_start = time.time()

    def _check_limits(
        self,
        model: str,
        tokens: int,
        cost: float,
        prospective: bool,
    ) -> None:
        """Check all configured limits and fire alerts / raise errors."""
        s = self._scope

        checks: List[Tuple[Optional[float], float, str]] = []

        # Session limits
        if s.max_tokens_per_session is not None:
            used = (self._session_tokens + tokens) if prospective else self._session_tokens
            checks.append((s.max_tokens_per_session, used, "session_tokens"))
        if s.max_cost_per_session_usd is not None:
            used = (self._session_cost + cost) if prospective else self._session_cost
            checks.append((s.max_cost_per_session_usd, used, "session_cost_usd"))

        # Daily limits
        if s.max_tokens_per_day is not None:
            used = (self._day_tokens + tokens) if prospective else self._day_tokens
            checks.append((s.max_tokens_per_day, used, "day_tokens"))
        if s.max_cost_per_day_usd is not None:
            used = (self._day_cost + cost) if prospective else self._day_cost
            checks.append((s.max_cost_per_day_usd, used, "day_cost_usd"))

        # Per-model limits
        if model in s.max_tokens_per_model:
            cur = self._model_tokens.get(model, 0)
            used = (cur + tokens) if prospective else cur
            checks.append((s.max_tokens_per_model[model], used, f"model_{model}_tokens"))
        if model in s.max_cost_per_model_usd:
            cur = self._model_cost.get(model, 0.0)
            used = (cur + cost) if prospective else cur
            checks.append((s.max_cost_per_model_usd[model], used, f"model_{model}_cost_usd"))

        for limit, used, limit_type in checks:
            if limit is None:
                continue
            pct = used / limit if limit else 0.0

            if pct >= 1.0:
                alert = BudgetAlert(
                    scope=s.name,
                    level=AlertLevel.CRITICAL,
                    message=f"Budget EXCEEDED: {limit_type} used={used:.4f} limit={limit:.4f}",
                    used=used,
                    limit=limit,
                )
                self._alerts.append(alert)
                self._fire_alert(alert)
                if s.hard_stop:
                    raise BudgetExceededError(
                        alert.message,
                        scope=s.name,
                        limit_type=limit_type,
                        used=used,
                        limit=limit,
                    )
            elif pct >= s.alert_at_pct:
                alert = BudgetAlert(
                    scope=s.name,
                    level=AlertLevel.WARNING,
                    message=f"Budget WARNING: {limit_type} at {pct:.0%} ({used:.4f}/{limit:.4f})",
                    used=used,
                    limit=limit,
                )
                self._alerts.append(alert)
                self._fire_alert(alert)

    def _fire_alert(self, alert: BudgetAlert) -> None:
        for cb in self._alert_callbacks:
            try:
                cb(alert)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# BudgetManager — registry for multiple scopes
# ---------------------------------------------------------------------------

class BudgetManager:
    """
    Manages multiple Budget instances keyed by scope name.

    Useful for multi-tenant or multi-session deployments where different
    callers have separate limits.
    """

    def __init__(self) -> None:
        self._budgets: Dict[str, Budget] = {}
        self._lock = threading.Lock()

    def register(self, scope: BudgetScope, rates: Optional[Dict[str, ModelRates]] = None) -> Budget:
        budget = Budget(scope, rates)
        with self._lock:
            self._budgets[scope.name] = budget
        return budget

    def get(self, name: str) -> Optional[Budget]:
        with self._lock:
            return self._budgets.get(name)

    def record(
        self,
        scope_name: str,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict] = None,
    ) -> UsageRecord:
        budget = self.get(scope_name)
        if budget is None:
            raise KeyError(f"No budget registered for scope '{scope_name}'")
        return budget.record(session_id, model, input_tokens, output_tokens, metadata)

    def all_summaries(self) -> Dict[str, Dict]:
        with self._lock:
            names = list(self._budgets.keys())
        return {name: self._budgets[name].summary() for name in names}

    def remove(self, name: str) -> bool:
        with self._lock:
            if name in self._budgets:
                del self._budgets[name]
                return True
            return False
