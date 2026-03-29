"""agent-budget: Zero-dependency token/cost budget enforcement for LLM agents."""

from .core import (
    Budget,
    BudgetManager,
    BudgetScope,
    BudgetAlert,
    BudgetExceededError,
    BudgetWarning,
    ModelRates,
    UsageRecord,
    AlertLevel,
)

__version__ = "0.1.0"
__all__ = [
    "Budget",
    "BudgetManager",
    "BudgetScope",
    "BudgetAlert",
    "BudgetExceededError",
    "BudgetWarning",
    "ModelRates",
    "UsageRecord",
    "AlertLevel",
]
