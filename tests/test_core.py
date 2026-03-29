"""Tests for agent-budget. Minimum 30 tests."""

import time
import threading
import pytest

from agent_budget.core import (
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


# ---------------------------------------------------------------------------
# ModelRates tests (1–7)
# ---------------------------------------------------------------------------

def test_01_model_rates_default():
    r = ModelRates()
    assert r.input_per_1k == 0.002
    assert r.output_per_1k == 0.002


def test_02_model_rates_known_model():
    r = ModelRates.for_model("gpt-4o")
    assert r.input_per_1k == 0.005
    assert r.output_per_1k == 0.015


def test_03_model_rates_unknown_model_uses_defaults():
    r = ModelRates.for_model("some-unknown-model")
    assert r.input_per_1k == 0.002


def test_04_estimate_cost_zero_tokens():
    r = ModelRates(input_per_1k=0.01, output_per_1k=0.03)
    assert r.estimate_cost(0, 0) == 0.0


def test_05_estimate_cost_input_only():
    r = ModelRates(input_per_1k=1.0, output_per_1k=0.0)
    cost = r.estimate_cost(1000, 0)
    assert abs(cost - 1.0) < 1e-9


def test_06_estimate_cost_output_only():
    r = ModelRates(input_per_1k=0.0, output_per_1k=2.0)
    cost = r.estimate_cost(0, 500)
    assert abs(cost - 1.0) < 1e-9


def test_07_estimate_cost_mixed():
    r = ModelRates(input_per_1k=1.0, output_per_1k=2.0)
    cost = r.estimate_cost(1000, 1000)
    assert abs(cost - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Budget basic recording (8–15)
# ---------------------------------------------------------------------------

def make_scope(**kwargs):
    defaults = {"name": "test", "hard_stop": True}
    defaults.update(kwargs)
    return BudgetScope(**defaults)


def make_budget(**kwargs):
    scope = make_scope(**kwargs)
    return Budget(scope)


def test_08_record_creates_usage_record():
    b = make_budget()
    rec = b.record("s1", "gpt-4o-mini", 100, 50)
    assert isinstance(rec, UsageRecord)
    assert rec.session_id == "s1"
    assert rec.model == "gpt-4o-mini"
    assert rec.total_tokens == 150


def test_09_record_accumulates_session_tokens():
    b = make_budget()
    b.record("s1", "gpt-4o-mini", 100, 50)
    b.record("s1", "gpt-4o-mini", 200, 100)
    summary = b.summary()
    assert summary["session"]["tokens"] == 450


def test_10_record_accumulates_day_tokens():
    b = make_budget()
    b.record("s1", "gpt-4o-mini", 100, 0)
    b.record("s1", "gpt-4o-mini", 200, 0)
    summary = b.summary()
    assert summary["day"]["tokens"] == 300


def test_11_record_accumulates_cost():
    rates = {"custom": ModelRates(input_per_1k=1.0, output_per_1k=0.0)}
    scope = make_scope(name="t")
    b = Budget(scope, rates)
    b.record("s1", "custom", 1000, 0)
    summary = b.summary()
    assert abs(summary["session"]["cost_usd"] - 1.0) < 1e-6


def test_12_record_per_model_tracking():
    b = make_budget()
    b.record("s1", "gpt-4o", 100, 50)
    b.record("s1", "gpt-4o-mini", 200, 100)
    summary = b.summary()
    assert "gpt-4o" in summary["per_model"]
    assert "gpt-4o-mini" in summary["per_model"]


def test_13_records_property_returns_all():
    b = make_budget()
    b.record("s1", "gpt-4o-mini", 100, 50)
    b.record("s2", "gpt-4o-mini", 200, 100)
    assert len(b.records) == 2


def test_14_reset_session_clears_session_totals():
    b = make_budget()
    b.record("s1", "gpt-4o-mini", 500, 0)
    b.reset_session()
    summary = b.summary()
    assert summary["session"]["tokens"] == 0
    assert summary["session"]["cost_usd"] == 0.0


def test_15_estimate_does_not_record():
    b = make_budget()
    b.estimate("gpt-4o-mini", 1000, 500)
    assert len(b.records) == 0


# ---------------------------------------------------------------------------
# Hard-stop enforcement (16–22)
# ---------------------------------------------------------------------------

def test_16_session_token_hard_stop():
    b = make_budget(max_tokens_per_session=100, hard_stop=True)
    b.record("s1", "gpt-4o-mini", 50, 0)  # 50 tokens — ok
    with pytest.raises(BudgetExceededError) as exc_info:
        b.record("s1", "gpt-4o-mini", 60, 0)  # 110 total — over
    assert exc_info.value.limit_type == "session_tokens"


def test_17_session_cost_hard_stop():
    rates = {"m": ModelRates(input_per_1k=1.0, output_per_1k=0.0)}
    scope = BudgetScope(name="t", max_cost_per_session_usd=0.5, hard_stop=True)
    b = Budget(scope, rates)
    b.record("s1", "m", 400, 0)  # $0.40
    with pytest.raises(BudgetExceededError) as exc_info:
        b.record("s1", "m", 200, 0)  # would be $0.60
    assert "session_cost" in exc_info.value.limit_type


def test_18_day_token_hard_stop():
    b = make_budget(max_tokens_per_day=100, hard_stop=True)
    b.record("s1", "gpt-4o-mini", 90, 0)
    with pytest.raises(BudgetExceededError) as exc_info:
        b.record("s1", "gpt-4o-mini", 20, 0)
    assert "day_tokens" in exc_info.value.limit_type


def test_19_day_cost_hard_stop():
    rates = {"m": ModelRates(input_per_1k=1.0, output_per_1k=0.0)}
    scope = BudgetScope(name="t", max_cost_per_day_usd=1.0, hard_stop=True)
    b = Budget(scope, rates)
    b.record("s1", "m", 900, 0)  # $0.90
    with pytest.raises(BudgetExceededError):
        b.record("s1", "m", 200, 0)  # $0.20 would exceed $1.00


def test_20_model_token_hard_stop():
    scope = BudgetScope(name="t", max_tokens_per_model={"gpt-4o": 100}, hard_stop=True)
    b = Budget(scope)
    b.record("s1", "gpt-4o", 80, 0)
    with pytest.raises(BudgetExceededError) as exc_info:
        b.record("s1", "gpt-4o", 30, 0)
    assert "gpt-4o" in exc_info.value.limit_type


def test_21_no_hard_stop_allows_overrun():
    b = make_budget(max_tokens_per_session=50, hard_stop=False)
    b.record("s1", "gpt-4o-mini", 30, 0)
    rec = b.record("s1", "gpt-4o-mini", 30, 0)  # 60 total — over, but no raise
    assert rec is not None


def test_22_check_raises_prospectively():
    b = make_budget(max_tokens_per_session=100, hard_stop=True)
    b.record("s1", "gpt-4o-mini", 80, 0)  # 80 recorded
    with pytest.raises(BudgetExceededError):
        b.check("gpt-4o-mini", 30)  # 110 prospective


# ---------------------------------------------------------------------------
# Alert callbacks (23–27)
# ---------------------------------------------------------------------------

def test_23_alert_callback_fired_on_warning():
    alerts = []
    b = make_budget(max_tokens_per_session=100, alert_at_pct=0.7, hard_stop=False)
    b.on_alert(alerts.append)
    b.record("s1", "gpt-4o-mini", 75, 0)  # 75% → warning
    assert any(a.level == AlertLevel.WARNING for a in alerts)


def test_24_alert_callback_fired_on_exceeded():
    alerts = []
    b = make_budget(max_tokens_per_session=100, hard_stop=False)
    b.on_alert(alerts.append)
    b.record("s1", "gpt-4o-mini", 110, 0)
    assert any(a.level == AlertLevel.CRITICAL for a in alerts)


def test_25_alerts_property_accumulates():
    b = make_budget(max_tokens_per_session=100, alert_at_pct=0.5, hard_stop=False)
    b.record("s1", "gpt-4o-mini", 60, 0)
    b.record("s1", "gpt-4o-mini", 50, 0)  # exceeds
    assert len(b.alerts) >= 2


def test_26_alert_contains_correct_scope():
    b = make_budget(name="myapp", max_tokens_per_session=50, hard_stop=False)
    b.record("s1", "gpt-4o-mini", 60, 0)
    assert b.alerts[-1].scope == "myapp"


def test_27_alert_pct_correct():
    b = make_budget(max_tokens_per_session=100, alert_at_pct=0.0, hard_stop=False)
    b.record("s1", "gpt-4o-mini", 50, 0)
    alert = b.alerts[0]
    assert abs(alert.pct - 0.5) < 0.01


# ---------------------------------------------------------------------------
# BudgetManager tests (28–32)
# ---------------------------------------------------------------------------

def test_28_manager_register_and_get():
    mgr = BudgetManager()
    scope = BudgetScope(name="s1")
    mgr.register(scope)
    assert mgr.get("s1") is not None


def test_29_manager_record_dispatches():
    mgr = BudgetManager()
    scope = BudgetScope(name="s1")
    mgr.register(scope)
    rec = mgr.record("s1", "sess", "gpt-4o-mini", 100, 50)
    assert rec.total_tokens == 150


def test_30_manager_unknown_scope_raises():
    mgr = BudgetManager()
    with pytest.raises(KeyError):
        mgr.record("nonexistent", "sess", "gpt-4o-mini", 100, 50)


def test_31_manager_all_summaries():
    mgr = BudgetManager()
    mgr.register(BudgetScope(name="a"))
    mgr.register(BudgetScope(name="b"))
    summaries = mgr.all_summaries()
    assert "a" in summaries and "b" in summaries


def test_32_manager_remove():
    mgr = BudgetManager()
    mgr.register(BudgetScope(name="to_remove"))
    removed = mgr.remove("to_remove")
    assert removed is True
    assert mgr.get("to_remove") is None


# ---------------------------------------------------------------------------
# Thread-safety (33–34)
# ---------------------------------------------------------------------------

def test_33_thread_safe_concurrent_records():
    scope = BudgetScope(name="concurrent", max_tokens_per_session=100000, hard_stop=True)
    b = Budget(scope)
    errors = []

    def worker():
        try:
            b.record("s1", "gpt-4o-mini", 10, 5)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert b.summary()["session"]["tokens"] == 1500


def test_34_thread_safe_manager():
    mgr = BudgetManager()
    mgr.register(BudgetScope(name="ts", max_tokens_per_session=1000000))
    errors = []

    def worker(i):
        try:
            mgr.record("ts", f"sess{i}", "gpt-4o-mini", 10, 5)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


# ---------------------------------------------------------------------------
# Metadata & edge cases (35)
# ---------------------------------------------------------------------------

def test_35_record_with_metadata():
    b = make_budget()
    rec = b.record("s1", "gpt-4o-mini", 100, 50, metadata={"request_id": "abc123"})
    assert rec.metadata["request_id"] == "abc123"
