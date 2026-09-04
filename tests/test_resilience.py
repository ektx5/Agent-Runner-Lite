"""Task 6 — Tests.

Covers:
  - retry: throttled-twice-then-succeeds (model.calls == 3), fatal-not-retried (model.calls == 1)
  - idempotency: same key twice, len(ws.messages) == 1
  - agent loop: completes with no tools; one write + verification; shadow leaves workspace clean;
                unknown_tool recovers; never_finishes → failed; bad_credentials → failed
  - end-to-end HTTP via the `client` fixture
"""
from __future__ import annotations

import dataclasses

import pytest

from app.agent import run_agent
from app.config import SETTINGS
from app.model_client import (
    FatalError,
    MockModelClient,
    ThrottleError,
    complete_with_retry,
)
from app.models import ExpectedEffect
from app.seed import SCENARIOS
from app.tools import ToolError, Workspace
from tests.helpers import make_run

# ── fast settings so retries don't actually sleep in tests ──────────────────
FAST = dataclasses.replace(SETTINGS, model_backoff_base_seconds=0.0)


# ─── Retry tests ────────────────────────────────────────────────────────────

def test_retry_throttled_twice_then_succeeds():
    """ThrottleError twice then success: model.calls == 3 and we get the result."""
    model = MockModelClient([
        ThrottleError("429"),
        ThrottleError("429"),
        '{"intent": "final", "answer": "ok"}',
    ])
    result = complete_with_retry(model, [], FAST)
    assert result == '{"intent": "final", "answer": "ok"}'
    assert model.calls == 3


def test_retry_fatal_not_retried():
    """FatalError must raise immediately — model.calls == 1 proves no retry happened."""
    model = MockModelClient([
        FatalError("bad key"),
        '{"intent": "final", "answer": "ok"}',
    ])
    with pytest.raises(FatalError):
        complete_with_retry(model, [], FAST)
    assert model.calls == 1  # The critical assertion: no retry on a fatal error


# ─── Idempotency tests ───────────────────────────────────────────────────────

def test_send_message_idempotent():
    """Calling send_message twice with the same key must NOT create two messages."""
    ws = Workspace([{"id": "c_1", "name": "Acme", "email": "a@test.com", "stage": "active"}])
    ws.send_message(contact_id="c_1", body="hello", idempotency_key="key-1")
    result2 = ws.send_message(contact_id="c_1", body="hello again", idempotency_key="key-1")
    # Assert on the WORLD, not the return value — this is the bug we're guarding against
    assert len(ws.messages) == 1
    assert result2["deduped"] is True


def test_send_message_different_keys():
    """Two distinct keys must each result in a real send."""
    ws = Workspace([{"id": "c_1", "name": "Acme", "email": "a@test.com", "stage": "active"}])
    ws.send_message(contact_id="c_1", body="first", idempotency_key="key-1")
    ws.send_message(contact_id="c_1", body="second", idempotency_key="key-2")
    assert len(ws.messages) == 2


def test_send_message_missing_key_raises():
    ws = Workspace([{"id": "c_1", "name": "Acme", "email": "a@test.com", "stage": "active"}])
    with pytest.raises(ToolError):
        ws.send_message(contact_id="c_1", body="oops", idempotency_key="")


# ─── Agent loop tests ────────────────────────────────────────────────────────

def test_loop_completes_no_tools():
    """The simplest path: model immediately says final, no tools, no effects."""
    run, deps = make_run(script=SCENARIOS["default"], expected=[])
    result = run_agent(run, deps)
    assert result.status == "completed"
    assert result.effects == []
    assert result.verdict.passed is True


def test_loop_one_write_passes_verification():
    """send_followup: one write, verified against the expected effect."""
    run, deps = make_run(
        script=SCENARIOS["send_followup"],
        expected=[ExpectedEffect(tool="send_message", match={"contact_id": "c_1"})],
        autonomy="autonomous",
    )
    result = run_agent(run, deps)
    assert result.status == "completed"
    assert len(result.effects) == 1
    assert result.effects[0].tool == "send_message"
    assert result.verdict.passed is True


def test_loop_shadow_records_simulated_leaves_workspace_empty():
    """Shadow mode: effect is recorded but workspace.messages is untouched."""
    run, deps = make_run(
        script=SCENARIOS["send_followup"],
        expected=[ExpectedEffect(tool="send_message", match={"contact_id": "c_1"})],
        autonomy="shadow",
    )
    result = run_agent(run, deps)
    assert result.status == "completed"
    assert result.effects[0].simulated is True
    assert result.verdict.passed is True
    # The key assertion: nothing ACTUALLY happened
    assert deps.workspace.messages == []


def test_loop_unknown_tool_recovers():
    """An unknown tool name must not crash the run — the loop feeds it back and continues."""
    run, deps = make_run(script=SCENARIOS["unknown_tool"], expected=[])
    result = run_agent(run, deps)
    assert result.status == "completed"
    bad_steps = [s for s in result.steps if s.type == "tool_result" and not s.ok]
    assert len(bad_steps) >= 1


def test_loop_never_finishes_fails_cleanly():
    """A model that loops forever must be caught by max_steps and marked failed."""
    run, deps = make_run(
        script=SCENARIOS["never_finishes"],
        expected=[],
        settings=dataclasses.replace(SETTINGS, max_steps=3),
    )
    result = run_agent(run, deps)
    assert result.status == "failed"
    assert result.error is not None
    assert "max steps reached" in result.error


def test_loop_bad_credentials_fails_cleanly():
    """FatalError from model must mark the run failed, not propagate out of run_agent."""
    run, deps = make_run(script=SCENARIOS["bad_credentials"], expected=[])
    result = run_agent(run, deps)
    assert result.status == "failed"
    assert result.error is not None


def test_loop_three_writes_budget_exhausted():
    """Under autonomous with max_auto_writes=1, the second write needs approval."""
    run, deps = make_run(
        script=SCENARIOS["three_writes"],
        expected=[],
        autonomy="autonomous",
        settings=dataclasses.replace(SETTINGS, max_auto_writes=1),
    )
    result = run_agent(run, deps)
    assert result.status == "completed"
    gate_steps = [s for s in result.steps if s.type == "gate"]
    approval_gates = [
        s for s in gate_steps
        if "budget exceeded" in s.message or "approval" in s.message.lower()
    ]
    assert len(approval_gates) >= 1


# ─── End-to-end HTTP test ────────────────────────────────────────────────────

def test_e2e_start_run_via_http(client):
    """Drive a full run over HTTP: create a task, start a run, assert it completed."""
    task_resp = client.post(
        "/api/v1/tasks",
        json={
            "goal": "send a follow-up",
            "scenario": "send_followup",
            "autonomy": "autonomous",
            "expected_effects": [{"tool": "send_message", "match": {"contact_id": "c_1"}}],
        },
    )
    assert task_resp.status_code == 200
    task_id = task_resp.json()["id"]

    run_resp = client.post("/api/v1/runs", json={"task_id": task_id})
    assert run_resp.status_code == 201
    run_data = run_resp.json()
    assert run_data["status"] == "completed"
    assert run_data["verdict"]["passed"] is True


def test_e2e_start_run_unknown_task(client):
    """Starting a run with a non-existent task_id must return 404."""
    resp = client.post("/api/v1/runs", json={"task_id": "t_does_not_exist"})
    assert resp.status_code == 404
