import pytest

from app.agent import run_agent
from tests.helpers import make_run
from app.seed import SCENARIOS

def test_loop_completes_with_no_tools():
    # "default" scenario has no tools expected, just answers
    run, deps = make_run(script=SCENARIOS["default"], expected=[])
    result = run_agent(run, deps)
    assert result.status == "completed"
    assert len(result.effects) == 0
    assert result.verdict.passed is True

def test_loop_send_followup_autonomous():
    # "send_followup" uses send_message
    from app.models import ExpectedEffect
    run, deps = make_run(
        script=SCENARIOS["send_followup"],
        expected=[ExpectedEffect(tool="send_message", match={"contact_id": "c_2"})],
        autonomy="autonomous"
    )
    result = run_agent(run, deps)
    assert result.status == "completed"
    assert len(result.effects) == 1
    assert result.effects[0].tool == "send_message"
    assert result.verdict.passed is True

def test_loop_send_followup_shadow():
    from app.models import ExpectedEffect
    run, deps = make_run(
        script=SCENARIOS["send_followup"],
        expected=[ExpectedEffect(tool="send_message", match={"contact_id": "c_2"})],
        autonomy="shadow"
    )
    result = run_agent(run, deps)
    assert result.status == "completed"
    assert len(result.effects) == 1
    assert result.effects[0].simulated is True
    assert result.verdict.passed is True
    # Assert nothing really happened to workspace messages
    assert len(deps.workspace.messages) == 0

def test_loop_unknown_tool_recovers():
    run, deps = make_run(script=SCENARIOS["unknown_tool"], expected=[])
    result = run_agent(run, deps)
    assert result.status == "completed"
    # Should have a tool_result with ok=False
    tool_results = [s for s in result.steps if s.type == "tool_result"]
    assert len(tool_results) > 0
    assert any(not s.ok for s in tool_results)

def test_loop_never_finishes():
    run, deps = make_run(script=SCENARIOS["never_finishes"], expected=[])
    result = run_agent(run, deps)
    assert result.status == "failed"
    assert "max steps reached" in result.error
