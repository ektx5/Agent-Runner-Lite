"""Governed autonomy: how much the agent is allowed to do on its own.

This is the smallest file in the project and one of the most important. An agent that can send
messages and update records needs a policy sitting between "the model asked for this" and "it
happened". That policy is one pure function.

TASK 1: evaluate_gate.
"""

from __future__ import annotations

from app.config import SETTINGS
from app.models import AutonomyLevel, GateDecision, ToolKind


def evaluate_gate(
    level: AutonomyLevel,
    tool_kind: ToolKind,
    writes_so_far: int,
    max_auto_writes: int = SETTINGS.max_auto_writes,
) -> GateDecision:
    """TASK 1 — TODO(candidate): decide how one tool call is allowed to proceed.

    Start here. It's the smallest task, it needs nothing else to be finished first, and it's the
    easiest thing in the project to test properly — which makes it the right place to practise
    writing the tests first.

    Notice what this function does NOT do: it doesn't call a tool, doesn't touch the workspace,
    doesn't log. It takes four plain values and returns a decision. That's what "pure" means, and
    it's why you can test every branch of it in a few lines with no setup. Keep it that way.

    The policy — three governance stages, which in the real product a run graduates through as it
    earns trust:

      read tools
          Always allowed, at every level. Looking something up can't break anything.

      shadow — "show me what you would do"
          Write tools are ALLOWED BUT SIMULATED (allow=True, simulate=True). The agent behaves
          exactly as it would in production and we record what it tried to do, but nothing
          actually changes. This is how a new agent gets evaluated against real tasks with zero
          risk: run it in shadow, then check the effects it *would* have produced.

      supervised — "ask me first"
          Every write stops and waits for a human (allow=False, requires_approval=True).

      autonomous — "go ahead, within limits"
          Writes are auto-approved while `writes_so_far < max_auto_writes`. Once the budget is
          used up, further writes need approval (allow=False, requires_approval=True). The budget
          is the safety rail: an autonomous agent stuck in a loop can send two messages, not two
          thousand.

    Fill in `reason` with a short human-readable string. It ends up in the run's audit trail, and
    "why was this allowed?" is the first question anyone asks about an agent that did something
    surprising.

    See GateDecision in app/models.py for the exact fields.

    Suggested tests — this is a decision table, so a table-driven test with
    `@pytest.mark.parametrize` covers it neatly:
      - a read tool at each of the three levels → allowed, not simulated, no approval
      - shadow + write → allow=True, simulate=True
      - supervised + write → requires_approval=True, allow=False
      - autonomous + write with writes_so_far=0 and max_auto_writes=2 → allowed
      - autonomous + write with writes_so_far=2 and max_auto_writes=2 → requires approval
        (the boundary — get this one exactly right; off-by-one here means the budget is 3, not 2)
    """
    if tool_kind == "read":
        return GateDecision(allow=True, simulate=False, requires_approval=False, reason="Reads are always allowed.")
    
    if level == "shadow":
        return GateDecision(allow=True, simulate=True, requires_approval=False, reason="Simulating write in shadow mode.")
    elif level == "supervised":
        return GateDecision(allow=False, simulate=False, requires_approval=True, reason="Writes require approval in supervised mode.")
    elif level == "autonomous":
        if writes_so_far < max_auto_writes:
            return GateDecision(allow=True, simulate=False, requires_approval=False, reason="Write allowed within autonomous budget.")
        else:
            return GateDecision(allow=False, simulate=False, requires_approval=True, reason="Autonomous write budget exceeded, requires approval.")
