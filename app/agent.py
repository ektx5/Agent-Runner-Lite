"""The agent loop. This is the centrepiece of the exercise.

An "agent" here is not magic. It's a loop:

    ask the model what to do  →  do it  →  tell the model what happened  →  repeat

…until the model says it's finished, or we run out of turns. Everything else in this project exists
to make that loop safe: the gate decides whether a step is allowed, the verifier checks afterwards
that the right things happened, the retry keeps a flaky provider from killing the run.

PROVIDED: AgentDeps, and the helpers `_decide`, `_emit`, `_ask_reviewer`, `_execute`, `_observation`.
TASK 3:   run_agent.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import ValidationError

from app.autonomy import evaluate_gate
from app.config import SETTINGS, Settings
from app.model_client import ModelClient, complete_with_retry
from app.models import AgentIntent, Effect, Run, Step, Task
from app.store import Store
from app.tools import ToolDef, ToolError, Workspace
from app.verifier import verify

SYSTEM_PROMPT = (
    "You are an automation agent. On each turn reply with ONLY a JSON object matching this schema:\n"
    '{"intent": "tool_use"|"final", "thought": str, "tool": str|null, '
    '"args": object, "answer": str|null}.\n'
    "Use tool_use to call a tool, or final with an answer when the task is done."
)


class AgentDeps:
    """Everything a run needs, handed in from outside rather than imported.

    This is dependency injection, and it's the reason this project is testable at all. `run_agent`
    never reaches for a global model or a global store — it uses what it was given. So a test can
    hand it a scripted model and a throwaway Store, and there's nothing to clean up afterwards.
    See `make_run` in tests/helpers.py, which does exactly that.
    """

    def __init__(
        self,
        model: ModelClient,
        workspace: Workspace,
        registry: dict[str, ToolDef],
        store: Store,
        settings: Settings = SETTINGS,
    ):
        self.model = model
        self.workspace = workspace
        self.registry = registry
        self.store = store
        self.settings = settings


def run_agent(run: Run, deps: AgentDeps) -> Run:
    """TASK 3 — TODO(candidate): drive one run from start to finish. The main event.

    Do this AFTER tasks 1, 2 and 4 — this function calls all of them, and it's far easier to debug
    a loop when the pieces it uses are already tested.

    The helpers below do the fiddly parts. Read all five before you start; roughly two thirds of
    this function is calling them in the right order.

    ── The shape of it ────────────────────────────────────────────────────────────────────────

        task = deps.store.get_task(run.task_id)
        messages = [system prompt, then the task goal as a user message]
        writes_done = 0
        run.status = "running"

        for _ in range(deps.settings.max_steps):

            1. DECIDE. intent = _decide(deps, messages)
               If it's None the model never produced valid JSON: emit an "error" step, set
               run.status = "failed" and run.error, and stop.

            2. FINISHED? If intent.intent == "final": emit a "final" step carrying
               intent.answer, set run.status = "completed", and break out of the loop.

            3. FIND THE TOOL. tool = deps.registry.get(intent.tool)
               If it isn't there, this is a hallucinated tool name. Do NOT crash and do NOT fail
               the run — emit a "tool_result" step with ok=False, append an observation saying the
               tool is unknown, and `continue`. The model gets to try something else. This is the
               single most common reason a real agent loop falls over, and the `unknown_tool`
               scenario exists so you can prove yours doesn't.

            4. GATE IT. decision = evaluate_gate(run.autonomy, tool.kind, writes_done,
                                                 deps.settings.max_auto_writes)
               Emit a "gate" step recording decision.reason — this is the audit trail.

               If decision.requires_approval:
                   approved = _ask_reviewer(task, run, deps)
                   If not approved: emit a failed "tool_result", append an observation saying the
                   reviewer declined, and `continue`. A rejection is not a crash — the run carries
                   on and the model can do something else or finish.

            5. RUN IT. result, ok = _execute(tool, intent.args, run, decision.simulate)
               Emit a "tool_call" step and a "tool_result" step.

               If ok and tool.kind == "write":
                   append an Effect to run.effects — tool name, the args used, and
                   simulated=decision.simulate — and increment writes_done. Both halves matter:
                   the Effect is what the verifier reads, and writes_done is what the gate's
                   budget counts.

               An unsuccessful tool call (ok is False, meaning it raised ToolError) is NOT fatal
               either. Feed it back like the unknown tool and carry on.

            6. TELL THE MODEL. messages.append({"role": "user",
                                                "content": _observation(result)})

        If the loop runs to its limit without a "final", the model never finished: emit an "error"
        step and set run.status = "failed". Do not leave the status as "running" — a caller polling
        this run would wait forever.

        Wrap the whole thing in try/except. A FatalError from the model, or any unexpected
        exception, should mark the run "failed" with the message in run.error rather than escaping
        to the caller. An agent runner that 500s when a provider rejects a key isn't finished.

        Finally, if the run completed: run.verdict = verify(task, run). Then return run.

    ── What we're looking for ─────────────────────────────────────────────────────────────────

    The theme running through all of the above: a tool that doesn't exist, a tool that errors, and
    a reviewer who says no are all NORMAL. They're observations, fed back to the model. Only two
    things end a run — the model saying it's done, or something genuinely unrecoverable.

    Every branch emits a Step. Afterwards, `run.steps` should read like a transcript of what
    happened and why each write was permitted.

    ── Tests worth writing ────────────────────────────────────────────────────────────────────

      - the "default" scenario completes with no tool calls and no effects
      - "send_followup" under `autonomous` completes, records exactly one effect, and passes
        verification when the task expects that message
      - the same scenario under `shadow` records the effect with simulated=True, still passes
        verification, and leaves `deps.workspace.messages` EMPTY (nothing really happened —
        assert on the workspace, not just the effect list)
      - "unknown_tool" completes rather than raising
      - "three_writes" under `autonomous` with max_auto_writes=1 → the later writes are gated
      - "never_finishes" ends with status "failed" and does not hang
      - "bad_credentials" ends with status "failed" and does not raise out of run_agent
    """
    task = deps.store.get_task(run.task_id)
    messages = initial_messages(task.goal)
    writes_done = 0
    run.status = "running"
    
    try:
        for _ in range(deps.settings.max_steps):
            intent = _decide(deps, messages)
            if intent is None:
                _emit(run, "error", message="model produced invalid JSON")
                run.status = "failed"
                run.error = "model produced invalid JSON"
                break
                
            if intent.intent == "final":
                _emit(run, "final", message=intent.answer or "")
                run.status = "completed"
                break
                
            tool_name = intent.tool
            tool = deps.registry.get(tool_name)
            if tool is None:
                _emit(run, "tool_result", tool=tool_name, result={"error": "unknown tool"}, ok=False)
                messages.append({"role": "user", "content": _observation({"error": f"unknown tool: {tool_name}"})})
                continue
                
            decision = evaluate_gate(run.autonomy, tool.kind, writes_done, deps.settings.max_auto_writes)
            _emit(run, "gate", message=decision.reason)
            
            if decision.requires_approval:
                approved = _ask_reviewer(task, run, deps)
                if not approved:
                    _emit(run, "tool_result", tool=tool_name, result={"error": "reviewer declined"}, ok=False)
                    messages.append({"role": "user", "content": _observation({"error": "reviewer declined"})})
                    continue
            
            result, ok = _execute(tool, intent.args, run, decision.simulate)
            _emit(run, "tool_call", tool=tool_name, args=intent.args)
            _emit(run, "tool_result", tool=tool_name, result=result, ok=ok)
            
            if ok and tool.kind == "write":
                run.effects.append(Effect(tool=tool_name, args=intent.args, simulated=decision.simulate))
                writes_done += 1
                
            messages.append({"role": "user", "content": _observation(result)})
            
        else:
            _emit(run, "error", message="max steps reached without final intent")
            run.status = "failed"
            run.error = "max steps reached without final intent"
            
    except Exception as e:
        _emit(run, "error", message=str(e))
        run.status = "failed"
        run.error = str(e)
        
    if run.status == "completed":
        run.verdict = verify(task, run)
        
    return run


# ─── Provided helpers ─────────────────────────────────────────────────────────────────────────


def _decide(deps: AgentDeps, messages: list[dict]) -> Optional[AgentIntent]:
    """Ask the model what to do next, and insist on a well-formed answer.

    PROVIDED — read it, it's the structured-output pattern you'd write by hand at work.

    The model returns text. We try to parse that text into an AgentIntent. If it isn't valid we
    tell the model so and ask again, up to `settings.parse_max_retries` times. Returns None if it
    never manages it, which the caller treats as a failed run.
    """
    raw = complete_with_retry(deps.model, messages, deps.settings)
    for attempt in range(deps.settings.parse_max_retries + 1):
        try:
            return AgentIntent.model_validate_json(raw)
        except ValidationError:
            if attempt >= deps.settings.parse_max_retries:
                return None
            messages.append(
                {
                    "role": "user",
                    "content": "Your last reply was not valid JSON for the schema. "
                    "Reply with ONLY the JSON object.",
                }
            )
            raw = complete_with_retry(deps.model, messages, deps.settings)
    return None


def _emit(run: Run, step_type: str, **fields: Any) -> Step:
    """Append a Step to the run's trail and return it.

        _emit(run, "gate", message=decision.reason)
        _emit(run, "tool_result", tool=name, result=result, ok=False)

    PROVIDED. Use it for every observable event.
    """
    step = Step(index=len(run.steps), type=step_type, **fields)  # type: ignore[arg-type]
    run.steps.append(step)
    return step


def _ask_reviewer(task: Task, run: Run, deps: AgentDeps) -> bool:
    """Ask the human reviewer whether this write may proceed.

    PROVIDED. In the real product this suspends the run, notifies a reviewer, and waits for them
    to click approve or reject. Here the answer is `task.reviewer_approves`, decided when the task
    was created — the same trick as the scripted model, and for the same reason: a test can't wait
    for a human.

    What matters is that your loop handles BOTH answers correctly, not how the answer arrives.
    """
    approved = task.reviewer_approves
    _emit(
        run,
        "gate",
        message=f"reviewer {'approved' if approved else 'rejected'} the write",
        ok=approved,
    )
    return approved


def _execute(tool: ToolDef, args: dict, run: Run, simulate: bool) -> tuple[Any, bool]:
    """Run a tool, or pretend to. Returns (result, ok).

    PROVIDED. Three things happen here:

      - in simulate mode (shadow), the tool is never called. We return a marker result instead.
        This is the line that makes shadow mode risk-free.
      - for `send_message`, an idempotency key is generated from the run id and the step count,
        so the same logical send always carries the same key. This is the key your TASK 4b
        implementation remembers.
      - a ToolError is caught and turned into (error_dict, False) rather than being allowed to
        propagate — which is how your loop gets to treat a tool failure as an observation instead
        of a crash.
    """
    if simulate:
        return {"simulated": True, "note": "shadow mode: the side effect was not executed"}, True

    if tool.name == "send_message" and "idempotency_key" not in args:
        args["idempotency_key"] = f"{run.id}:{len(run.steps)}"

    try:
        return tool.func(**args), True
    except ToolError as exc:
        return {"error": str(exc)}, False


def _observation(result: Any) -> str:
    """Format a tool result as the message the model sees next turn. PROVIDED."""
    return f"Observation: {json.dumps(result, default=str)}"


def initial_messages(goal: str) -> list[dict]:
    """The opening conversation: the system prompt, then the goal. PROVIDED."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {goal}"},
    ]


# Re-exported so tests and the API can import them from one place.
__all__ = [
    "AgentDeps",
    "run_agent",
    "initial_messages",
    "evaluate_gate",
    "verify",
    "Effect",
]
