"""Behaviour-equivalence checking: did the agent do what it was supposed to, and nothing else?

An agent that finishes and says "done!" has told you nothing. The model's own summary of its work
is not evidence. So after every run we compare the side effects the task *expected* against the
effects that *actually happened*, and that diff is the verdict.

This is the part that makes shadow mode valuable. Effects recorded in shadow mode are simulated,
but the diff works exactly the same on them — so you can prove an agent would have behaved
correctly before you ever let it touch production.

TASK 2: verify.
"""

from __future__ import annotations

from app.models import Run, Task, Verdict


def verify(task: Task, run: Run) -> Verdict:
    """TASK 2 — TODO(candidate): diff what was expected against what happened.

    Another pure function — two objects in, a Verdict out, no I/O. Do this second.

    You're comparing:
      task.expected_effects : list[ExpectedEffect]  — each has `tool` and `match` (a dict)
      run.effects           : list[Effect]          — each has `tool`, `args` (a dict), `simulated`

    The rules:

      1. An ExpectedEffect is MATCHED by an actual Effect when the `tool` names are equal AND
         every key/value in the expected `match` appears in the actual `args`. It's a SUBSET
         check, not an equality check — the expectation says "a message to c_1" and doesn't care
         what the body was. So `{"contact_id": "c_1"} matches args {"contact_id": "c_1",
         "body": "hi", "idempotency_key": "r_1:2"}`.

      2. Each actual effect can only be used once. If the task expects two messages to c_1 and the
         agent sent one, that's one matched and one missing — not two matched off the same send.
         This is the rule people get wrong. If you loop over expectations and search the full list
         of effects each time, a single effect will satisfy every expectation that looks like it.
         Keep track of which effects you've already spent.

      3. Any actual effect that no expectation claimed goes in `unexpected`. This half matters as
         much as the other: an agent that did everything asked of it *and also* messaged three
         other people has failed. Missing effects are a bug; extra effects are a liability.

      4. passed = there is nothing in `missing` AND nothing in `unexpected`.

      5. Set `mode` to the run's autonomy level, and `detail` to a short summary a human can read
         at a glance — e.g. "2 matched, 0 missing, 1 unexpected".

      6. Treat simulated effects exactly like real ones. Don't filter them out, don't special-case
         shadow mode. That identical treatment IS the feature.

    Suggested tests — one per outcome:
      - everything expected happened, nothing else       → passed=True
      - an expected effect never happened                → passed=False, it's in `missing`
      - the agent did something nobody asked for         → passed=False, it's in `unexpected`
      - a shadow run with simulated effects              → same verdict as the real run would give
      - two identical expectations, only one real effect  → 1 matched, 1 missing (rule 2)

    Write at least the first three before you write the function.
    """
    matched = []
    missing = []
    
    # Copy effects so we can "spend" them without mutating the original run object
    available_effects = run.effects.copy()
    
    for expected in task.expected_effects:
        found = False
        for i, effect in enumerate(available_effects):
            if effect.tool == expected.tool:
                # Check subset match
                matches_all = True
                for k, v in expected.match.items():
                    if effect.args.get(k) != v:
                        matches_all = False
                        break
                
                if matches_all:
                    found = True
                    matched.append(expected)
                    available_effects.pop(i)  # Rule 2: spend the effect
                    break
        
        if not found:
            missing.append(expected)
            
    # Rule 3: Any leftover effects are unexpected
    unexpected = available_effects
    
    # Rule 4: pass if no missing and no unexpected
    passed = len(missing) == 0 and len(unexpected) == 0
    
    # Rule 5: detailed string
    detail = f"{len(matched)} matched, {len(missing)} missing, {len(unexpected)} unexpected"
    
    return Verdict(
        passed=passed,
        matched=matched,
        missing=missing,
        unexpected=unexpected,
        mode=run.autonomy,
        detail=detail
    )
