import pytest

from app.autonomy import evaluate_gate
from app.models import AutonomyLevel, GateDecision, ToolKind

@pytest.mark.parametrize(
    "level, tool_kind, writes_so_far, max_auto_writes, expected",
    [
        # Reads are always allowed, not simulated, no approval
        ("shadow", "read", 0, 2, GateDecision(allow=True, simulate=False, requires_approval=False, reason="Reads are always allowed.")),
        ("supervised", "read", 0, 2, GateDecision(allow=True, simulate=False, requires_approval=False, reason="Reads are always allowed.")),
        ("autonomous", "read", 5, 2, GateDecision(allow=True, simulate=False, requires_approval=False, reason="Reads are always allowed.")),
        
        # Shadow + write -> allowed, simulated
        ("shadow", "write", 0, 2, GateDecision(allow=True, simulate=True, requires_approval=False, reason="Simulating write in shadow mode.")),
        ("shadow", "write", 10, 2, GateDecision(allow=True, simulate=True, requires_approval=False, reason="Simulating write in shadow mode.")),

        # Supervised + write -> not allowed, requires approval
        ("supervised", "write", 0, 2, GateDecision(allow=False, simulate=False, requires_approval=True, reason="Writes require approval in supervised mode.")),

        # Autonomous + write -> allowed if under budget
        ("autonomous", "write", 0, 2, GateDecision(allow=True, simulate=False, requires_approval=False, reason="Write allowed within autonomous budget.")),
        ("autonomous", "write", 1, 2, GateDecision(allow=True, simulate=False, requires_approval=False, reason="Write allowed within autonomous budget.")),
        
        # Autonomous + write -> requires approval if at or over budget
        ("autonomous", "write", 2, 2, GateDecision(allow=False, simulate=False, requires_approval=True, reason="Autonomous write budget exceeded, requires approval.")),
        ("autonomous", "write", 3, 2, GateDecision(allow=False, simulate=False, requires_approval=True, reason="Autonomous write budget exceeded, requires approval.")),
    ]
)
def test_evaluate_gate(level: AutonomyLevel, tool_kind: ToolKind, writes_so_far: int, max_auto_writes: int, expected: GateDecision):
    decision = evaluate_gate(level, tool_kind, writes_so_far, max_auto_writes)
    
    assert decision.allow == expected.allow
    assert decision.simulate == expected.simulate
    assert decision.requires_approval == expected.requires_approval
