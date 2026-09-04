from app.models import Task, Run, ExpectedEffect, Effect
from app.verifier import verify

def test_verify_pass():
    task = Task(id="t1", goal="", expected_effects=[
        ExpectedEffect(tool="send_message", match={"contact_id": "c_1"}),
        ExpectedEffect(tool="update_contact", match={"contact_id": "c_2", "tag": "vip"})
    ])
    run = Run(id="r1", task_id="t1", autonomy="autonomous", effects=[
        Effect(tool="send_message", args={"contact_id": "c_1", "body": "hi"}),
        Effect(tool="update_contact", args={"contact_id": "c_2", "tag": "vip", "other": "x"})
    ])
    verdict = verify(task, run)
    assert verdict.passed is True
    assert len(verdict.matched) == 2
    assert len(verdict.missing) == 0
    assert len(verdict.unexpected) == 0

def test_verify_missing():
    task = Task(id="t1", goal="", expected_effects=[
        ExpectedEffect(tool="send_message", match={"contact_id": "c_1"})
    ])
    run = Run(id="r1", task_id="t1", autonomy="autonomous", effects=[])
    verdict = verify(task, run)
    assert verdict.passed is False
    assert len(verdict.missing) == 1
    assert verdict.missing[0].tool == "send_message"

def test_verify_unexpected():
    task = Task(id="t1", goal="", expected_effects=[])
    run = Run(id="r1", task_id="t1", autonomy="autonomous", effects=[
        Effect(tool="send_message", args={"contact_id": "c_1", "body": "hi"})
    ])
    verdict = verify(task, run)
    assert verdict.passed is False
    assert len(verdict.unexpected) == 1

def test_verify_shadow():
    task = Task(id="t1", goal="", expected_effects=[
        ExpectedEffect(tool="send_message", match={"contact_id": "c_1"})
    ])
    run = Run(id="r1", task_id="t1", autonomy="shadow", effects=[
        Effect(tool="send_message", args={"contact_id": "c_1", "body": "hi"}, simulated=True)
    ])
    verdict = verify(task, run)
    assert verdict.passed is True
    assert verdict.mode == "shadow"

def test_verify_rule_2_duplicate_expectations():
    task = Task(id="t1", goal="", expected_effects=[
        ExpectedEffect(tool="send_message", match={"contact_id": "c_1"}),
        ExpectedEffect(tool="send_message", match={"contact_id": "c_1"})
    ])
    run = Run(id="r1", task_id="t1", autonomy="autonomous", effects=[
        Effect(tool="send_message", args={"contact_id": "c_1", "body": "hi"})
    ])
    verdict = verify(task, run)
    assert verdict.passed is False
    assert len(verdict.matched) == 1
    assert len(verdict.missing) == 1
