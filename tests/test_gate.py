import numpy as np

from loggate.features import featurize
from loggate.gate import Gate, GateConfig

NORMAL = [5, 5, 22, 5, 11, 9, 11, 9, 11, 9, 26, 26, 26]
ANOMALY = [5, 22, 5, 5, 11, 9, 11, 9, 11, 9, 26, 26, 26] + [4] * 40   # a burst of one event


def _trained_gate(cfg=GateConfig(warmup=10, retrain_every=20)):
    gate = Gate(cfg)
    for i in range(20):
        sess = ANOMALY if i % 4 == 0 else NORMAL
        gate.observe(featurize(sess), int(sess is ANOMALY))
    return gate


def test_untrained_gate_forwards_everything():
    gate = Gate(GateConfig(warmup=0))
    fwd, reason = gate.should_forward(featurize(NORMAL))
    assert fwd and reason == "warmup"
    assert gate.score(featurize(NORMAL)) == 1.0


def test_warmup_forwards_regardless_of_score():
    gate = _trained_gate(GateConfig(warmup=5, retrain_every=20))
    for _ in range(5):
        assert gate.should_forward(featurize(NORMAL))[1] == "warmup"


def test_trained_gate_separates_normal_from_anomaly():
    gate = _trained_gate()
    assert gate.is_trained and gate.n_fits == 1
    assert gate.score(featurize(ANOMALY)) > 0.9
    assert gate.score(featurize(NORMAL)) < 0.1


def test_policy_reasons():
    gate = _trained_gate(GateConfig(warmup=0, retrain_every=20, explore=0.0))
    assert gate.should_forward(featurize(ANOMALY)) == (True, "suspicious")
    assert gate.should_forward(featurize(NORMAL)) == (False, "dropped")


def test_audit_fires_at_configured_rate():
    gate = _trained_gate(GateConfig(warmup=0, retrain_every=20, explore=0.5, seed=1))
    reasons = [gate.should_forward(featurize(NORMAL))[1] for _ in range(400)]
    audits = reasons.count("audit")
    assert 150 < audits < 250                     # ~50% +- noise
    assert set(reasons) <= {"audit", "dropped"}


def test_no_fit_with_a_single_class():
    gate = Gate(GateConfig(retrain_every=5))
    for _ in range(10):
        gate.observe(featurize(NORMAL), 0)
    assert not gate.is_trained


def test_frozen_gate_stops_learning():
    gate = _trained_gate(GateConfig(warmup=0, retrain_every=20, freeze_after_warmup=True))
    before = gate.buffer_size
    gate.observe(featurize(NORMAL), 0)
    assert gate.buffer_size == before and gate.n_fits == 1
