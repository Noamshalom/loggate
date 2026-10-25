import random

from loggate import GateConfig, run_cascade
from loggate.data import Example
from loggate.oracle import SimulatedLLM

NORMAL = [5, 5, 22, 5, 11, 9, 11, 9, 11, 9, 26, 26, 26]
ANOMALY = NORMAL + [4] * 40


def synthetic_stream(n=3000, anomaly_rate=0.03, seed=0):
    rng = random.Random(seed)
    return [Example(ANOMALY, 1) if rng.random() < anomaly_rate else Example(NORMAL, 0) for _ in range(n)]


def test_oracle_error_rates():
    llm = SimulatedLLM(fn_rate=0.5, fp_rate=0.0, seed=0)
    misses = sum(llm.judge(NORMAL, 1) == 0 for _ in range(2000))
    assert 900 < misses < 1100
    assert all(llm.judge(NORMAL, 0) == 0 for _ in range(100))   # fp_rate 0 -> never false alarm
    assert llm.calls == 2100


def test_oracle_refuses_missing_observation():
    import pytest
    with pytest.raises(ValueError):
        SimulatedLLM().judge(None, 1)


def test_cascade_hands_the_llm_a_real_observation():
    """Regression: with precomputed features the LLM used to receive None."""
    import numpy as np
    seen = []

    class Spy(SimulatedLLM):
        def judge(self, observation, true_label):
            seen.append(observation)
            return super().judge(observation, true_label)

    from loggate import Cascade, Gate
    cascade = Cascade(Gate(GateConfig(warmup=5)), Spy())
    cascade.step(Example(None, 0, np.ones(3)))          # flow-style example
    cascade.step(Example(NORMAL, 0))                    # session-style example
    assert isinstance(seen[0], np.ndarray) and seen[0].shape == (3,)
    assert seen[1] == NORMAL


def test_cascade_learns_and_cuts_calls_without_losing_anomalies():
    stream = synthetic_stream()
    res = run_cascade(stream, GateConfig(warmup=200, retrain_every=100), window=500)
    call_rate, recall = res.steady_state(last_n_windows=2)
    assert recall == 1.0
    assert call_rate < 0.10          # ~3% anomalies + 1% audit, with slack
    assert res.reasons["warmup"] == 200
    assert res.windows[0].call_rate == 1.0 or res.windows[0].call_rate > call_rate   # it learned


def test_windows_cover_the_whole_stream():
    stream = synthetic_stream(n=1234)
    res = run_cascade(stream, GateConfig(warmup=50, retrain_every=50), window=500)
    assert [w.sessions for w in res.windows] == [500, 500, 234]
    assert res.windows[-1].end == 1234
    assert sum(res.reasons.values()) == 1234


def test_end_to_end_recall_is_measured_not_computed():
    """With a verifier that misses 30%, counted detections must land near 70% of
    forwarded anomalies.  The stream is built with a different seed from the
    oracle's: two RNGs seeded alike draw the same numbers, which would make the
    oracle's miss decision equal the draw that created the anomaly."""
    rng = random.Random(1)
    stream = [Example(ANOMALY, 1) if rng.random() < 0.03 else Example(NORMAL, 0) for _ in range(6000)]
    res = run_cascade(stream, GateConfig(warmup=200, retrain_every=100), llm_fn_rate=0.3,
                      window=1000, seed=0)
    _, gate_recall = res.steady_state()
    e2e = res.steady_state_end_to_end()
    assert gate_recall == 1.0
    assert 0.55 < e2e < 0.85
    assert all(w.anomalies_detected <= w.anomalies_forwarded for w in res.windows)
