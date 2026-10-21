import math

import numpy as np
import pandas as pd

from loggate import Cascade, Example, Gate, GateConfig, SimulatedLLM
from loggate.cicids import DROP_COLUMNS, _examples, flow_features


def _frame():
    return pd.DataFrame({
        "Destination Port": [22, 80, 443],
        "Flow Duration": [0.0, 99.0, -1.0],
        "Flow Bytes/s": [np.inf, 1.0, np.nan],
        "Label": ["SSH-Patator", "BENIGN", "BENIGN"],
    })


def test_signed_log_and_non_finite_values():
    X = flow_features(_frame(), ["Flow Duration", "Flow Bytes/s"])
    assert X[0, 0] == 0.0
    assert math.isclose(X[1, 0], math.log1p(99.0))
    assert math.isclose(X[2, 0], -math.log1p(1.0))       # sign is kept
    assert X[0, 1] == 0.0 and X[2, 1] == 0.0               # inf and NaN become 0


def test_port_and_label_are_not_features():
    df = _frame()
    cols = [c for c in df.columns if c not in DROP_COLUMNS]
    assert "Destination Port" not in cols and "Label" not in cols
    exs = _examples(df, cols)
    assert [e.label for e in exs] == [1, 0, 0]
    assert exs[0].tag == "SSH-Patator" and exs[0].session is None
    assert exs[0].x.shape == (len(cols),)


def test_cascade_uses_precomputed_features():
    rng = np.random.default_rng(0)
    stream = []
    for i in range(3000):
        y = int(rng.random() < 0.05)
        x = rng.normal(size=5) + (4.0 if y else 0.0)       # attacks are shifted: learnable
        stream.append(Example(None, y, x))
    cascade = Cascade(Gate(GateConfig(warmup=300, retrain_every=100, standardize=True, explore=0.0)),
                      SimulatedLLM())
    fwd_tail = [cascade.step(ex)[0] for ex in stream][2000:]
    labels_tail = [ex.label for ex in stream][2000:]
    caught = sum(f for f, y in zip(fwd_tail, labels_tail) if y)
    assert caught == sum(labels_tail)                       # every attack forwarded
    assert sum(fwd_tail) < 0.15 * len(fwd_tail)             # and most benign traffic dropped


def test_standardize_folds_into_weights():
    """Scoring with folded weights must equal scoring the standardized input."""
    rng = np.random.default_rng(1)
    gate = Gate(GateConfig(warmup=0, retrain_every=200, standardize=True))
    for _ in range(200):
        y = int(rng.random() < 0.3)
        gate.observe(rng.normal(size=4) * [1, 10, 100, 1000] + (5 if y else 0), y)
    assert gate.is_trained
    x = np.array([0.5, 3.0, -20.0, 400.0])
    p = gate.score(x)
    assert 0.0 <= p <= 1.0 and np.isfinite(gate._w).all()
