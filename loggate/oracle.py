"""
A simulated LLM.

Experiments need a *reproducible* expensive stage.  Calling a real LLM 575 000
times is neither cheap nor deterministic, so the oracle plays its role: it
returns the ground-truth label, optionally corrupted so we can ask "what if
the LLM is sometimes wrong?".

The two kinds of mistake are modelled separately because they are not
equally harmful to the gate:

    fn_rate   P(LLM says "normal" | session is an anomaly)   - a miss
    fp_rate   P(LLM says "anomaly" | session is normal)      - a false alarm

Anomalies are ~3% of traffic.  A false-alarm rate of even 5% therefore
produces more wrong "anomaly" labels than there are real anomalies, and the
gate is trained mostly on noise.  A miss rate of 20% just removes a fifth of
the positive examples, which the gate tolerates easily.  The README shows both.

This is a simulation and the README says so.  Everything the gate learns
comes through this interface, so swapping in a real model later is a
one-class change: implement `.judge(session, true_label) -> int` and ignore
`true_label`.
"""

from __future__ import annotations

import random
from typing import Sequence


class SimulatedLLM:
    def __init__(self, fn_rate: float = 0.0, fp_rate: float = 0.0, seed: int = 0):
        for name, v in (("fn_rate", fn_rate), ("fp_rate", fp_rate)):
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        self.fn_rate = fn_rate
        self.fp_rate = fp_rate
        self._rng = random.Random(seed)
        self.calls = 0

    def judge(self, session: Sequence[int], true_label: int) -> int:
        """Return a verdict (1 = anomaly, 0 = normal) for one session."""
        self.calls += 1
        r = self._rng.random()
        if true_label == 1 and r < self.fn_rate:
            return 0
        if true_label == 0 and r < self.fp_rate:
            return 1
        return true_label
