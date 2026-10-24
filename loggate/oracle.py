"""
A simulated LLM.

Experiments need a *reproducible* expensive stage.  Calling a real LLM 575 000
times is neither cheap nor deterministic, so the oracle plays its role: it
returns the ground-truth label, optionally corrupted so we can ask "what if
the LLM is sometimes wrong?".

The two kinds of mistake are modelled separately because they are not
equally harmful to the gate:

    fn_rate   P(LLM says "normal" | event is an attack)   - a miss
    fp_rate   P(LLM says "attack" | event is normal)      - a false alarm

What this simulation does NOT establish: that a real language model can
produce good labels from the observations used here (an HDFS event sequence,
or 77 flow statistics).  The gate experiments are valid as a study of a
filter learning from partial, possibly noisy feedback; whether an LLM is a
good source of that feedback for this data is an open question, stated in
the README.

Interface: `judge(observation, true_label)`.  The observation is what a real
backend would read (the raw session, or the feature vector for flows); the
true label is what the simulation uses.  A real backend ignores `true_label`.
"""

from __future__ import annotations

import random
from typing import Any


class SimulatedLLM:
    def __init__(self, fn_rate: float = 0.0, fp_rate: float = 0.0, seed: int = 0):
        for name, v in (("fn_rate", fn_rate), ("fp_rate", fp_rate)):
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        self.fn_rate = fn_rate
        self.fp_rate = fp_rate
        self._rng = random.Random(seed)
        self.calls = 0

    def judge(self, observation: Any, true_label: int) -> int:
        """Return a verdict (1 = attack, 0 = normal) for one event."""
        if observation is None:
            raise ValueError("judge() needs the observation; a real backend would read it")
        self.calls += 1
        r = self._rng.random()
        if true_label == 1 and r < self.fn_rate:
            return 0
        if true_label == 0 and r < self.fp_rate:
            return 1
        return true_label
