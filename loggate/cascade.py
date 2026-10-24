"""
Run a labelled stream through gate + LLM and record what happened.

Two numbers summarise a run:

    call rate   fraction of sessions that reached the LLM     (cost)
    recall      fraction of anomalies that reached the LLM    (safety)

"Reached the LLM" is the right notion of recall for a gatekeeper: the gate's
job is not to diagnose anomalies, it is to make sure the LLM sees them.  An
anomaly the gate drops is a miss, full stop; an anomaly the gate forwards is
caught, even if the (simulated) LLM then misjudges it.

Both numbers are reported per window of `window` sessions, so a run gives a
learning curve rather than a single average that hides the warm-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np

from .data import Example
from .features import featurize
from .gate import Gate, GateConfig
from .oracle import SimulatedLLM


@dataclass
class WindowStats:
    end: int            # index of the last session in this window
    sessions: int
    forwarded: int
    anomalies: int
    anomalies_forwarded: int
    gate_fits: int

    @property
    def call_rate(self) -> float:
        return self.forwarded / self.sessions

    @property
    def recall(self) -> float:
        return self.anomalies_forwarded / self.anomalies if self.anomalies else float("nan")


@dataclass
class RunResult:
    windows: List[WindowStats] = field(default_factory=list)
    reasons: dict = field(default_factory=dict)       # forward-reason -> count, whole run

    def steady_state(self, last_n_windows: int = 4) -> tuple[float, float]:
        """(call_rate, recall) pooled over the last few windows, i.e. after learning."""
        ws = self.windows[-last_n_windows:]
        fwd = sum(w.forwarded for w in ws); n = sum(w.sessions for w in ws)
        af = sum(w.anomalies_forwarded for w in ws); a = sum(w.anomalies for w in ws)
        return fwd / n, (af / a if a else float("nan"))


class Cascade:
    def __init__(self, gate: Gate, llm: SimulatedLLM):
        self.gate = gate
        self.llm = llm

    def step(self, example: Example) -> tuple[bool, str]:
        """Process one session.  Returns (forwarded?, reason)."""
        x = example.x if example.x is not None else featurize(example.session)
        forward, reason = self.gate.should_forward(x)
        if forward:
            # The LLM gets what a real backend would read: the raw session when
            # there is one, otherwise the feature vector.  Never None.
            observation = example.session if example.session is not None else x
            verdict = self.llm.judge(observation, example.label)
            self.gate.observe(x, verdict)
        return forward, reason


def run_cascade(
    stream: Iterable[Example],
    gate_config: GateConfig = GateConfig(),
    llm_fn_rate: float = 0.0,
    llm_fp_rate: float = 0.0,
    window: int = 25_000,
    seed: int = 0,
) -> RunResult:
    gate = Gate(gate_config)
    llm = SimulatedLLM(fn_rate=llm_fn_rate, fp_rate=llm_fp_rate, seed=seed)
    cascade = Cascade(gate, llm)
    result = RunResult()

    n = fwd = anom = anom_fwd = 0
    for i, ex in enumerate(stream, start=1):
        forward, reason = cascade.step(ex)
        result.reasons[reason] = result.reasons.get(reason, 0) + 1
        n += 1; fwd += forward; anom += ex.label; anom_fwd += ex.label and forward
        if n == window:
            result.windows.append(WindowStats(i, n, fwd, anom, anom_fwd, gate.n_fits))
            n = fwd = anom = anom_fwd = 0
    if n and result.windows and n < window // 5:
        # a trailing sliver would be a noisy, misleading point: fold it into the last window
        last = result.windows[-1]
        result.windows[-1] = WindowStats(i, last.sessions + n, last.forwarded + fwd,
                                         last.anomalies + anom, last.anomalies_forwarded + anom_fwd,
                                         gate.n_fits)
    elif n:
        result.windows.append(WindowStats(i, n, fwd, anom, anom_fwd, gate.n_fits))
    return result
