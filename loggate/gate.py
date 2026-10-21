"""
The gate: a cheap classifier plus a forwarding policy.

The classifier is plain logistic regression.  It is the right tool here for
three reasons: it is linear (so scoring a session is one dot product, which
is the whole point of a cheap gate), it is probabilistic (so a threshold on
its output has a meaning), and on count features it is known to separate
HDFS anomalies well.  Nothing fancier is needed, and anything fancier would
be harder to defend.

The *policy* is where the design decisions live.  For every incoming session
the gate must answer one question: forward it to the LLM or not?

    forward if   (1) we are still in the warm-up period, or
                 (2) p(anomaly) > threshold, or
                 (3) a random audit fires (probability `explore`)

(1)  At the start the gate knows nothing; forwarding everything is the only
     safe choice.  Warm-up ends after `warmup` sessions.
(2)  The core rule.  `threshold` trades cost for recall: lower = more
     forwarding, fewer misses.
(3)  The subtle one.  A session the gate drops never gets a verdict, so the
     gate never finds out if it was wrong.  Without (3) a gate that starts
     dropping a *kind* of anomaly would keep dropping it forever: it can
     only learn from what it forwards.  A small random audit rate breaks the
     loop at a small, fixed cost.  Ops teams already do this by hand.

Learning: every verdict goes into a buffer; every `retrain_every` verdicts
the classifier is refit from scratch on the buffer.  Refitting (rather than
one-sample SGD) is simpler and more stable.  The buffer is a sliding window
of the last `buffer_max` verdicts, which bounds the cost of each refit and
lets the gate follow a drifting stream.  `class_weight="balanced"` matters
because anomalies are a small minority of what the gate sees.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from .features import N_FEATURES


@dataclass
class GateConfig:
    warmup: int = 2000          # sessions forwarded unconditionally at the start
    threshold: float = 0.2      # forward when p(anomaly) exceeds this
    explore: float = 0.01       # random-audit probability for sessions below threshold
    retrain_every: int = 2000   # refit after this many new verdicts
    buffer_max: int = 50_000    # keep only the most recent verdicts: bounds refit cost, follows drift
    standardize: bool = False   # z-score features at fit time (needed when feature scales differ wildly)
    freeze_after_warmup: bool = False   # ablation: stop learning after the first fit
    seed: int = 0


class Gate:
    def __init__(self, config: GateConfig = GateConfig()):
        self.cfg = config
        self._rng = random.Random(config.seed)
        self._X: Deque[np.ndarray] = deque(maxlen=config.buffer_max)   # verdict buffer (features)
        self._y: Deque[int] = deque(maxlen=config.buffer_max)          # verdict buffer (LLM labels)
        self._w: Optional[np.ndarray] = None   # fitted weights, shape (N_FEATURES,)
        self._b: float = 0.0
        self.seen = 0
        self.n_fits = 0
        self.n_verdicts = 0

    # ---- scoring ---------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._w is not None

    def score(self, x: np.ndarray) -> float:
        """p(anomaly) for one feature vector.  1.0 before the first fit (forward everything)."""
        if self._w is None:
            return 1.0
        z = float(self._w @ x + self._b)
        if z < -30.0:                  # avoid overflow in exp; p is 0 to double precision anyway
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    # ---- policy -----------------------------------------------------------

    def should_forward(self, x: np.ndarray) -> tuple[bool, str]:
        """Decide for one session.  Returns (forward?, reason) so runs are auditable."""
        self.seen += 1
        if self.seen <= self.cfg.warmup or not self.is_trained:
            return True, "warmup"
        if self.score(x) > self.cfg.threshold:
            return True, "suspicious"
        if self._rng.random() < self.cfg.explore:
            return True, "audit"
        return False, "dropped"

    # ---- learning ---------------------------------------------------------

    def observe(self, x: np.ndarray, verdict: int) -> None:
        """Record an LLM verdict and refit when enough new ones have accumulated."""
        if self.cfg.freeze_after_warmup and self.is_trained:
            return
        self._X.append(x)
        self._y.append(int(verdict))
        self.n_verdicts += 1                 # the deques drop the oldest verdict automatically
        if self.n_verdicts % self.cfg.retrain_every == 0:
            self._fit()

    def _fit(self) -> None:
        y = np.asarray(self._y)
        if y.min() == y.max():          # need both classes to fit a classifier
            return
        X = np.stack(self._X)
        mu = np.zeros(X.shape[1]); sd = np.ones(X.shape[1])
        if self.cfg.standardize:
            mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-9
        model = LogisticRegression(class_weight="balanced", max_iter=2000)
        model.fit((X - mu) / sd, y)
        # Fold the scaling into the weights, so scoring stays one dot product on raw features:
        #   w·((x - mu)/sd) + b  ==  (w/sd)·x + (b - Σ w·mu/sd)
        w = model.coef_[0].astype(np.float64) / sd
        self._w = w
        self._b = float(model.intercept_[0] - w @ mu)
        self.n_fits += 1

    @property
    def buffer_size(self) -> int:
        return len(self._y)
