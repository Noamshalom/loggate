"""
Session -> feature vector.

The gate has to be *cheap*, so the representation is deliberately simple:
a bag of event counts.  For HDFS this is enough, and it is enough for a
well-understood reason: most HDFS anomalies are not "events in a strange
order" but "too many / too few of some event" (a block that was received
only twice instead of three times, a block that was never deleted, ten
retries of the same step).  Counts capture exactly that.

Two details that matter:

* log1p(count) instead of raw count.  A session with 200 retries should
  look "very wrong", not "200x more wrong than one retry".  Compressing
  the scale keeps the linear model stable.
* log(length) as an extra feature.  Length is the cheapest anomaly cue
  there is, and the counts alone do not encode it directly.

The vocabulary size is fixed up front (DeepLog's HDFS uses event ids 1..28)
so the vector shape never changes while the system is running.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

N_EVENT_TYPES = 29          # ids 1..28 used; index 0 stays empty, keeps indexing simple
N_FEATURES = N_EVENT_TYPES + 1   # + log(length)


def featurize(session: Sequence[int]) -> np.ndarray:
    """Return a float vector of shape (N_FEATURES,) for one session."""
    x = np.zeros(N_FEATURES, dtype=np.float64)
    for event_id in session:
        if 0 <= event_id < N_EVENT_TYPES:
            x[event_id] += 1.0
        # ids outside the known range are ignored rather than crashing:
        # a production parser may emit a new template at any time.
    x[:N_EVENT_TYPES] = np.log1p(x[:N_EVENT_TYPES])
    x[N_EVENT_TYPES] = math.log(max(len(session), 1))
    return x


def featurize_many(sessions: Sequence[Sequence[int]]) -> np.ndarray:
    """Vectorise a batch; shape (n_sessions, N_FEATURES)."""
    return np.stack([featurize(s) for s in sessions]) if sessions else np.zeros((0, N_FEATURES))
