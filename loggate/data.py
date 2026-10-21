"""
Loading HDFS sessions in the DeepLog format.

A *session* is the ordered list of log-event types produced while HDFS handled
one block (e.g. "allocate -> receive -> receive -> receive -> ... -> delete").
The DeepLog authors already parsed the raw 11M-line log into these sequences,
one session per line, event types as small integers:

    5 5 22 5 11 9 11 9 11 9 26 26 26 ...

Three files, same format:
    hdfs_train          4 855 normal sessions   (DeepLog's training split)
    hdfs_test_normal  553 366 normal sessions
    hdfs_test_abnormal 16 838 anomalous sessions

We do not use DeepLog's train/test split: the whole point of loggate is that
there is no offline training phase.  We merge everything into one labelled
stream, shuffle it with a fixed seed, and replay it as if it were live traffic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np

Session = List[int]

FILES = {
    "hdfs_train": 0,            # label 0 = normal
    "hdfs_test_normal": 0,
    "hdfs_test_abnormal": 1,    # label 1 = anomaly
}


@dataclass(frozen=True, eq=False)
class Example:
    session: Optional[Session]  # raw event sequence (HDFS); None when features are precomputed
    label: int                  # ground truth; only the oracle is allowed to look at it
    x: Optional[np.ndarray] = None   # precomputed feature vector (network flows)
    tag: str = ""               # attack type or other annotation, for reporting only


def read_sessions(path: Path) -> List[Session]:
    """Parse one DeepLog file: whitespace-separated ints, one session per line."""
    sessions: List[Session] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                sessions.append([int(tok) for tok in line.split()])
    return sessions


def load_stream(data_dir: Path | str, seed: int = 0, limit: int | None = None) -> List[Example]:
    """
    Load all HDFS sessions as one shuffled, labelled list.

    seed   controls the shuffle, so experiments are reproducible.
    limit  optionally truncate (useful for tests and quick runs).
    """
    data_dir = Path(data_dir)
    examples: List[Example] = []
    for name, label in FILES.items():
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/get_data.sh to download the HDFS sessions."
            )
        examples.extend(Example(s, label) for s in read_sessions(path))
    random.Random(seed).shuffle(examples)
    return examples[:limit] if limit else examples


def iter_stream(examples: Sequence[Example]) -> Iterator[Example]:
    """Yield examples one at a time, as a live system would receive them."""
    yield from examples
