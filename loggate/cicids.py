"""
CICIDS2017 network flows as a security scenario.

CICIDS2017 (Canadian Institute for Cybersecurity) recorded a week of traffic on
a small corporate network while a red team ran real attacks against it.  Each
row of the "MachineLearningCVE" CSVs is one network flow (a connection),
summarised by CICFlowMeter into 78 numeric features (durations, packet counts,
byte counts, inter-arrival times, TCP flag counts, ...) plus a label.

We use two days, in order, to ask the question a security team actually cares
about: *what happens when an attack the gate has never seen shows up?*

    phase 1  Tuesday               benign + FTP/SSH password brute-force (3.1% attacks)
    phase 2  Friday afternoon      benign + DDoS  -- an attack class absent from phase 1

Preprocessing choices, each of which is a deliberate decision:

* Destination port is dropped.  Every Tuesday attack targets port 21 or 22, so
  a gate given the port learns "port 22 = attack" and nothing about behaviour.
  That is a shortcut an attacker defeats by moving a port.
* Features are signed-log transformed, sign(v)·log(1+|v|).  Byte and packet
  counts span ten orders of magnitude; a few CICFlowMeter columns can be
  negative; a few are inf/NaN (division by a zero duration), which we set to 0.
* Friday is 57% DDoS in the raw file, because the red team ran a flood.  A
  triage system never sees that ratio in its normal life, and at 57% "forward
  everything" is nearly optimal, which would make the experiment meaningless.
  We keep all benign Friday flows and subsample DDoS to 3% of the stream, the
  same base rate as Tuesday.  The subsampling is seeded and reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .data import Example

TUESDAY = "Tuesday-WorkingHours_pcap_ISCX.csv"
FRIDAY_DDOS = "Friday-WorkingHours-Afternoon-DDos_pcap_ISCX.csv"
DROP_COLUMNS = ("Label", "Destination Port")


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def flow_features(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """Signed-log transform of the numeric flow features; inf/NaN -> 0."""
    a = df[columns].to_numpy(dtype=np.float64, copy=True)   # copy: pandas may hand back a read-only view
    a[~np.isfinite(a)] = 0.0
    return np.sign(a) * np.log1p(np.abs(a))


def _examples(df: pd.DataFrame, columns: List[str]) -> List[Example]:
    X = flow_features(df, columns)
    labels = (df["Label"] != "BENIGN").to_numpy()
    tags = df["Label"].to_numpy()
    return [Example(None, int(y), X[i], str(t)) for i, (y, t) in enumerate(zip(labels, tags))]


def load_phases(
    data_dir: Path | str,
    attack_rate: float = 0.03,
    seed: int = 0,
) -> Tuple[List[Example], List[Example]]:
    """
    Return (tuesday, friday) as two shuffled lists of Examples.

    attack_rate  target fraction of DDoS flows in the Friday stream (benign kept whole).
    seed         controls both shuffles and the DDoS subsample.
    """
    data_dir = Path(data_dir)
    for name in (TUESDAY, FRIDAY_DDOS):
        if not (data_dir / name).exists():
            raise FileNotFoundError(
                f"{data_dir / name} not found. See README: download MachineLearningCSV.zip "
                "from the CICIDS2017 page and put these two CSVs in data/cicids2017/."
            )
    tue = _read(data_dir / TUESDAY)
    fri = _read(data_dir / FRIDAY_DDOS)
    columns = [c for c in tue.columns if c not in DROP_COLUMNS]

    benign = fri[fri["Label"] == "BENIGN"]
    attacks = fri[fri["Label"] != "BENIGN"]
    k = int(round(len(benign) * attack_rate / (1.0 - attack_rate)))
    fri = pd.concat([benign, attacks.sample(n=min(k, len(attacks)), random_state=seed)])

    tue = tue.sample(frac=1.0, random_state=seed)
    fri = fri.sample(frac=1.0, random_state=seed)
    return _examples(tue, columns), _examples(fri, columns)
