"""
Independent random streams derived from one master seed.

Every source of randomness in an experiment (the data shuffle, the gate's
random audit, the simulated verifier's errors) must be independent of the
others.  Seeding several `random.Random` objects with the *same* integer
makes them produce *identical* sequences, which silently couples decisions
that should be unrelated, e.g. which events get audited and which verdicts
the verifier gets wrong.  An earlier version of this repo had exactly that
coupling.

`derive(master, component)` gives each component its own stream via numpy's
SeedSequence, so a run is still fully reproducible from one number.

The gate keeps the master seed itself; the data shuffle and the verifier use
derived streams.  (CICIDS2017 sampling goes through pandas, i.e. numpy's
generator, which was already independent of Python's `random`.)
"""

from __future__ import annotations

import numpy as np

_COMPONENTS = {"data": 1, "oracle": 2}


def derive(master: int, component: str) -> int:
    """A seed for `component` that is independent of `master` and of every other component."""
    key = _COMPONENTS[component]
    return int(np.random.SeedSequence(entropy=master, spawn_key=(key,)).generate_state(1)[0])
