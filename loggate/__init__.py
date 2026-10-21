"""
loggate - a self-training gatekeeper that decides which log sessions an LLM should see.

The pipeline is a two-stage cascade:

    log session  -->  Gate (cheap, local)  --[suspicious?]-->  LLM (expensive)
                            ^                                     |
                            |______ learns from LLM verdicts ______|

The gate starts out knowing nothing and forwards everything.  Every verdict the
LLM returns becomes a training example, so the gate gradually learns what
"normal" looks like and stops forwarding it.  No human labels are involved.

Modules
-------
data      load HDFS sessions (DeepLog format) as a stream
features  turn a session into a fixed-size numeric vector
gate      the cheap classifier + the forwarding policy
oracle    a simulated LLM (ground truth + optional noise), so experiments are reproducible
cascade   run a stream through gate + oracle and record what happened
"""

from .cascade import Cascade, run_cascade
from .gate import Gate, GateConfig
from .oracle import SimulatedLLM

__all__ = ["Cascade", "run_cascade", "Gate", "GateConfig", "SimulatedLLM"]
__version__ = "0.1.0"
