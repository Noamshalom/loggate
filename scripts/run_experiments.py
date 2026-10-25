#!/usr/bin/env python3
"""
Run the three experiments reported in the README and write results/ .

  1. learning curve      call rate and recall per window, default config
  2. threshold sweep     cost-vs-recall trade-off at steady state
  3. robustness          same, with an LLM that is wrong 5% / 10% / 20% of the time,
                         plus the frozen-gate ablation

Usage:  python scripts/run_experiments.py [--limit N] [--data DIR]
Runs on the full 575k-session stream in a few minutes on a laptop.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loggate import GateConfig, run_cascade
from loggate.data import load_stream

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def learning_curve(stream, window):
    window = min(window, 5_000)          # fine windows: the drop happens in the first ~10k sessions
    res = run_cascade(stream, GateConfig(), window=window)
    with open(RESULTS / "learning_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sessions_seen", "call_rate", "recall", "gate_fits"])
        for ws in res.windows:
            w.writerow([ws.end, f"{ws.call_rate:.4f}", f"{ws.recall:.4f}", ws.gate_fits])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    xs = [ws.end / 1000 for ws in res.windows]
    ax1.plot(xs, [100 * ws.call_rate for ws in res.windows], color="tab:blue", lw=2)
    ax1.axhline(100 * sum(e.label for e in stream) / len(stream), color="gray", ls="--", lw=1,
                label="anomaly rate (floor)")
    ax1.set_ylabel("sessions forwarded to LLM (%)")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right")
    ax1.set_title("Gate learns from LLM verdicts and takes over the traffic")
    ax2.plot(xs, [100 * ws.recall for ws in res.windows], color="tab:red", lw=2)
    ax2.set_ylabel("anomalies reaching LLM (%)")
    ax2.set_ylim(94, 100.5)
    ax2.set_xlabel(f"sessions seen (thousands, log scale; one point per {window} sessions)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curve.png", dpi=150)
    cr, rc = res.steady_state(last_n_windows=20)
    print(f"[learning curve] steady state: call rate {100*cr:.2f}%  recall {100*rc:.2f}%  reasons={res.reasons}")
    return res


def threshold_sweep(stream, window, thresholds=(0.5, 0.3, 0.2, 0.1, 0.05, 0.02)):
    rows = []
    for t in thresholds:
        res = run_cascade(stream, GateConfig(threshold=t), window=window)
        cr, rc = res.steady_state()
        rows.append((t, cr, rc))
        print(f"[sweep] threshold={t:<5} call rate {100*cr:.2f}%  recall {100*rc:.2f}%")
    with open(RESULTS / "threshold_sweep.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["threshold", "call_rate", "recall"])
        for t, cr, rc in rows: w.writerow([t, f"{cr:.4f}", f"{rc:.4f}"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([100 * r[1] for r in rows], [100 * r[2] for r in rows], "o-", color="tab:green")
    for t, cr, rc in rows:
        ax.annotate(f"τ={t}", (100 * cr, 100 * rc), textcoords="offset points", xytext=(6, -4), fontsize=8)
    ax.set_xlabel("sessions forwarded to LLM (%)  - cost")
    ax.set_ylabel("anomalies reaching LLM (%)  - recall")
    ax.set_title("Cost vs recall at steady state (threshold sweep)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "threshold_sweep.png", dpi=150)
    return rows


def robustness(stream, window):
    """
    LLM error modes.  Two recall numbers per row:
      gate recall        share of anomalies the gate forwards (what the gate controls)
      end-to-end recall  share of anomalies that were forwarded AND flagged by the
                         verifier.  Counted per event, not computed from the miss
                         rate.  This is what a user of the whole system sees; the
                         gate's number alone overstates it.
    The 5% false-alarm case is run at two thresholds because the outcome depends
    on it (see README): the default 0.2 collapses, 0.5 does not.
    """
    rows = []
    settings = [("perfect LLM", 0.0, 0.0, 0.2),
                ("LLM misses 10% of anomalies", 0.10, 0.0, 0.2),
                ("LLM misses 20% of anomalies", 0.20, 0.0, 0.2),
                ("LLM misses 40% of anomalies", 0.40, 0.0, 0.2),
                ("LLM false-alarms on 1% of normals", 0.0, 0.01, 0.2),
                ("LLM false-alarms on 5% of normals, threshold 0.2", 0.0, 0.05, 0.2),
                ("LLM false-alarms on 5% of normals, threshold 0.5", 0.0, 0.05, 0.5)]
    for name, fn, fp, thr in settings:
        res = run_cascade(stream, GateConfig(threshold=thr), llm_fn_rate=fn, llm_fp_rate=fp, window=window)
        cr, rc = res.steady_state()
        e2e = res.steady_state_end_to_end()
        rows.append((name, cr, rc, e2e))
        print(f"[robustness] {name:<50} call rate {100*cr:6.2f}%  gate recall {100*rc:6.2f}%  end-to-end {100*e2e:6.2f}%")
    for name, cfg in (("frozen after first fit", GateConfig(freeze_after_warmup=True)),
                      ("no random audit (explore=0)", GateConfig(explore=0.0))):
        res = run_cascade(stream, cfg, window=window)
        cr, rc = res.steady_state()
        rows.append((name, cr, rc, res.steady_state_end_to_end()))
        print(f"[ablation] {name:<50} call rate {100*cr:6.2f}%  gate recall {100*rc:6.2f}%")
    with open(RESULTS / "robustness.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["setting", "call_rate", "gate_recall", "end_to_end_recall"])
        for s_, cr, rc, e2e in rows: w.writerow([s_, f"{cr:.4f}", f"{rc:.4f}", f"{e2e:.4f}"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N sessions")
    ap.add_argument("--window", type=int, default=25_000)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    t0 = time.time()
    stream = load_stream(args.data, limit=args.limit)
    n_anom = sum(e.label for e in stream)
    print(f"loaded {len(stream)} sessions, {n_anom} anomalies ({100*n_anom/len(stream):.2f}%) in {time.time()-t0:.1f}s")

    learning_curve(stream, args.window)
    threshold_sweep(stream, args.window)
    robustness(stream, args.window)
    print(f"done in {time.time()-t0:.0f}s; see {RESULTS}/")


if __name__ == "__main__":
    main()
