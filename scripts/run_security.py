#!/usr/bin/env python3
"""
Security scenario: how long does the gate stay blind to an attack it has never seen?

Phase 1 (Tuesday): benign traffic + FTP/SSH brute force.  The gate learns from LLM verdicts.
Phase 2 (Friday):  benign traffic + DDoS, an attack class absent from phase 1.

For each configuration (audit rate x retraining cadence) and each seed we record:
    routine cost      % of Tuesday flows forwarded (second half, after learning)
    Tuesday recall    % of brute-force flows forwarded
    Friday cost       % of Friday flows forwarded
    time to detect    Friday flows elapsed before DDoS recall first reaches 90% in a window

Usage:  python scripts/run_security.py [--seeds 0 1 2] [--workers 2]
Writes results/security_summary.csv, results/security_windows.csv, results/security_ddos.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from loggate import Cascade, Gate, GateConfig, SimulatedLLM
from loggate.cicids import load_phases

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cicids2017"
RESULTS = ROOT / "results"
WINDOW = 5_000
DETECT_AT = 0.90

# (label, retrain_every, buffer_max) x audit rates
CADENCES = {"slow retrain (every 2000 verdicts, 50k memory)": (2000, 50_000),
            "fast retrain (every 200 verdicts, 10k memory)": (200, 10_000)}
AUDITS = (0.0, 0.01, 0.02, 0.05)

_cache: dict = {}


def _phases(seed):
    if seed not in _cache:
        _cache.clear()
        _cache[seed] = load_phases(DATA, seed=seed)
    return _cache[seed]


def run_one(args):
    cadence, explore, seed = args
    retrain, buf = CADENCES[cadence]
    tue, fri = _phases(seed)
    gate = Gate(GateConfig(explore=explore, retrain_every=retrain, buffer_max=buf,
                           standardize=True, seed=seed))
    cascade = Cascade(gate, SimulatedLLM(seed=seed))

    # phase 1: measure only the second half, after the gate has learned
    half = len(tue) // 2
    n = f = a = af = 0
    for i, ex in enumerate(tue):
        fwd, _ = cascade.step(ex)
        if i >= half:
            n += 1; f += fwd; a += ex.label; af += ex.label and fwd
    tue_calls, tue_recall = f / n, af / a

    # phase 2: per-window recall on the unseen attack
    windows = []
    n = f = a = af = 0
    fri_f = 0
    for i, ex in enumerate(fri, start=1):
        fwd, _ = cascade.step(ex)
        n += 1; f += fwd; a += ex.label; af += ex.label and fwd; fri_f += fwd
        if n == WINDOW or i == len(fri):
            windows.append((i, f / n, af / a if a else float("nan")))
            n = f = a = af = 0
    detect = next((end for end, _, r in windows if r >= DETECT_AT), None)
    return dict(cadence=cadence, explore=explore, seed=seed, tue_calls=tue_calls,
                tue_recall=tue_recall, fri_calls=fri_f / len(fri), detect=detect,
                windows=windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    # Each finished run is appended to a checkpoint file, so an interrupted batch resumes.
    ckpt = RESULTS / "security_runs.jsonl"
    runs = [json.loads(l) for l in open(ckpt)] if ckpt.exists() else []
    done = {(r["cadence"], r["explore"], r["seed"]) for r in runs}
    jobs = [(c, e, s) for s in args.seeds for c in CADENCES for e in AUDITS if (c, e, s) not in done]
    print(f"{len(runs)} runs loaded from checkpoint, {len(jobs)} to run", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool, open(ckpt, "a") as fh:
        for r in pool.imap_unordered(run_one, jobs):
            runs.append(r)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            d = "never" if r["detect"] is None else f"{r['detect']//1000}k"
            print(f"[{time.time()-t0:4.0f}s] seed={r['seed']} {r['cadence'][:4]} audit={r['explore']:<4} "
                  f"tue {100*r['tue_calls']:.2f}%/{100*r['tue_recall']:.2f}%  "
                  f"fri {100*r['fri_calls']:.2f}%  detect {d}", flush=True)

    # ---- per-run windows
    with open(RESULTS / "security_windows.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cadence", "audit", "seed", "friday_flows_seen", "call_rate", "ddos_recall"])
        for r in sorted(runs, key=lambda r: (r["cadence"], r["explore"], r["seed"])):
            for end, c, rc in r["windows"]:
                w.writerow([r["cadence"], r["explore"], r["seed"], end, f"{c:.4f}", f"{rc:.4f}"])

    # ---- summary, aggregated over seeds
    rows = []
    for c in CADENCES:
        for e in AUDITS:
            rs = [r for r in runs if r["cadence"] == c and r["explore"] == e]
            det = [r["detect"] for r in rs]
            never = sum(d is None for d in det)
            det_str = ("never" if never == len(det) else
                       f"{np.mean([d for d in det if d is not None])/1000:.0f}k"
                       + (f" ({never}/{len(det)} seeds never)" if never else ""))
            rows.append([c, e, np.mean([r["tue_calls"] for r in rs]), np.mean([r["tue_recall"] for r in rs]),
                         np.mean([r["fri_calls"] for r in rs]), det_str, len(rs)])
    with open(RESULTS / "security_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cadence", "audit", "tuesday_call_rate", "tuesday_recall", "friday_call_rate",
                    "ddos_detected_90pct_after_flows", "seeds"])
        for row in rows:
            w.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}", row[5], row[6]])
    print("\nsummary (mean over seeds):")
    for row in rows:
        print(f"  {row[0][:4]} audit={row[1]:<4}  routine cost {100*row[2]:5.2f}%  brute-force recall {100*row[3]:6.2f}%  "
              f"friday cost {100*row[4]:5.2f}%  DDoS>=90% after {row[5]}")

    # ---- plot: DDoS recall over Friday, fast retrain, one line per audit rate (mean, min-max band)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = {0.0: "tab:gray", 0.01: "tab:orange", 0.02: "tab:blue", 0.05: "tab:green"}
    for ax, c in zip(axes, CADENCES):
        for e in AUDITS:
            rs = [r for r in runs if r["cadence"] == c and r["explore"] == e]
            xs = np.array([end for end, _, _ in rs[0]["windows"]]) / 1000
            R = np.array([[rc for _, _, rc in r["windows"]] for r in rs]) * 100
            ax.plot(xs, R.mean(0), color=colors[e], lw=2, label=f"audit {int(100*e)}%")
            ax.fill_between(xs, R.min(0), R.max(0), color=colors[e], alpha=0.15)
        ax.axhline(100 * DETECT_AT, color="black", ls=":", lw=1)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("Friday flows seen (thousands)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("DDoS flows reaching the LLM (%)")
    axes[0].legend(loc="center right")
    fig.suptitle("A gate trained on brute-force meets a DDoS it has never seen", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "security_ddos.png", dpi=150)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
