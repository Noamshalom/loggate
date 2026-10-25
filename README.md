# loggate

![tests](https://img.shields.io/badge/tests-22%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

**A simulation study of a self-training gate in front of an expensive verifier: how a cheap filter that learns only from the verifier's verdicts behaves on real security traffic, and what happens when an attack it has never seen arrives.**

> **TL;DR** - In simulation, a logistic-regression gate that learns only from
> the verdicts of an oracle "LLM" forwards ~4% of HDFS log sessions at 99.6%
> gate recall, and ~5-11% of CICIDS2017 network flows at 99% gate recall on
> known attacks. Against an attack class absent from its training traffic, the
> gate with no random audit forwarded **zero** DDoS flows in 100k flows; with a
> 5% audit and fast retraining it adapted (90% of DDoS forwarded) after ~17k
> flows. The audit rate is a dial with a cost on each side. **The LLM is
> simulated throughout**; whether a real LLM can supply these verdicts for this
> data is the main open question.

Security teams are starting to put LLMs on alert triage. An LLM reads one
event well; it cannot read the hundreds of thousands a network produces per
day, and at per-call pricing nobody wants it to. `loggate` puts a cheap local
classifier in front of the expensive stage and lets it learn **from that
stage's own verdicts** what routine traffic looks like, so it can stop
forwarding it. No human labels, no offline training: the gate forwards
everything at first and takes over the traffic as it learns.

This repo studies that mechanism with a simulated verifier. It does not test a
real LLM. See *What this does and does not show*.

```
event ──► Gate (one dot product) ──[suspicious?]──► LLM (expensive)
             ▲                                          │
             └──────── learns from every verdict ◄──────┘
```

Two questions, two datasets:

1. **How much does it save?** On 575k HDFS log sessions the gate ends up
   forwarding **4.0%** of traffic while **99.6%** of anomalies still reach the verifier.
2. **What happens when a *new kind* of attack arrives?** On CICIDS2017 network
   traffic a gate trained during password brute-force attacks meets a DDoS
   absent from its training traffic. With no random audit it forwarded no DDoS
   flow at all in the 100k flows tested. The audit rate is the dial: with fast
   retraining and a **5% audit**, the gate was forwarding 90% of DDoS flows
   after **~17k Friday flows** (15-20k over 3 seeds), at a routine cost of 11%
   of traffic.

![unseen attack](results/security_ddos.png)

## How it works

The gate is logistic regression over a feature vector - event-type counts for
log sessions, CICFlowMeter flow statistics for network traffic. Scoring is one
dot product, which is the point of a *cheap* gate.

The forwarding policy is three rules, in order:

1. **Warm-up.** The first `warmup` events are forwarded unconditionally.
2. **Threshold.** Forward if `p(attack) > threshold`.
3. **Random audit.** Forward a random `explore` fraction of what would have
   been dropped. A dropped event never gets a verdict, so without the audit
   the gate can only learn from what it already forwards. Ops teams do this by
   hand; here it is a parameter.

Every verdict goes into a sliding buffer of the last `buffer_max` verdicts;
every `retrain_every` verdicts the classifier is refit with
`class_weight="balanced"`. For network flows, whose features span ten orders
of magnitude, standardization is applied at fit time and folded back into the
weights, so scoring stays one dot product on raw features.

The LLM is **simulated**: an oracle that returns the ground-truth label,
optionally with a configurable miss rate and false-alarm rate, which keeps
575k "LLM calls" reproducible and free. The oracle receives the same
observation a real backend would (the raw log session, or the flow feature
vector), so a real model can be plugged in behind the same interface
(`loggate/oracle.py`). Plugging one in has **not** been done; see open questions.

## Result 1 - an attack the gate has never seen (CICIDS2017)

CICIDS2017 recorded a week of traffic on a small corporate network while a
red team ran real attacks. We replay two days in order:

| phase | traffic | attack | share |
|---|---|---|---:|
| Tuesday | 446k flows | FTP / SSH password brute-force | 3.1% |
| Friday afternoon | 101k flows | DDoS - absent from Tuesday | 3.0% (subsampled, see below) |

Destination port is **excluded** from the features: every Tuesday attack hits
port 21 or 22, and a gate that learns "port 22 = attack" has learned a shortcut,
not behaviour. Flow statistics are signed-log transformed. Friday is 57% DDoS
in the raw capture because the red team ran a flood. At that rate "forward
everything" is nearly optimal and the experiment says nothing, so we subsample
DDoS to 3%, matching Tuesday's attack rate. This is an experimental choice,
not a claim about real-world attack rates; other rates and bursts are untested.

Mean over 3 seeds. *Routine cost* is the share of Tuesday traffic forwarded
after the gate has learned. *Flows to adapt* is the amount of Friday traffic
processed before the first 5k-flow window in which at least 90% of DDoS flows
were forwarded. It is a count of flows, not a clock time, and it is **not**
the time to the first forwarded DDoS flow: with any audit, individual DDoS
flows reach the verifier from the first window on (the plot shows this). The
Friday stream is shuffled within the day, so this measures adaptation to a
new class, not reaction to a burst.

| retraining | audit | routine cost (Tue) | brute-force forwarded | Friday cost | flows to adapt (90% DDoS forwarded) |
|---|---:|---:|---:|---:|---|
| slow | 0% | 5.4% | 99.4% | 1.0% | not reached in 100k |
| slow | 1% | 6.5% | 99.4% | 2.5% | not reached in 100k |
| slow | 2% | 7.6% | 99.4% | 3.9% | not reached in 100k |
| slow | 5% | 10.5% | 99.4% | 8.7% | 62k (55-75k) |
| fast | 0% | 2.6% | 78.4% | 0.1% | not reached in 100k |
| fast | 1% | 6.7% | 99.4% | 4.0% | not reached in 2 of 3 seeds (75k in the third) |
| fast | 2% | 7.9% | 99.4% | 5.9% | 63k (45-85k) |
| fast | 5% | 11.0% | 99.5% | 9.5% | **17k (15-20k)** |

*Slow* = refit every 2 000 verdicts on the last 50 000; *fast* = every 200 on
the last 10 000. Note that these two settings differ in **both** refit cadence
and memory size, so the slow/fast difference cannot be attributed to either
one alone. Ranges are min-max over seeds.

Three things to read off this table:

* **With no audit, the gate forwarded no DDoS flow in 100k flows.** It
  learned Tuesday's attacks well and forwards ~1% of Friday, all of it benign.
  DDoS flows never score above threshold, never get a verdict, and the gate
  gets no signal that it is wrong. Whether this would persist beyond 100k
  flows is not tested.
* **The audit rate buys adaptation speed.** With fast retraining, going from
  a 2% to a 5% audit costs 3 more points of routine traffic and cuts flows to
  adapt from ~63k to ~17k. At 1%, the gate usually did not reach 90% within
  the day.
* **Fast retraining without an audit is worse than slow, on known attacks
  too.** With no audit, the fast gate forwards only 78% of the brute-force
  attacks it was trained on (vs 99% for the slow gate). A gate that only sees
  what it already chose to forward narrows its own view of the world: the
  selective-labels loop the audit exists to break.

For a product this is one dial with a price on each side: audit traffic costs
verifier calls every day; slower adaptation costs whatever undetected DDoS
costs. Neither price is in this repo; the trade-off curve is.

## Result 2 - how much it saves at scale (HDFS)

575 059 real HDFS log sessions (2.93% anomalous), replayed as one shuffled
stream. Steady state, seed 0:

| threshold | sessions forwarded | anomalies reaching LLM |
|----------:|-------------------:|-----------------------:|
| 0.5 | 3.93% | 98.27% |
| 0.2 (default) | 4.00% | 99.63% |
| 0.02 | 4.11% | 99.93% |

![learning curve](results/learning_curve.png)

The gate goes from forwarding everything to ~4% within the first 10k sessions
and stays on the anomaly-rate floor for the remaining 565k.

**What if the verifier is wrong?** Two recall columns: *gate recall* is the
share of anomalies the gate forwards, which is what the gate controls.
*End-to-end* is the share of anomalies that were forwarded **and** then
flagged by the verifier. It is counted per event in the run, not computed
from the miss rate, and it is what a user of the whole system would see under
the simulation's assumptions. The first number alone overstates the system.

All rows: full 575k stream, seed 0, steady state over the last 100k sessions
(`scripts/run_experiments.py`, `robustness()`).

| verifier behaviour | sessions forwarded | gate recall | end-to-end recall |
|---|---:|---:|---:|
| perfect verifier (threshold 0.2) | 4.00% | 99.63% | 99.63% |
| misses 20% of anomalies | 4.05% | 99.77% | 79.85% |
| misses 40% of anomalies | 3.90% | 94.20% | 54.70% |
| false-alarms on 1% of normals | 4.04% | 99.87% | 99.87% |
| false-alarms on 5% of normals, threshold 0.2 | 100.00% | 100.00% | 100.00% |
| false-alarms on 5% of normals, threshold 0.5 | 3.97% | 99.33% | 99.33% |

Two readings:

* **Verifier misses degrade the gate gracefully, but they are not free.** A
  verifier that overlooks 20% of anomalies leaves the gate's recall intact
  (it just sees fewer positives), but the system as a whole then detects ~80%,
  not ~100%.
* **The 5% false-alarm collapse depends on the threshold.** At the default
  threshold 0.2 the gate forwards everything; at threshold 0.5 it does not.
  An earlier version of this README explained the collapse as a base-rate
  inevitability (more wrong "anomaly" labels than real anomalies). That
  explanation was wrong: the same label noise does not collapse the gate at
  0.5. The gate's scores are not calibrated probabilities
  (`class_weight="balanced"` inflates them), and 0.2 is simply too low for a
  noisy teacher. The 0.5 setting was chosen after seeing the 0.2 result, so it
  shows that collapse is avoidable, not that 0.5 is the right threshold in
  general. (A first check of this on the first 150k sessions gave 3.95% /
  99.8%; the table reports the full stream.)

Full tables (threshold sweep, robustness, ablations): `results/*.csv`.

## What is and isn't new here

The building blocks are well known. A cheap model in front of an expensive one
is a *cascade*; the random audit is *exploration*; a learner that only sees the
cases it chose to inspect suffers from *selective labels*. None of that is
original to this repo. What this repo adds is the application and the
measurement: using the verifier's verdicts as the only training signal for a
security triage gate, and quantifying how much traffic such a gate needs
before it adapts to a new attack class, as a function of the audit rate, on
real attack traffic. The numbers in the tables are the contribution; the
algorithm is not.

## What this does and does not show

Shown, in simulation:

* A gate trained only on a verifier's verdicts can forward a few percent of
  traffic while forwarding ~99% of known anomalies (HDFS, CICIDS2017).
* Without a random audit, the gate forwarded no flow of an attack class
  absent from its training traffic, over 100k flows; audit rate and retrain
  cadence control how fast it adapts.
* The gate tolerates a verifier that misses anomalies; a verifier that
  false-alarms on 5% of normals collapses the gate at threshold 0.2 but not at
  0.5.

Not shown:

* **That a real LLM can supply these verdicts.** The verifier is an oracle.
  Whether a language model can label an HDFS event sequence or 77 flow
  statistics accurately enough, and whether it adds anything over a plain
  classifier on this data, is untested. This is the central open question.
* **Clock time.** "Flows to adapt" counts flows in a shuffled stream.
* **That the results hold at other attack rates.** DDoS was subsampled to a 3%
  base rate; the raw Friday capture is 57% DDoS. Bursts were not tested.
* **Which of refit cadence or memory size drives the slow/fast difference.**
* **That the thresholds are well chosen.** Both 0.2 and 0.5 were picked after
  seeing results on the same data.

## Open questions

In rough order of importance:

1. Replace the oracle with a real model on a small labelled sample, with the
   observation it would actually receive, and measure its miss and false-alarm
   rates. Then rerun the robustness experiment with those rates.
2. Separate refit cadence from memory size (vary one at a time).
3. Compare random audit with uncertainty sampling at the same verifier budget.
4. Choose thresholds on a held-out split and evaluate on a separate one.
5. Test adaptation under a DDoS burst (the raw 57% rate) and at other base rates.

## Caveats

* **Lab data.** CICIDS2017 is a testbed capture, not a customer network; HDFS
  anomalies are system faults, not attacks, and are nearly linearly separable.
* **One transition.** "Never seen" is tested on brute-force followed by DDoS.
  Other transitions may behave differently.
* **Gate recall is "reached the verifier".** The gate's job is to make sure
  the verifier sees attacks, not to diagnose them. End-to-end numbers are
  given where they differ.

## Run it

```bash
pip install -e ".[dev]"
pytest                                   # 22 tests, no data needed

scripts/get_data.sh                      # HDFS sessions, ~30 MB
python scripts/run_experiments.py        # Result 2, ~10 min

# Result 1: download MachineLearningCSV.zip from
# https://www.unb.ca/cic/datasets/ids-2017.html and put these two files in data/cicids2017/:
#   Tuesday-WorkingHours.pcap_ISCX.csv   Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
python scripts/run_security.py           # 8 configs x 3 seeds, ~1 h on 2 cores; resumable
```

```python
from loggate import Gate, GateConfig, SimulatedLLM, Cascade
from loggate.cicids import load_phases

tuesday, friday = load_phases("data/cicids2017")
cascade = Cascade(Gate(GateConfig(explore=0.05, standardize=True)), SimulatedLLM())
for flow in tuesday + friday:
    forwarded, reason = cascade.step(flow)   # reason in {warmup, suspicious, audit, dropped}
```

## Layout

```
loggate/
  gate.py       classifier + forwarding policy (warm-up / threshold / audit)
  cascade.py    run a stream through gate + LLM, record per-window stats
  oracle.py     simulated LLM with miss and false-alarm rates
  data.py       HDFS sessions as a labelled, shuffled stream
  features.py   log session -> count vector
  cicids.py     CICIDS2017 flows -> features; Tuesday / Friday phases
scripts/        get_data.sh, run_experiments.py (HDFS), run_security.py (CICIDS)
tests/          pytest, no data needed
results/        CSVs and plots from the full runs
```

## Author's note

Built with AI assistance (Claude). The problem framing, experimental design,
decisions along the way and the analysis are mine; the AI helped write and
debug the code and run the experiments. I'm a CS and entrepreneurship student
at Reichman University and happy to talk about any of it.

## Data

* CICIDS2017, Canadian Institute for Cybersecurity, University of New Brunswick
  ([unb.ca/cic/datasets/ids-2017](https://www.unb.ca/cic/datasets/ids-2017.html)).
  Not redistributed here; download requires a name and e-mail.
* HDFS sessions preprocessed by the DeepLog authors
  ([wuyifan18/DeepLog](https://github.com/wuyifan18/DeepLog)) from Loghub
  ([logpai/loghub](https://github.com/logpai/loghub)).
