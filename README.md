# MintA

Code for the multi-instance attack on GNN-based malicious domain detection (IEEE S&P 2024).
The experiments run on the public PDNS-Net dataset, which is publicly released; `data/README.md`
builds the graphs from it.

Two experiments, each at 1, 5, 20, 40, 60, 80, and 100 perturbed domains over 30 trials:

| Experiment | What is perturbed |
|---|---|
| `feature` | Domain names: each detected domain is renamed (subdomain edits, registered domain fixed); its features are recomputed from the new name. |
| `joint` | The renamed domains, then the apex edges of the adversary (greedy edge flips on a linearized GCN surrogate). |

Both score candidates with the multi-instance objective (a domain and its adversary neighbors) on a
surrogate trained only on the detector's answers for queried domains. The detector is never queried
during scoring.

## Results

`results/` holds the run on PDNS-Net: `results_table.md` (ASR and NFR with 95% confidence interval,
and detection rate before and after, per attack and perturbation budget) and `figures/` (for each
attack, `<attack>_summary` attack-summary bars and `<attack>_metrics` ASR/NFR curves). At 100
perturbed domains:

| Attack | ASR |
|---|---|
| feature | 0.718 [0.682, 0.755] |
| combined | 0.851 [0.811, 0.891] |

Numbers are over the completed trials (27 of 30; 3 did not finish within GPU memory and are recorded
as failed, not dropped). Rerunning the pipeline below regenerates `results/`. The figures redraw from
`results/results_table.md` with one command:

    cd MintA && python scripts/make_figures.py

## Install

    cd MintA
    python3.11 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt && pip install -e . --no-deps

or `conda env create -f environment.yml`. Python 3.11.

## Run

Build the data first (`data/README.md`), then:

    cd MintA
    scripts/launch.sh configs/pdns.yaml     # full graph, detached tmux session
    scripts/launch.sh configs/bdns.yaml     # 7,495-domain subset

| Command | Purpose |
|---|---|
| `scripts/launch.sh <config>` | start every stage unattended |
| `scripts/watch.sh <config>` | follow the progress line |
| `scripts/attach.sh <config>` | attach to the tmux session (Ctrl-b d to detach) |
| `scripts/status.sh <config>` | stage markers and process state |
| `scripts/stop.sh <config>` | stop the run |
| `scripts/resume.sh <config>` | continue after a stop, crash, or reboot |

The run uses the A40 or H200 when present, otherwise the CPU. For the full graph on one GPU,
`MINTA_RUN_ARGS="--trial-workers auto" scripts/launch.sh configs/pdns.yaml` runs trials in parallel,
sized by a measured probe. `configs/smoke.yaml` is a one-trial check on the subset.

## Output

Under `runs/<name>/`: `results.csv` (one row per domain and unit), `summary.csv`,
`results_table.md` (ASR, NFR, and detected domains before and after, mean and 95% interval over
trials), and `figures/` (`feature` and `joint`, PDF and PNG).

## Tests

    cd MintA && python -m pytest tests

## Verification

- Verified without the dataset or a GPU: byte-compile, `ruff`, `mypy`, and the unit and
  integration tests (seeding, deterministic order, atomic writes and torn-line recovery, resume
  after kill, lifecycle transitions, device selection, retries). The suite runs on a small
  synthetic graph.
- Requires the dataset and a GPU, run by the user: the full PDNS-Net pipeline and the numbers in
  `results/`. `configs/smoke.yaml` is a one-trial check once the subset is built.

## Citation and license

See `CITATION.cff`. MIT license.
