# ARC-Compressor Model Comparison Framework

This document outlines a lightweight workflow for benchmarking **alternative model specifications** (e.g.
different `layers.py` variants) against the current baseline.  It focuses on *final* metrics after a fixed
training budget, rather than full learning curves.

---

## 1. Outputs required from each training run

| Artifact | Purpose | Notes |
|----------|---------|-------|
| `predictions_<split>.npz` | holds `solution_contribution_logs` & `solution_picks_histories`; used to compute top-1 / top-2 accuracy | Already produced by `solution_selection.save_predictions()`.
| `*_KL_curves.npz` (or equivalent) | contains `loss_curve`, `total_KL_curve`, `reconstruction_error_curve` | These files already exist in `results_for_the_blog_post/` and will be reused.  Only the **final element** of each curve is needed. |
| `timing_result.txt` | wall-clock time for the whole run | Keep as is, but it will be copied into the run folder. |

All other metrics are derived from these three artefacts, so no extra saving step is necessary.

### Required code additions

1. **Auto-incremented run folder**  
   Each invocation writes to `runs/000`, `runs/001`, … (next integer not yet present).  The new folder is created automatically at the start of the run, and _all_ artefacts are copied/moved into it when training finishes.
2. **Single combined script (`run_and_report.py`)**  
   – Executes the training loop (logic currently in `train.py`).  
   – Copies/renames raw outputs into the newly created run folder.  
   – Immediately computes final metrics _across all existing run folders_ **and those in `results_for_the_blog_post/`** and writes a report table (`summary.md` _and_ `summary.csv`) into the same folder.
3. **Accuracy helper**  
   Function to compute point-wise match percentage (returns 0 if predicted grid shape ≠ true grid shape).
4. **Readable report**  
   The script emits a Markdown table identical to the sample output below and saves it as `summary.md` inside the run folder.  A CSV copy (`summary.csv`) enables spreadsheet usage.

---

## 2. Integrated training + reporting script (`run_and_report.py`)

### Command-line usage
```bash
python run_and_report.py          # trains with current layers.py, saves to next numeric folder and updates the global summary
python run_and_report.py --no-train   # skip training, just regenerate report from existing folders
```
*No layer-switch flag is required; you simply edit `layers.py` before running the script.*

### What the script does
1. Determine `run_id = max(existing_ids) + 1` where `existing_ids` includes both `runs/*` and numeric folders inside `results_for_the_blog_post/`.
2. Execute the training routine (importing `take_step` etc.) for the chosen split and task list.
3. Move / rename these files into `runs/{run_id:03d}/`:  
   • `predictions_training.npz` (and/or evaluation)  
   • `*_KL_curves.npz`  
   • `timing_result.txt`
4. Call the internal **report builder** that iterates over every folder found in
   `runs/` **plus** the historical ones in `results_for_the_blog_post/` and computes:  
   * mean & median of final `total_loss`, `reconstruction_error`, `total_KL` across tasks  
   * mean & median of top-1 and top-2 accuracy  
   * total wall-clock time per run
5. Save the resulting comparison table both as `summary.md` (for humans) and `summary.csv` (for tooling) _inside the newly created folder_.

Sample `summary.md` (generated automatically):
```
| Run | mean_total_loss | mean_recon_loss | mean_KL | mean_acc_top1 | mean_acc_top2 | time_s |
|-----|-----------------|-----------------|---------|---------------|---------------|--------|
| 000 | 123.4 | 12.3 | 111.1 | 0.72 | 0.79 | 841.2 |
| 001 | 118.9 | 10.5 | 108.4 | 0.75 | 0.81 | 867.5 |
```

---

## 3. End-to-end workflow

1. **Edit `layers.py` to your new architecture.**
2. Run:
   ```bash
   python run_and_report.py
   ```
3. Inspect `runs/00X/summary.md` for the updated comparison.  If the changes are not beneficial, revert `layers.py` and repeat; the next script execution becomes run `00X+1` automatically.

---

## 4. Current gaps in `train.py`

* Uses fixed filenames.  The wrapper script will _move_ those files, so no internal change is strictly required.
* Loss/KL/reconstruction curves are available in existing `*_KL_curves.npz` files found in historical data, so no additional persistence step is necessary.
* Point-wise accuracy can be computed from `predictions_<split>.npz`; no change inside training is needed.

---
