import argparse
import os
import re
import json
import glob
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import preprocessing


def _discover_run_paths(runs_root: Path, results_root: Path) -> Dict[str, Path]:
    """Return mapping from run name -> path.

    We treat the *root* of `results_for_the_blog_post` as the historical baseline run named
    "blog".  Additionally, we include every numeric sub-directory in `runs/` and in
    `results_for_the_blog_post/` as separate runs.
    """
    run_paths: Dict[str, Path] = {}

    # Baseline blog run (aggregated outputs live directly in the root dir)
    if results_root.exists():
        run_paths["blog"] = results_root

    # Runs inside runs/
    if runs_root.exists():
        for p in runs_root.iterdir():
            if p.is_dir() and re.fullmatch(r"\d+", p.name):
                run_paths[p.name] = p

    # Numeric sub-dirs inside results_for_the_blog_post
    for p in results_root.iterdir():
        if p.is_dir() and re.fullmatch(r"\d+", p.name):
            run_paths[f"blog_{p.name}"] = p

    return run_paths


def _compute_accuracy(pred_file: Path) -> Tuple[float, float]:
    """Compute top-1 and top-2 accuracy (mean across tasks) at the final iteration."""
    if not pred_file.exists():
        return np.nan, np.nan

    # Infer split from filename
    m = re.search(r"predictions_(.*?)\.npz", pred_file.name)
    split = m.group(1) if m else "training"

    # Load true hashes for the split
    task_nums = list(range(400))
    tasks = preprocessing.preprocess_tasks(split, task_nums)
    true_hashes = [t.solution_hash >> 16 for t in tasks]

    data = np.load(pred_file, allow_pickle=True)
    histories = data["solution_picks_histories"]

    n_tasks = len(histories)
    assert n_tasks == len(true_hashes), (
        f"Predictions file contains {n_tasks} tasks but preprocess produced {len(true_hashes)} tasks for split '{split}'."
    )

    final_top1 = []
    final_top2 = []
    for task_idx, task_hist in enumerate(histories):
        last_pair = task_hist[-1]  # pair of two hashed ints
        first_hash = int(last_pair[0]) >> 16
        second_hash = int(last_pair[1]) >> 16
        true_hash = true_hashes[task_idx]
        final_top1.append(1 if first_hash == true_hash else 0)
        final_top2.append(1 if (first_hash == true_hash or second_hash == true_hash) else 0)

    return float(np.mean(final_top1)), float(np.mean(final_top2))


def _collect_curve_metrics(run_dir: Path) -> Tuple[float, float, float]:
    """Return mean final total_loss, reconstruction_error and total_KL across tasks.

    Searches for all *KL_curves.npz files inside the run directory recursively.
    """
    kl_files = list(run_dir.rglob("*_KL_curves.npz"))
    if not kl_files:
        return np.nan, np.nan, np.nan

    final_total_losses = []
    final_recons = []
    final_kls = []

    for f in kl_files:
        data = np.load(f, allow_pickle=True)
        recon_curve = data.get("reconstruction_error_curve")
        recon_final = float(recon_curve[-1]) if recon_curve is not None else np.nan

        kl_curves_obj = data.get("KL_curves")
        if kl_curves_obj is None:
            total_kl_final = np.nan
        else:
            kl_curves_dict = kl_curves_obj.item()
            total_kl_final = float(sum(curve[-1] for curve in kl_curves_dict.values()))

        total_loss_final = recon_final + total_kl_final if not np.isnan(recon_final) and not np.isnan(total_kl_final) else np.nan

        final_recons.append(recon_final)
        final_kls.append(total_kl_final)
        final_total_losses.append(total_loss_final)

    # Compute mean across tasks
    mean_total_loss = float(np.nanmean(final_total_losses))
    mean_recon = float(np.nanmean(final_recons))
    mean_kl = float(np.nanmean(final_kls))
    return mean_total_loss, mean_recon, mean_kl


def _parse_time(run_dir: Path) -> float:
    """Read timing_result*.txt and return seconds (float) or NaN."""
    for fname in ["timing_result.txt", "timing_result_training.txt", "timing_result_evaluation.txt"]:
        fpath = run_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    m = re.search(r"([0-9.]+)", line)
                    if m:
                        return float(m.group(1))
    return float("nan")


def build_report(run_paths: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for run_name, path in sorted(run_paths.items()):
        # Accuracy – pick training predictions if available, else any predictions_*.npz
        pred_files = list(path.glob("predictions_training.npz"))
        if not pred_files:
            pred_files = list(path.glob("predictions_*.npz"))
        acc_top1, acc_top2 = _compute_accuracy(pred_files[0]) if pred_files else (np.nan, np.nan)

        # Loss / KL
        mean_total_loss, mean_recon, mean_kl = _collect_curve_metrics(path)

        # Time
        time_s = _parse_time(path)

        rows.append({
            "Run": run_name,
            "mean_total_loss": mean_total_loss,
            "mean_recon_loss": mean_recon,
            "mean_KL": mean_kl,
            "mean_acc_top1": acc_top1,
            "mean_acc_top2": acc_top2,
            "time_s": time_s,
        })

    df = pd.DataFrame(rows)
    return df


def save_report(df: pd.DataFrame, out_dir: Path):
    df_sorted = df.sort_values("Run")
    # Markdown table
    try:
        md_table = df_sorted.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        # Fallback if "tabulate" is not available.
        md_table = df_sorted.to_string(index=False)
    with open(out_dir / "summary.md", "w") as f:
        f.write(md_table)
    # CSV
    df_sorted.to_csv(out_dir / "summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training (optional) and build comparison report.")
    parser.add_argument("--no-train", action="store_true", help="Skip training step and only build the report.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    runs_root = project_root / "runs"
    results_root = project_root / "results_for_the_blog_post"

    # Future: trigger training here if not args.no_train (not yet implemented)
    if not args.no_train:
        print("Training step not implemented; for now please use --no-train to only build report.")

    run_paths = _discover_run_paths(runs_root, results_root)
    if not run_paths:
        raise RuntimeError("No runs found to summarise.")

    df_report = build_report(run_paths)
    # For now, save in project root; when training added we'll place inside new run folder
    save_report(df_report, project_root)
    print("Report generated:\n")
    try:
        print(df_report.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        print(df_report.to_string(index=False)) 