"""
evaluate.py -- score every finished run against the ground truth and make the
summary tables and figures.

  python -m bench.evaluate config.yaml
  python -m bench.evaluate config.yaml --out results/   (default: <work_dir>/evaluation)

Reads <work_dir>/runs.csv written by run_benchmark.py. Writes:
  accuracy_per_run.csv   detection + trace metrics for every run
  pairs/<pipe>_rep<k>.csv  one row per matched unit (IoU, distance, trace r ...)
  summary.csv            per pipeline: mean and SD over repeats of every metric
  fig_resources.png      wall time and peak memory per pipeline
  fig_accuracy.png       precision / recall / F1 and trace-r distributions
  fig_memory_timeline.png  memory over time, one line per pipeline (rep 1)
  fig_footprints_<pipe>.png  ground-truth vs detected outlines on the mean GT image
"""

import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.converters import load_any, load_groundtruth
from bench.metrics import estimate_global_shift, score, score_centroid, shift_result, summed_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out")
    ap.add_argument("--runs", help="runs.csv path (default <work_dir>/runs.csv)")
    ap.add_argument("--work-dir", help="override work_dir")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    work_dir = os.path.abspath(a.work_dir or cfg["work_dir"])
    out = a.out or os.path.join(work_dir, "evaluation")
    os.makedirs(os.path.join(out, "pairs"), exist_ok=True)
    dims = tuple(cfg["dims"])
    mcfg = dict(cfg.get("matching", {}))
    global_align = mcfg.pop("global_align", True)
    ccfg = cfg.get("centroid_matching", {"thresh_px": 15.0})

    runs = pd.read_csv(a.runs or os.path.join(work_dir, "runs.csv"))
    gt = load_groundtruth(cfg["ground_truth"], dims)
    print("ground truth: %d units, %d frames" % (gt.n, gt.C.shape[1] if gt.C is not None else 0))
    fmt = {p["name"]: p.get("format", "standard") for p in cfg["pipelines"]}

    acc_rows = []
    loaded = {}
    for _, r in runs.iterrows():
        tag = "%s rep%d" % (r.pipeline, r.rep)
        if r.exit_code != 0 or r.timed_out:
            print("skip %s (exit %s, timed_out %s)" % (tag, r.exit_code, r.timed_out))
            acc_rows.append({"pipeline": r.pipeline, "rep": r.rep, "status": "failed"})
            continue
        try:
            res = load_any({"name": r.pipeline, "format": fmt.get(r.pipeline, "standard"),
                            "output": r.output}, dims)
        except Exception as exc:
            print("could not load %s: %s" % (tag, exc))
            acc_rows.append({"pipeline": r.pipeline, "rep": r.rep, "status": "load_error: %s" % exc})
            continue
        shift = (0, 0)
        if global_align:
            shift = estimate_global_shift(gt, res)
            res = shift_result(res, *shift)
        summ, pairs = score(gt, res, **mcfg)
        if ccfg is not None:
            summ.update(score_centroid(gt, res, **ccfg))
        summ.update({"pipeline": r.pipeline, "rep": r.rep, "status": "ok",
                     "global_shift_dy": shift[0], "global_shift_dx": shift[1]})
        acc_rows.append(summ)
        pd.DataFrame(pairs).to_csv(os.path.join(out, "pairs", "%s_rep%d.csv" % (r.pipeline, r.rep)),
                                   index=False)
        if r.pipeline not in loaded:
            loaded[r.pipeline] = (res, pairs)
        print("%-12s IoU-match F1 %.3f  P %.3f  R %.3f | centroid-match F1 %s | trace r %.3f  shift %s"
              % (tag, summ["f1"], summ["precision"], summ["recall"], summ.get("centroid_f1"),
                 summ["median_trace_r"], shift))

    acc = pd.DataFrame(acc_rows)
    acc.to_csv(os.path.join(out, "accuracy_per_run.csv"), index=False)

    merged = runs.merge(acc, on=["pipeline", "rep"], how="left")
    metrics = ["wall_s", "peak_rss_gb", "peak_uss_gb", "n_detected", "precision", "recall", "f1",
               "median_iou", "median_spatial_cos", "median_trace_r", "median_spike_r",
               "centroid_precision", "centroid_recall", "centroid_f1", "centroid_duplicates",
               "centroid_median_trace_r"]
    metrics = [m for m in metrics if m in merged and merged[m].notna().any()]
    ok = merged[merged.get("status", "ok") == "ok"]
    summary = ok.groupby("pipeline")[metrics].agg(["mean", "std"])
    summary.columns = ["%s_%s" % c for c in summary.columns]
    summary["n_ok_runs"] = ok.groupby("pipeline").size()
    summary["n_runs"] = merged.groupby("pipeline").size()
    summary.to_csv(os.path.join(out, "summary.csv"))
    print("\n", summary.round(3).to_string())

    order = [p["name"] for p in cfg["pipelines"] if p["name"] in set(merged.pipeline)]
    _fig_resources(merged, order, out)
    if not ok.empty:
        _fig_accuracy(ok, order, out)
    _fig_timeline(runs, order, out)
    for name, (res, pairs) in loaded.items():
        _fig_footprints(gt, res, pairs, out)
    print("\nwrote results to", out)


def _bar(ax, df, order, col, ylabel):
    g = df.groupby("pipeline")[col]
    m = g.mean().reindex(order)
    s = g.std().reindex(order).fillna(0)
    ax.bar(range(len(order)), m.values, yerr=s.values, capsize=4, color="#9bb7d4",
           edgecolor="#3a5a80")
    for i, p in enumerate(order):
        vals = df.loc[df.pipeline == p, col].dropna().values
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.12, 0.12, len(vals)), vals,
                   s=12, color="#3a5a80", zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def _fig_resources(df, order, out):
    ok = df[(df.exit_code == 0) & (~df.timed_out.astype(bool))]
    mem_col = "peak_uss_gb" if ok["peak_uss_gb"].notna().any() else "peak_rss_gb"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    _bar(axes[0], ok.assign(wall_min=ok.wall_s / 60), order, "wall_min", "Wall time (min)")
    _bar(axes[1], ok, order, mem_col, "Peak memory, %s (GB)" % ("USS" if "uss" in mem_col else "RSS"))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_resources.png"), dpi=200)
    plt.close(fig)


def _fig_accuracy(ok, order, out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, col in zip(axes[:3], ["precision", "recall", "f1"]):
        _bar(ax, ok, order, col, col.capitalize() if col != "f1" else "F1")
        ax.set_ylim(0, 1.05)
    ax = axes[3]
    data = []
    for p in order:
        f = os.path.join(out, "pairs", "%s_rep1.csv" % p)
        data.append(pd.read_csv(f)["trace_r"].dropna().values if os.path.exists(f) else np.array([]))
    # a violin needs at least two values; show single points (or nothing) otherwise
    for i, d in enumerate(data, start=1):
        d = np.asarray(d, dtype=float)
        d = d[np.isfinite(d)]
        if len(d) >= 2 and np.ptp(d) > 0:
            ax.violinplot([d], positions=[i], showmedians=True)
        elif len(d):
            ax.scatter(np.full(len(d), i), d, s=12, color="#3a5a80")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_ylabel("Trace correlation, matched units")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_accuracy.png"), dpi=200)
    plt.close(fig)


def _fig_timeline(runs, order, out):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for p in order:
        r = runs[(runs.pipeline == p) & (runs.rep == 1)]
        if r.empty:
            continue
        f = os.path.join(r.iloc[0].run_dir, "memory_timeline.csv")
        if not os.path.exists(f):
            continue
        tl = pd.read_csv(f)
        col = "uss_gb" if tl["uss_gb"].notna().any() else "rss_gb"
        ax.plot(tl.t_s / 60, tl[col], label=p, lw=1.2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Memory, process tree (GB)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_memory_timeline.png"), dpi=200)
    plt.close(fig)


def _fig_footprints(gt, res, pairs, out):
    from bench.metrics import _binarize
    H, W = gt.dims
    bg = summed_image(gt)
    Bg = _binarize(gt, 0.2)
    Br = _binarize(res, 0.2)
    matched_r = set(np.asarray(pairs["res_idx"]).tolist())
    matched_g = set(np.asarray(pairs["gt_idx"]).tolist())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(bg, cmap="gray_r", alpha=0.25)

    def outline(B, idxs, color, ls):
        for i in idxs:
            m = np.asarray(B[i].todense()).reshape(H, W)
            if m.any():
                ax.contour(m, levels=[0.5], colors=[color], linewidths=0.7, linestyles=ls)

    outline(Bg, [i for i in range(gt.n) if i in matched_g], "#4a8f5a", "solid")
    outline(Bg, [i for i in range(gt.n) if i not in matched_g], "#c0504d", "solid")
    outline(Br, [i for i in range(res.n) if i not in matched_r], "#e3a33b", "dashed")
    ax.set_title("%s: green = found, red = missed, orange dashed = false positive" % res.name,
                 fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_footprints_%s.png" % res.name), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
