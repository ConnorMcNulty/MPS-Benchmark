"""
diagnose.py -- look inside one pipeline's result when a score looks wrong.

  python -m bench.diagnose config.yaml MPS
  python -m bench.diagnose config.yaml minian --rep 1

Prints, for the matched units:
  * frame bookkeeping: how many frames the result has, its frame_idx range,
    gaps and duplicates (frames dropped or relabelled by the pipeline)
  * trace correlation with the ground truth at lag 0, and the lag (-10..10)
    that maximises it. A consistent non-zero best lag, or a lag that changes
    partway through the recording, means the frames are misaligned rather
    than the traces being poor.
  * r computed separately before and after the first dropped frame
  * how much of each trace is exactly zero or constant (clipping, gaps)
  * r of the ground truth against the result's spikes (S), if present
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from bench.converters import load_any, load_groundtruth
from bench.metrics import _pearson_rows, estimate_global_shift, match, shift_result


def lag_r(x, y, lag):
    if lag > 0:
        x, y = x[lag:], y[:-lag]
    elif lag < 0:
        x, y = x[:lag], y[-lag:]
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("pipeline")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--work-dir", help="override work_dir")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dims = tuple(cfg["dims"])
    work = os.path.abspath(a.work_dir or cfg["work_dir"])
    runs = pd.read_csv(os.path.join(work, "runs.csv"))
    r = runs[(runs.pipeline == a.pipeline) & (runs.rep == a.rep)].iloc[0]
    fmt = {p["name"]: p.get("format", "standard") for p in cfg["pipelines"]}[a.pipeline]
    gt = load_groundtruth(cfg["ground_truth"], dims)
    res = load_any({"name": a.pipeline, "format": fmt, "output": r.output}, dims)
    res = shift_result(res, *estimate_global_shift(gt, res))

    fi = res.frame_idx
    print("== %s rep%d" % (a.pipeline, a.rep))
    print("units: %d   C shape: %s   S: %s" % (res.n, res.C.shape, None if res.S is None else res.S.shape))
    print("frame_idx: %d entries, min %d, max %d, unique %d" % (len(fi), fi.min(), fi.max(), len(np.unique(fi))))
    d = np.diff(fi)
    gaps = np.where(d > 1)[0]
    dups = np.where(d == 0)[0]
    back = np.where(d < 0)[0]
    print("gaps (dropped frames) after positions: %s" % (gaps[:10].tolist() or "none"))
    print("duplicate labels at positions: %s" % (dups[:10].tolist() or "none"))
    print("labels going backwards at: %s" % (back[:10].tolist() or "none"))
    if fi.max() >= gt.C.shape[1]:
        print("NOTE: %d frame labels are beyond the ground truth length %d"
              % ((fi >= gt.C.shape[1]).sum(), gt.C.shape[1]))

    m = match(gt, res, **{k: v for k, v in cfg.get("matching", {}).items() if k != "global_align"})
    gi, ri = m["gt_idx"], m["res_idx"]
    print("\nmatched units: %d" % len(gi))
    if not len(gi):
        return
    keep = (fi >= 0) & (fi < gt.C.shape[1])
    G = gt.C[gi][:, fi[keep]]
    R = res.C[ri][:, keep]

    r0 = _pearson_rows(G, R)
    print("trace r at lag 0 (frame labels as given): median %.3f, 10th pct %.3f, 90th pct %.3f"
          % (np.nanmedian(r0), np.nanpercentile(r0, 10), np.nanpercentile(r0, 90)))
    # same, ignoring the frame labels (assume column k = ground-truth frame k)
    T = min(res.C.shape[1], gt.C.shape[1])
    r_naive = _pearson_rows(gt.C[gi][:, :T], res.C[ri][:, :T])
    print("trace r if frame labels are ignored (column k = frame k): median %.3f" % np.nanmedian(r_naive))

    lags = np.arange(-10, 11)
    best = []
    for k in range(len(gi)):
        rs = [lag_r(R[k], G[k], l) for l in lags]
        best.append(lags[int(np.nanargmax(rs))])
    vals, cnt = np.unique(best, return_counts=True)
    print("best lag (res vs gt), counts: " + ", ".join("%+d: %d" % (v, c) for v, c in zip(vals, cnt)))

    if gaps.size:
        cut = gaps[0] + 1
        ra = _pearson_rows(G[:, :cut], R[:, :cut])
        rb = _pearson_rows(G[:, cut:], R[:, cut:])
        print("r before first dropped frame (%d frames): %.3f   after (%d frames): %.3f"
              % (cut, np.nanmedian(ra), G.shape[1] - cut, np.nanmedian(rb)))

    zero = (R == 0).mean(1)
    const = np.array([np.mean(np.diff(x) == 0) for x in R])
    print("fraction of trace exactly 0: median %.3f   fraction of steps with no change: median %.3f"
          % (np.median(zero), np.median(const)))
    print("trace value range (median over units): min %.3g  max %.3g"
          % (np.median(R.min(1)), np.median(R.max(1))))

    if res.S is not None:
        S = res.S[ri][:, keep]
        rs = _pearson_rows(G, S)
        print("r(ground truth, result S): median %.3f" % np.nanmedian(rs))

    worst = np.argsort(r0)[:5]
    print("\nworst 5 matched units (gt index, res index, r):")
    for k in worst:
        print("  gt %3d  res %3d  r %.3f  best lag %+d" % (gi[k], ri[k], r0[k], best[k]))


if __name__ == "__main__":
    main()
