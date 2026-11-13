"""
check_alignment.py -- confirm the decoded video lines up frame-for-frame with
the ground-truth traces, independent of which decoder read it.

  python -m bench.check_alignment config.yaml
  python -m bench.check_alignment config.yaml --video ../min1pipe_input/synthetic_5min.tif

For each ground-truth neuron it builds a trace straight from the pixels
(footprint-weighted mean of every frame) and cross-correlates it with the
ground-truth fluorescence at lags of -max_lag..+max_lag frames, separately in
the first and second half of the recording.

  every neuron peaks at lag 0 in both halves -> frames are complete and in order
  a constant non-zero lag                   -> frames missing/added at the start
  lag 0 in one half, non-zero in the other  -> frames dropped or duplicated mid-file

It also decodes the AVI a second time with a single decoder thread (FFV1's
"Invalid change of global parameters" / "non-keyframe" messages come from
multi-threaded decoding) and reports any frame that differs.
"""

import argparse
import os

import numpy as np
import yaml

from bench.converters import _footprints_to_nhw, _load_array


def read_video(path, threads=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(path) as tf:   # one page per frame
            return np.stack([pg.asarray() for pg in tf.pages])
    import cv2
    if threads is not None and hasattr(cv2, "CAP_PROP_N_THREADS"):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads])
    else:
        cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr[..., 0] if fr.ndim == 3 else fr)
    cap.release()
    return np.stack(frames)


def lagged_r(x, y, lag):
    if lag > 0:
        x, y = x[lag:], y[:-lag]
    elif lag < 0:
        x, y = x[:lag], y[-lag:]
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def best_lags(V, A, F, max_lag):
    n = A.shape[0]
    W = A.reshape(n, -1)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    Vt = V.reshape(V.shape[0], -1).astype(np.float32)
    P = Vt @ W.T                       # (T, n) pixel-derived traces
    lags = np.arange(-max_lag, max_lag + 1)
    out = []
    for i in range(n):
        if F[i].std() < 1e-9:
            out.append((np.nan, np.nan))
            continue
        rs = np.array([lagged_r(P[:, i], F[i], l) for l in lags])
        k = np.nanargmax(rs)
        out.append((lags[k], rs[k]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--video", help="default: the config's video")
    ap.add_argument("--max-lag", type=int, default=60)
    ap.add_argument("--skip-thread-check", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    gt = cfg["ground_truth"]
    dims = tuple(cfg["dims"])
    video = a.video or cfg["video"]

    V = read_video(video)
    print("decoded %d frames from %s" % (V.shape[0], video))
    A = _footprints_to_nhw(_load_array(gt["footprints"]), dims, gt.get("footprint_layout", "auto"))
    F = np.asarray(_load_array(gt["traces"]), dtype=np.float32)
    if F.shape[0] != A.shape[0]:
        F = F.T
    off = int(gt.get("frame_offset", 0))
    T = V.shape[0]
    F = F[:, off:off + T]
    if F.shape[1] < T:
        print("NOTE: ground truth has only %d frames after the offset; comparing those" % F.shape[1])
        V = V[:F.shape[1]]
        T = F.shape[1]

    h = T // 2
    res = {}
    for name, sl in (("first half", slice(0, h)), ("second half", slice(h, T))):
        bl = best_lags(V[sl], A, F[:, sl], a.max_lag)
        ok = ~np.isnan(bl[:, 0])
        lag_vals, counts = np.unique(bl[ok, 0].astype(int), return_counts=True)
        res[name] = bl
        print("\n%s (frames %d-%d): %d neurons with activity" % (name, sl.start, sl.stop - 1, ok.sum()))
        print("  best-lag counts: " + ", ".join("lag %+d: %d" % (l, c) for l, c in zip(lag_vals, counts)))
        print("  median r at best lag: %.3f" % np.nanmedian(bl[ok, 1]))

    # Sparse neurons with few events can peak at a random lag by chance, so the
    # test is on the population: at least 90% of active neurons must peak at 0.
    good = True
    for k, bl in res.items():
        l = bl[~np.isnan(bl[:, 0]), 0]
        frac0 = float(np.mean(l == 0)) if l.size else 0.0
        if frac0 < 0.9:
            good = False
        print("%s: %.0f%% of active neurons peak at lag 0" % (k, 100 * frac0))

    same = True
    if not a.skip_thread_check and os.path.splitext(video)[1].lower() == ".avi":
        import cv2
        if hasattr(cv2, "CAP_PROP_N_THREADS"):
            V1 = read_video(video, threads=1)
            print("\nsingle-thread decode: %d frames" % V1.shape[0])
            if V1.shape[0] != V.shape[0]:
                same = False
                print("  frame COUNT differs from default decode (%d vs %d)" % (V1.shape[0], V.shape[0]))
            else:
                diff = np.where((V1 != V[:V1.shape[0]]).reshape(V1.shape[0], -1).any(1))[0]
                same = diff.size == 0
                print("  identical to default decode" if same else
                      "  %d frames differ, first few: %s" % (diff.size, diff[:10].tolist()))
        else:
            print("\n(this OpenCV build cannot force single-thread decoding; skipped)")

    print()
    if good and same:
        print("OK: video frames line up with the ground truth (lag 0 throughout)")
    else:
        print("PROBLEM: see above. Do not benchmark until this is resolved.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
