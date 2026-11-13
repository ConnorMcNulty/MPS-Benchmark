"""
metrics.py -- match detected units to ground-truth units and score them.

Matching
  Each footprint is binarized at `bin_thresh` x its own max (default 0.2), and
  the pairwise cost is 1 - IoU of the binary masks. Pairs farther apart than
  `max_centroid_dist` pixels are excluded. Hungarian assignment
  (scipy.optimize.linear_sum_assignment) picks the one-to-one matching with the
  lowest total cost, and a pair only counts as a true positive if IoU >=
  `min_iou`. This is the standard approach (e.g., CaImAn's
  register_ROIs / Giovannucci et al. 2019 use a Jaccard-based Hungarian match).

Detection scores: precision, recall, F1.
Per matched pair:  IoU, centroid distance, spatial cosine similarity of the raw
                   (non-binarized) footprints, Pearson r of the calcium traces,
                   and Pearson r of the deconvolved activity if both have it.

Optional global alignment: if the pipelines motion-correct to a template that
is offset from the ground-truth frame, every footprint is shifted by a constant.
`estimate_global_shift` finds that shift by phase correlation of the summed
footprint images and is applied before matching (reported in the output).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _binarize(res, thresh):
    A = res.A.tocsr().astype(np.float32)
    mx = np.asarray(A.max(axis=1).todense()).ravel()
    mx[mx == 0] = 1
    rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    keep = A.data >= thresh * mx[rows]
    B = A.copy()
    B.data = keep.astype(np.float32)
    B.eliminate_zeros()
    return B


def _centroids(res):
    H, W = res.dims
    A = res.A.tocsr()
    yy, xx = np.divmod(np.arange(H * W), W)
    w = np.asarray(A.sum(axis=1)).ravel()
    w[w == 0] = 1
    cy = np.asarray(A @ yy).ravel() / w
    cx = np.asarray(A @ xx).ravel() / w
    return np.stack([cy, cx], axis=1)


def summed_image(res):
    A = res.A.tocsr()
    mx = np.asarray(A.max(axis=1).todense()).ravel()
    mx[mx == 0] = 1
    norm = A.multiply(1.0 / mx[:, None])
    return np.asarray(norm.sum(axis=0)).reshape(res.dims)


def estimate_global_shift(gt, res, max_shift=20):
    """(dy, dx) to ADD to res coordinates to land on gt coordinates."""
    from numpy.fft import fft2, ifft2
    a = summed_image(gt)
    b = summed_image(res)
    F = fft2(a) * np.conj(fft2(b))
    F /= np.abs(F) + 1e-12
    r = np.real(ifft2(F))
    H, W = r.shape
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    dy = dy - H if dy > H // 2 else dy
    dx = dx - W if dx > W // 2 else dx
    if abs(dy) > max_shift or abs(dx) > max_shift:
        return (0, 0)
    return (int(dy), int(dx))


def shift_result(res, dy, dx):
    if dy == 0 and dx == 0:
        return res
    import scipy.sparse as sp
    from copy import copy
    H, W = res.dims
    A = res.A.tocoo()
    y, x = np.divmod(A.col, W)
    y2, x2 = y + dy, x + dx
    ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
    A2 = sp.csr_matrix((A.data[ok], (A.row[ok], y2[ok] * W + x2[ok])), shape=A.shape)
    out = copy(res)
    out.A = A2
    return out


def _pearson_rows(X, Y):
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    num = (X * Y).sum(axis=1)
    den = np.sqrt((X ** 2).sum(axis=1) * (Y ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def match(gt, res, bin_thresh=0.2, min_iou=0.3, max_centroid_dist=10.0):
    """Return dict with matched index pairs and per-pair IoU / distance."""
    Bg = _binarize(gt, bin_thresh)
    Br = _binarize(res, bin_thresh)
    inter = (Bg @ Br.T).toarray()
    sg = np.asarray(Bg.sum(axis=1)).ravel()
    sr = np.asarray(Br.sum(axis=1)).ravel()
    union = sg[:, None] + sr[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    cg, cr = _centroids(gt), _centroids(res)
    dist = np.sqrt(((cg[:, None, :] - cr[None, :, :]) ** 2).sum(-1))

    cost = 1.0 - iou
    cost[dist > max_centroid_dist] = 1e6
    if cost.size == 0:
        gi = ri = np.array([], dtype=int)
    else:
        gi, ri = linear_sum_assignment(cost)
    ok = (iou[gi, ri] >= min_iou) & (dist[gi, ri] <= max_centroid_dist)
    gi, ri = gi[ok], ri[ok]
    return {"gt_idx": gi, "res_idx": ri, "iou": iou[gi, ri], "dist": dist[gi, ri],
            "n_gt": gt.n, "n_res": res.n}


def _align_time(gt, res, arr_res, arr_gt):
    """Restrict both to the frames the pipeline kept (frame drops, trims)."""
    fi = res.frame_idx
    keep = (fi >= 0) & (fi < arr_gt.shape[1])
    return arr_gt[:, fi[keep]], arr_res[:, keep]


def score(gt, res, **match_kw):
    m = match(gt, res, **match_kw)
    tp = len(m["gt_idx"])
    prec = tp / res.n if res.n else 0.0
    rec = tp / gt.n if gt.n else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    pairs = {"gt_idx": m["gt_idx"], "res_idx": m["res_idx"],
             "iou": m["iou"], "centroid_dist": m["dist"]}

    # spatial cosine similarity of raw footprints
    if tp:
        Ag = gt.A[m["gt_idx"]].toarray()
        Ar = res.A[m["res_idx"]].toarray()
        num = (Ag * Ar).sum(1)
        den = np.linalg.norm(Ag, axis=1) * np.linalg.norm(Ar, axis=1)
        pairs["spatial_cos"] = np.where(den > 0, num / den, np.nan)
    else:
        pairs["spatial_cos"] = np.array([])

    if gt.C is not None and tp:
        g, r = _align_time(gt, res, res.C[m["res_idx"]], gt.C[m["gt_idx"]])
        pairs["trace_r"] = _pearson_rows(g, r)
    else:
        pairs["trace_r"] = np.full(tp, np.nan)

    if gt.S is not None and res.S is not None and tp:
        g, r = _align_time(gt, res, res.S[m["res_idx"]], gt.S[m["gt_idx"]])
        pairs["spike_r"] = _pearson_rows(g, r)
    else:
        pairs["spike_r"] = np.full(tp, np.nan)

    def med(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size else float("nan")

    summary = {
        "n_gt": gt.n, "n_detected": res.n, "true_pos": tp,
        "false_pos": res.n - tp, "false_neg": gt.n - tp,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "median_iou": med(pairs["iou"]),
        "median_centroid_dist_px": med(pairs["centroid_dist"]),
        "median_spatial_cos": med(pairs["spatial_cos"]),
        "median_trace_r": med(pairs["trace_r"]),
        "median_spike_r": med(pairs["spike_r"]),
        "n_frames_scored": int(((res.frame_idx >= 0) &
                                (res.frame_idx < (gt.C.shape[1] if gt.C is not None else 0))).sum()),
    }
    return summary, pairs


# ---------------------------------------------------------------------------
# Centroid matching, as in Ari's MPS_Ground_Truth_Comparison notebook
# ---------------------------------------------------------------------------

def _dedupe(res, spat_thresh=0.95, centroid_thresh=5.0):
    """Indices of units that are near-copies of an earlier unit (Ari's rule:
    centroids within centroid_thresh px AND footprint cosine >= spat_thresh)."""
    A = res.A.tocsr()
    c = _centroids(res)
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    is_dup = np.zeros(res.n, dtype=bool)
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    for i in range(res.n):
        if is_dup[i] or norms[i] < 1e-9:
            continue
        cand = np.where((d[i] <= centroid_thresh) & (np.arange(res.n) > i) & ~is_dup)[0]
        if cand.size:
            dots = np.asarray((A[cand] @ A[i].T).todense()).ravel()
            sc = dots / (norms[cand] * norms[i] + 1e-12)
            is_dup[cand[sc >= spat_thresh]] = True
    return is_dup


def score_centroid(gt, res, thresh_px=15.0, offset_correct=True,
                   dup_spat=0.95, dup_centroid=5.0):
    """Reproduces the notebook's scoring: Hungarian on centroid distance,
    <= thresh_px counts as a match, a median (dx, dy) offset from a first pass
    is removed before the final match, and duplicate components are left out
    of the precision denominator instead of counting as false positives."""
    cg, cr = _centroids(gt), _centroids(res)
    is_dup = _dedupe(res, dup_spat, dup_centroid)
    uniq = np.where(~is_dup)[0]

    def hung(cr_):
        cost = np.sqrt(((cr_[:, None, :] - cg[None, :, :]) ** 2).sum(-1))
        if cost.size == 0:
            return np.array([], int), np.array([], int), cost
        r, g = linear_sum_assignment(cost)
        ok = cost[r, g] <= thresh_px
        return r[ok], g[ok], cost

    dy = dx = 0.0
    if offset_correct and res.n:
        r0, g0, _ = hung(cr)
        if r0.size:
            dy = float(np.median(cr[r0, 0] - cg[g0, 0]))
            dx = float(np.median(cr[r0, 1] - cg[g0, 1]))
    crc = cr - np.array([dy, dx])
    r, g, cost = hung(crc[uniq])
    r = uniq[r]
    tp = len(r)
    n_u = len(uniq)
    prec = tp / n_u if n_u else 0.0
    rec = tp / gt.n if gt.n else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    tr = np.full(tp, np.nan)
    if gt.C is not None and tp:
        a, b = _align_time(gt, res, res.C[r], gt.C[g])
        tr = _pearson_rows(a, b)
    return {
        "centroid_precision": round(prec, 4), "centroid_recall": round(rec, 4),
        "centroid_f1": round(f1, 4), "centroid_true_pos": tp,
        "centroid_duplicates": int(is_dup.sum()),
        "centroid_offset_dy": round(dy, 2), "centroid_offset_dx": round(dx, 2),
        "centroid_median_trace_r": float(np.nanmedian(tr)) if tp else float("nan"),
    }
