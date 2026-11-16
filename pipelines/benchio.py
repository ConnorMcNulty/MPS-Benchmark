"""
benchio.py -- tiny helpers imported by the per-pipeline wrapper scripts.
Only needs numpy + scipy, so it works inside every pipeline's own environment.
Copy it next to the wrapper scripts (it already is, in pipelines/).

stage(name)        context manager that writes start/end timestamps for a
                   pipeline stage to the file named by $BENCH_STAGE_FILE
                   (set by bench/monitor.py). No-op if the variable is unset.

save_standard(...) writes the common result format that bench/evaluate.py
                   reads, so every pipeline ends up in the same shape:
                     A : (n_units, height, width) footprints, full-frame pixels
                     C : (n_units, n_frames) denoised calcium traces
                     S : (n_units, n_frames) deconvolved activity (optional)
                     frame_idx : original frame index of every column of C
"""

import json
import os
import time
from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp


@contextmanager
def stage(name):
    path = os.environ.get("BENCH_STAGE_FILE")
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        print("[bench] stage %-24s %.1fs" % (name, t1 - t0), flush=True)
        if path:
            with open(path, "a") as f:
                f.write(json.dumps({"stage": name, "start": t0, "end": t1,
                                    "dur_s": round(t1 - t0, 3)}) + "\n")


def save_standard(path, A, C, S=None, frame_idx=None, dims=None, meta=None):
    """A may be dense (n, h, w), dense (n, h*w), or scipy sparse (n, h*w) / (h*w, n).

    dims = (height, width) of the full original frame. Required if A is 2-D.
    For a 2-D A, pixel order must be row-major (C order): pixel = y * width + x.
    CaImAn stores A as (pixels, n) in Fortran order; convert before calling
    (run_caiman.py does this).
    """
    C = np.asarray(C, dtype=np.float32)
    n = C.shape[0]
    if sp.issparse(A):
        A = A.tocsr()
        if A.shape[0] != n and A.shape[1] == n:
            A = A.T.tocsr()
        if dims is None:
            raise ValueError("dims required for 2-D A")
    else:
        A = np.asarray(A)
        if A.ndim == 3:
            dims = A.shape[1:]
            A = A.reshape(A.shape[0], int(dims[0]) * int(dims[1]))
        elif dims is None:
            raise ValueError("dims required for 2-D A")
        if A.shape[0] != n and A.shape[1] == n:
            A = A.T
        A = sp.csr_matrix(A.astype(np.float32))
    if A.shape != (n, int(dims[0]) * int(dims[1])):
        raise ValueError("A shape %s does not match n=%d, dims=%s" % (A.shape, n, dims))
    if frame_idx is None:
        frame_idx = np.arange(C.shape[1])
    out = dict(
        A_data=A.data.astype(np.float32), A_indices=A.indices, A_indptr=A.indptr,
        A_shape=np.array(A.shape), dims=np.array(dims, dtype=int),
        C=C, frame_idx=np.asarray(frame_idx, dtype=int),
        meta=json.dumps(meta or {}),
    )
    if S is not None:
        out["S"] = np.asarray(S, dtype=np.float32)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(path, **out)
    print("[bench] saved %d units, %d frames -> %s" % (n, C.shape[1], path), flush=True)
