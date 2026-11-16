"""
converters.py -- load every pipeline's output, and the ground truth, into one
common in-memory form so metrics never need to know which tool made it.

    Result.A          scipy.sparse CSR, shape (n_units, height*width), row-major pixels
    Result.dims       (height, width) of the full original frame
    Result.C          (n_units, n_frames) float32 calcium traces
    Result.S          (n_units, n_frames) or None
    Result.frame_idx  original frame index of each column of C/S

Loaders:
    load_standard(path)        .npz written by pipelines/benchio.save_standard
                               (minian, CaImAn, MIN1PIPE wrappers all write this)
    load_mps(result_dir, dims) MPS native export: numpy_files/A_final.npy etc.
    load_groundtruth(spec)     flexible loader for the synthetic ground truth
"""

import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp


@dataclass
class Result:
    name: str
    A: sp.csr_matrix
    dims: tuple
    C: np.ndarray
    S: Optional[np.ndarray]
    frame_idx: np.ndarray

    @property
    def n(self):
        return self.A.shape[0]

    def footprint(self, i):
        return np.asarray(self.A[i].todense()).reshape(self.dims)


# ---------------------------------------------------------------------------
# Standard format
# ---------------------------------------------------------------------------

def load_standard(path, name=None):
    z = np.load(path, allow_pickle=False)
    A = sp.csr_matrix((z["A_data"], z["A_indices"], z["A_indptr"]),
                      shape=tuple(z["A_shape"]))
    S = z["S"] if "S" in z.files else None
    return Result(name or os.path.basename(path), A, tuple(int(d) for d in z["dims"]),
                  z["C"].astype(np.float32), S, z["frame_idx"].astype(int))


# ---------------------------------------------------------------------------
# MPS native output
# ---------------------------------------------------------------------------

def _find(result_dir, fname):
    hits = glob.glob(os.path.join(result_dir, "**", fname), recursive=True)
    if not hits:
        raise FileNotFoundError("%s not found under %s" % (fname, result_dir))
    # most recent if several runs are present
    return max(hits, key=os.path.getmtime)


def load_mps(result_dir, dims, name="MPS"):
    """Read MPS step 8c NumPy export and place footprints back in the full frame.

    MPS crops in step 3a with xarray .isel(), which keeps the ORIGINAL height /
    width coordinate labels, and drops frames in step 2d the same way. The
    *_coords.json files record those labels, so we use them directly instead of
    re-deriving crop offsets. Requires 'export_npy' on in step 8c.
    """
    A_path = _find(result_dir, "A_final.npy")
    npy_dir = os.path.dirname(A_path)
    A = np.load(A_path)
    with open(os.path.join(npy_dir, "A_final_coords.json")) as f:
        a_meta = json.load(f)
    C = np.load(os.path.join(npy_dir, "C_final.npy"))
    with open(os.path.join(npy_dir, "C_final_coords.json")) as f:
        c_meta = json.load(f)
    S_path = os.path.join(npy_dir, "S_final.npy")
    S = np.load(S_path) if os.path.exists(S_path) else None

    a_dims = a_meta["dims"]
    # Put A in (unit_id, height, width) order
    order = [a_dims.index(d) for d in ("unit_id", "height", "width")]
    A = np.transpose(A, order)
    ys = np.asarray(a_meta["coords"]["height"])
    xs = np.asarray(a_meta["coords"]["width"])
    if not (np.issubdtype(ys.dtype, np.integer) or np.allclose(ys, np.round(ys))):
        raise ValueError("MPS height/width coords are not integer pixel labels; "
                         "downsampling may be on. Map coordinates manually.")
    ys = np.round(ys).astype(int)
    xs = np.round(xs).astype(int)
    H, W = dims
    full = np.zeros((A.shape[0], H, W), dtype=np.float32)
    full[:, ys[:, None], xs[None, :]] = A
    A_sp = sp.csr_matrix(full.reshape(A.shape[0], -1))

    c_dims = c_meta["dims"]
    if c_dims[0] != "unit_id":
        C = C.T
        if S is not None:
            S = S.T
    frame_idx = np.asarray(c_meta["coords"]["frame"]).astype(int)
    frame_idx = _mps_original_frames(result_dir, frame_idx)
    unit_a = a_meta["coords"]["unit_id"]
    unit_c = c_meta["coords"]["unit_id"]
    if list(unit_a) != list(unit_c):
        idx = {u: i for i, u in enumerate(unit_c)}
        sel = [idx[u] for u in unit_a]
        C = C[sel]
        if S is not None:
            S = S[sel]
    return Result(name, A_sp, (H, W), C.astype(np.float32),
                  None if S is None else S.astype(np.float32), frame_idx)


def _mps_original_frames(result_dir, frame_idx):
    """MPS step 2d renumbers the kept frames 0..n-1 after dropping frames
    (assign_coords(frame=np.arange(...))), so the exported frame labels are not
    the original frame numbers. It writes the original indices it removed to
    all_removed_frames.txt; rebuild the true numbering from that."""
    import ast
    hits = glob.glob(os.path.join(result_dir, "**", "all_removed_frames.txt"), recursive=True)
    if not hits:
        print("MPS: all_removed_frames.txt not found; using exported frame labels as-is")
        return frame_idx
    txt = open(max(hits, key=os.path.getmtime)).read()
    key = "Original frame indices removed:"
    removed = ast.literal_eval(txt.split(key, 1)[1].strip()) if key in txt else []
    removed = sorted(int(r) for r in removed)
    n_kept = len(frame_idx)
    contiguous = np.array_equal(frame_idx, np.arange(n_kept))
    if not removed:
        return frame_idx
    if not contiguous:
        print("MPS: frame labels are not 0..n-1; assuming they are already original indices")
        return frame_idx
    total = n_kept + len(removed)
    orig = np.setdiff1d(np.arange(total), removed)
    if len(orig) != n_kept:
        print("MPS: removed-frames list (%d) does not fit %d kept frames; labels left as-is"
              % (len(removed), n_kept))
        return frame_idx
    print("MPS: restored original frame numbers (removed frames: %s)" % removed[:20])
    return orig


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def _load_array(spec):
    """spec = 'file.ext' or 'file.ext:key'. Supports npy, npz, mat (v7), tif, h5/mat v7.3."""
    path, _, key = spec.partition("::") if "::" in spec else (spec, "", "")
    if not key and spec.count(":") >= 1 and not os.path.exists(spec):
        path, key = spec.rsplit(":", 1)  # allow file:key when unambiguous
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        z = np.load(path)
        return z[key or z.files[0]]
    if ext in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    if ext == ".mat":
        import scipy.io
        try:
            m = scipy.io.loadmat(path)
            return m[key]
        except NotImplementedError:
            pass  # v7.3 -> HDF5
    if ext in (".mat", ".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            arr = f[key][()]
        return arr.T if ext == ".mat" else arr  # MATLAB v7.3 stores transposed
    if ext == ".csv":
        return np.loadtxt(path, delimiter=",")
    raise ValueError("unsupported ground-truth file: %s" % path)


def _footprints_to_nhw(A, dims, layout="auto"):
    H, W = dims
    A = np.asarray(A)
    if layout == "auto":
        if A.ndim == 3 and A.shape[1:] == (H, W):
            layout = "n,h,w"
        elif A.ndim == 3 and A.shape[:2] == (H, W):
            layout = "h,w,n"
        elif A.ndim == 2 and A.shape[0] == H * W:
            layout = "pixels,n"
        elif A.ndim == 2 and A.shape[1] == H * W:
            layout = "n,pixels"
        else:
            raise ValueError("cannot infer footprint layout from shape %s with dims %s"
                             % (A.shape, dims))
    if layout == "n,h,w":
        return A
    if layout == "h,w,n":
        return np.moveaxis(A, -1, 0)
    if layout in ("pixels,n", "pixels,n,F"):
        order = "F" if layout.endswith("F") else "C"
        return np.stack([A[:, i].reshape((H, W), order=order) for i in range(A.shape[1])])
    if layout == "n,pixels":
        return A.reshape(A.shape[0], H, W)
    raise ValueError("unknown layout %s" % layout)


def load_groundtruth(spec, dims):
    """spec (dict, from config.yaml 'ground_truth' section):
        footprints: path[:key]            required
        traces:     path[:key]            calcium traces, (n, T) or (T, n)
        spikes:     path[:key]            optional
        footprint_layout: auto | n,h,w | h,w,n | pixels,n | pixels,n,F | n,pixels
    """
    A = _footprints_to_nhw(_load_array(spec["footprints"]), dims,
                           spec.get("footprint_layout", "auto")).astype(np.float32)
    n = A.shape[0]

    def _nT(arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape[0] != n and arr.shape[1] == n:
            arr = arr.T
        return arr

    C = _nT(_load_array(spec["traces"])) if spec.get("traces") else None
    S = _nT(_load_array(spec["spikes"])) if spec.get("spikes") else None

    # Ground truth may be longer than the video (e.g. a 10-min simulation with a
    # 5-min video). Keep only the frames the video shows: video frame 0 is
    # ground-truth frame `frame_offset`.
    off = int(spec.get("frame_offset", 0))
    nf = spec.get("n_frames")
    if nf is not None:
        sl = slice(off, off + int(nf))
        C = C[:, sl] if C is not None else None
        S = S[:, sl] if S is not None else None
        if C is not None and C.shape[1] < int(nf):
            raise ValueError("ground truth has only %d frames after offset %d, need %d"
                             % (C.shape[1], off, int(nf)))

    # Units that never fire inside the window are invisible in the video; no
    # pipeline can find them, so by default they are removed from the ground truth.
    if spec.get("drop_inactive", True) and (S is not None or C is not None):
        if S is not None:
            active = S.max(axis=1) > 0
        else:
            rng = C.max(axis=1) - C.min(axis=1)
            active = rng > 1e-3 * (np.abs(C).max() or 1)
        if not active.all():
            print("ground truth: dropping %d of %d units with no activity in the video window"
                  % ((~active).sum(), n))
        A, n = A[active], int(active.sum())
        C = C[active] if C is not None else None
        S = S[active] if S is not None else None

    T = C.shape[1] if C is not None else (S.shape[1] if S is not None else 0)
    return Result("ground_truth", sp.csr_matrix(A.reshape(n, -1)), tuple(dims),
                  C, S, np.arange(T))


def load_min1pipe(path, name="MIN1PIPE"):
    """min1pipe_standard.mat written by pipelines/run_min1pipe.m (saved as -v7)."""
    import scipy.io
    m = scipy.io.loadmat(path)
    A = np.asarray(m["A"], dtype=np.float32)
    H, W = int(np.squeeze(m["H"])), int(np.squeeze(m["W"]))
    if A.ndim == 2:  # a single unit gets squeezed by MATLAB
        A = A[None]
    C = np.atleast_2d(np.asarray(m["C"], dtype=np.float32))
    S = np.atleast_2d(np.asarray(m["S"], dtype=np.float32)) if "S" in m else None
    n = A.shape[0]
    return Result(name, sp.csr_matrix(A.reshape(n, -1)), (H, W), C, S,
                  np.arange(C.shape[1]))


def load_any(entry, dims):
    """entry from config 'pipelines' list: {name, format, output}."""
    fmt = entry.get("format", "standard")
    if fmt == "standard":
        return load_standard(entry["output"], entry["name"])
    if fmt == "mps":
        return load_mps(entry["output"], dims, entry["name"])
    if fmt == "min1pipe":
        return load_min1pipe(entry["output"], entry["name"])
    raise ValueError("unknown format %s" % fmt)
