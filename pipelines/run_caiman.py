"""
run_caiman.py -- CaImAn CNMF-E (1-photon) on one video, headless, writing the
benchmark's standard result file. Run inside the CaImAn environment.

  python run_caiman.py --video D:\\bench\\data\\synth.avi --out D:\\bench\\runs\\CaImAn\\rep1 ^
                       --fps 10 --workers 8 --params caiman_params.json

Parameters follow CaImAn's demo_pipeline_cnmfE for 1p data. Every value can be
overridden in the JSON file (keys are CNMFParams names, e.g. "gSig": [4, 4]).
Tune min_corr / min_pnr / gSig on the synthetic data the same way you tune
the other tools, and record the final JSON in the paper's supplement.

Written against the CaImAn 1.9 - 1.11 API. If your version differs, check
caiman/demos/notebooks/demo_pipeline_cnmfE.ipynb in your install.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchio import save_standard, stage  # noqa: E402

DEFAULTS = {
    # data
    "fr": 10, "decay_time": 0.4,
    # motion correction (rigid; 1p uses a high-pass filtered template)
    "motion_correct": True, "pw_rigid": False, "max_shifts": [10, 10],
    "gSig_filt": [3, 3], "strides": [48, 48], "overlaps": [24, 24],
    "max_deviation_rigid": 3, "border_nan": "copy",
    # CNMF-E
    "method_init": "corr_pnr", "K": None, "gSig": [4, 4], "gSiz": [17, 17],
    "center_psf": True, "ring_size_factor": 1.4, "min_corr": 0.8, "min_pnr": 10,
    "nb": 0, "nb_patch": 0, "low_rank_background": None,
    "update_background_components": True, "only_init": True, "ssub_B": 2,
    "rf": 48, "stride": 24, "merge_thr": 0.7, "p": 1, "tsub": 1, "ssub": 1,
    "normalize_init": False, "del_duplicates": True, "method_deconvolution": "oasis",
    # component evaluation
    "min_SNR": 3, "rval_thr": 0.85, "use_cnn": False,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--params", help="JSON file of CNMFParams overrides")
    ap.add_argument("--no-eval", action="store_true", help="keep all components (skip evaluate_components)")
    a = ap.parse_args()

    p = dict(DEFAULTS)
    p["fr"] = a.fps
    if a.params:
        with open(a.params) as f:
            p.update(json.load(f))
    for k in ("max_shifts", "gSig_filt", "strides", "overlaps", "gSig", "gSiz"):
        if isinstance(p.get(k), list):
            p[k] = tuple(p[k])
    p["fnames"] = [os.path.abspath(a.video)]
    os.makedirs(a.out, exist_ok=True)

    with stage("import"):
        import numpy as np
        import caiman as cm
        from caiman.motion_correction import MotionCorrect
        from caiman.source_extraction import cnmf
        from caiman.source_extraction.cnmf import params as cparams

    with stage("cluster_start"):
        _, dview, n_proc = cm.cluster.setup_cluster(backend="multiprocessing",
                                                    n_processes=a.workers,
                                                    single_thread=False)
    try:
        opts = cparams.CNMFParams(params_dict={k: v for k, v in p.items()
                                               if k not in ("min_SNR", "rval_thr", "use_cnn")})
        opts.change_params({"min_SNR": p["min_SNR"], "rval_thr": p["rval_thr"],
                            "use_cnn": p["use_cnn"]})

        with stage("motion_correction"):
            if p["motion_correct"]:
                mc = MotionCorrect(p["fnames"], dview=dview, **opts.get_group("motion"))
                mc.motion_correct(save_movie=True)
                mmap_in = mc.fname_tot_els if p["pw_rigid"] else mc.fname_tot_rig
                bord_px = 0 if p["border_nan"] == "copy" else int(np.ceil(np.max(np.abs(mc.shifts_rig))))
            else:
                mmap_in = p["fnames"]
                bord_px = 0

        with stage("memmap"):
            fname_new = cm.save_memmap(mmap_in, base_name=os.path.join(a.out, "memmap_"),
                                       order="C", border_to_0=bord_px)
            Yr, dims, T = cm.load_memmap(fname_new)
            images = Yr.T.reshape((T,) + dims, order="F")
            if p["motion_correct"]:
                for fn in mmap_in:  # intermediate motion-corrected memmap(s)
                    try:
                        os.remove(fn)
                    except OSError:
                        pass

        with stage("cnmfe_fit"):
            cnm = cnmf.CNMF(n_processes=n_proc, dview=dview, Ain=None, params=opts)
            cnm.fit(images)   # results land in cnm.estimates; some CaImAn versions return None here

        with stage("evaluate_components"):
            if not a.no_eval:
                cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
                keep = cnm.estimates.idx_components
            else:
                keep = np.arange(cnm.estimates.A.shape[1])
    finally:
        cm.stop_server(dview=dview)

    with stage("export"):
        est = cnm.estimates
        A = est.A.tocsc()[:, keep]                      # (pixels, n), Fortran pixel order
        H, W = dims
        # Fortran (column-major) pixel index -> row-major index
        A = A.tocoo()
        y = A.row % H
        x = A.row // H
        import scipy.sparse as sp
        A_rm = sp.csr_matrix((A.data, (A.col, y * W + x)), shape=(A.shape[1], H * W))
        C = est.C[keep]
        S = est.S[keep] if est.S is not None else None
        save_standard(os.path.join(a.out, "result_standard.npz"), A_rm, C, S,
                      frame_idx=np.arange(C.shape[1]), dims=(H, W),
                      meta={"tool": "CaImAn", "version": getattr(cm, "__version__", "?"),
                            "params": {k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in p.items() if k != "fnames"},
                            "border_px": bord_px})
        with open(os.path.join(a.out, "params_used.json"), "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}, f, indent=2)
        # memmap files are large; remove so disk does not fill over repeats
        for fn in os.listdir(a.out):
            if fn.endswith(".mmap"):
                try:
                    os.remove(os.path.join(a.out, fn))
                except OSError:
                    pass


if __name__ == "__main__":
    main()
