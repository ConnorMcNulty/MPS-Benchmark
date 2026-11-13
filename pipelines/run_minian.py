"""
run_minian.py -- minian pipeline on one video, headless, following the steps
of minian's pipeline.ipynb (preprocess -> motion correction -> seed init ->
CNMF: spatial, temporal, merge, spatial, temporal). Writes the benchmark's
standard result file. Run inside the minian environment.

  python run_minian.py --video D:\\bench\\data\\synth.avi --out D:\\bench\\runs\\minian\\rep1 ^
                       --fps 10 --workers 8 --memory 8GB --params minian_params.json

Defaults are minian's notebook defaults with two changes for this data:
  * no spatial crop (the synthetic field is the full frame)
  * noise_freq scaled for 10 fps. minian's 0.06 default assumes ~30 fps
    (frequency is in cycles/frame); 0.06 * 30 / 10 = 0.18 keeps the same
    cutoff in Hz. Override if Ari's MPS settings use something else, and keep
    MPS and minian on the same value where the parameters correspond.

Written against minian 1.2.x.
"""

import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchio import save_standard, stage  # noqa: E402

DEFAULTS = {
    "denoise": {"method": "median", "ksize": 7},
    "background_removal": {"method": "tophat", "wnd": 15},
    "estimate_motion": {"dim": "frame"},
    "seeds_init": {"wnd_size": 1000, "method": "rolling", "stp_size": 500,
                   "max_wnd": 15, "diff_thres": 3},
    "pnr_refine": {"noise_freq": 0.18, "thres": 1.5},
    "ks_refine": {"sig": 0.05},
    "seeds_merge": {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.18},
    "initialize": {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.18},
    "init_merge": {"thres_corr": 0.8},
    "get_noise": {"noise_range": [0.18, 0.5]},
    "first_spatial": {"dl_wnd": 10, "sparse_penal": 0.01, "size_thres": [25, None]},
    "first_temporal": {"noise_freq": 0.18, "sparse_penal": 1, "p": 1,
                       "add_lag": 20, "jac_thres": 0.2},
    "first_merge": {"thres_corr": 0.8},
    "second_spatial": {"dl_wnd": 10, "sparse_penal": 0.01, "size_thres": [25, None]},
    "second_temporal": {"noise_freq": 0.18, "sparse_penal": 1, "p": 1,
                        "add_lag": 20, "jac_thres": 0.4},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--memory", default="8GB", help="per-worker memory limit")
    ap.add_argument("--params", help="JSON overrides, same structure as DEFAULTS")
    ap.add_argument("--keep-intermediate", action="store_true")
    a = ap.parse_args()

    P = json.loads(json.dumps(DEFAULTS))
    if a.params:
        with open(a.params) as f:
            for k, v in json.load(f).items():
                P.setdefault(k, {}).update(v)
    P["get_noise"]["noise_range"] = tuple(P["get_noise"]["noise_range"])
    for k in ("first_spatial", "second_spatial"):
        P[k]["size_thres"] = tuple(P[k]["size_thres"])

    os.makedirs(a.out, exist_ok=True)
    intpath = os.path.join(a.out, "minian_intermediate")
    os.environ["MINIAN_INTERMEDIATE"] = intpath
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"

    with stage("import"):
        import numpy as np
        from dask.distributed import Client, LocalCluster
        from minian.cnmf import (compute_trace, get_noise_fft, unit_merge,
                                 update_background, update_spatial, update_temporal)
        from minian.initialization import (initA, initC, ks_refine, pnr_refine,
                                           seeds_init, seeds_merge)
        from minian.motion_correction import apply_transform, estimate_motion
        from minian.preprocessing import denoise, remove_background
        from minian.utilities import (TaskAnnotation, get_optimal_chk, load_videos,
                                      save_minian)

    with stage("cluster_start"):
        cluster = LocalCluster(n_workers=a.workers, memory_limit=a.memory,
                               resources={"MEM": 1}, threads_per_worker=a.threads_per_worker,
                               dashboard_address=":8788")
        cluster.scheduler.add_plugin(TaskAnnotation())
        client = Client(cluster)

    def sv(arr, name, **kw):
        return save_minian(arr.rename(name), intpath, overwrite=True, **kw)

    try:
        with stage("load"):
            vdir, vname = os.path.split(os.path.abspath(a.video))
            varr = load_videos(vdir, pattern="^" + re.escape(vname) + "$", dtype=np.uint8,
                               downsample=dict(frame=1, height=1, width=1),
                               downsample_strategy="subset")
            chk, _ = get_optimal_chk(varr, dtype=float)
            varr = sv(varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}), "varr")

        with stage("preprocess"):
            varr_ref = varr - varr.min("frame").compute()
            varr_ref = denoise(varr_ref, **P["denoise"])
            varr_ref = remove_background(varr_ref, **P["background_removal"])
            varr_ref = sv(varr_ref, "varr_ref")

        with stage("motion_correction"):
            motion = estimate_motion(varr_ref, **P["estimate_motion"])
            motion = sv(motion.chunk({"frame": chk["frame"]}), "motion")
            Y = apply_transform(varr_ref, motion, fill=0)
            Y_fm_chk = sv(Y.astype(float), "Y_fm_chk")
            Y_hw_chk = sv(Y_fm_chk, "Y_hw_chk",
                          chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})

        with stage("initialization"):
            max_proj = sv(Y_fm_chk.max("frame"), "max_proj").compute()
            seeds = seeds_init(Y_fm_chk, **P["seeds_init"])
            seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **P["pnr_refine"])
            seeds = ks_refine(Y_hw_chk, seeds, **P["ks_refine"])
            seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
            seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **P["seeds_merge"])
            A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **P["initialize"])
            # One chunk: initC turns A into scipy.sparse blocks and then rechunks
            # them to a single block anyway; with several blocks, some dask
            # versions fail to concatenate them ("setting an array element with
            # a sequence"). Same memory as minian's own rechunk(-1).
            A_init = sv(A_init, "A_init", chunks={"unit_id": -1, "height": -1, "width": -1})
            C_init = initC(Y_fm_chk, A_init)
            C_init = sv(C_init, "C_init", chunks={"unit_id": 1, "frame": -1})
            A, C = unit_merge(A_init, C_init, **P["init_merge"])
            A = sv(A, "A")
            C = sv(C, "C")
            C_chk = sv(C, "C_chk", chunks={"unit_id": -1, "frame": chk["frame"]})
            b, f = update_background(Y_fm_chk, A, C_chk)
            f = sv(f, "f")
            b = sv(b, "b")

        with stage("noise_estimation"):
            sn_spatial = get_noise_fft(Y_hw_chk, **P["get_noise"]).persist()

        def spatial(A, C, C_chk, params, tag):
            A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **params)
            C_new = sv(C.sel(unit_id=mask) * norm_fac, "C_new")
            C_chk_new = sv(C_chk.sel(unit_id=mask) * norm_fac, "C_chk_new")
            b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
            A = sv(A_new, "A", chunks={"unit_id": 1, "height": -1, "width": -1})
            b_ = sv(b_new, "b")
            f_ = sv(f_new.chunk({"frame": chk["frame"]}), "f")
            C = sv(C_new, "C")
            C_chk = sv(C_chk_new, "C_chk")
            return A, C, C_chk, b_, f_

        def temporal(A, C, C_chk, b_, f_, params):
            YrA = sv(compute_trace(Y_fm_chk, A, b_, C_chk, f_), "YrA",
                     chunks={"unit_id": 1, "frame": -1})
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **params)
            C = sv(C_new.chunk({"unit_id": 1, "frame": -1}), "C")
            C_chk = sv(C, "C_chk", chunks={"unit_id": -1, "frame": chk["frame"]})
            S = sv(S_new.chunk({"unit_id": 1, "frame": -1}), "S")
            b0 = sv(b0_new.chunk({"unit_id": 1, "frame": -1}), "b0")
            c0 = sv(c0_new.chunk({"unit_id": 1, "frame": -1}), "c0")
            A = A.sel(unit_id=C.coords["unit_id"].values)
            return A, C, C_chk, S, b0, c0

        with stage("cnmf_spatial_1"):
            A, C, C_chk, b, f = spatial(A, C, C_chk, P["first_spatial"], 1)
        with stage("cnmf_temporal_1"):
            A, C, C_chk, S, b0, c0 = temporal(A, C, C_chk, b, f, P["first_temporal"])
        with stage("merge"):
            A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **P["first_merge"])
            A = sv(A_mrg, "A_mrg")
            C = sv(sig_mrg, "C_mrg")
            C_chk = sv(C, "C_mrg_chk", chunks={"unit_id": -1, "frame": chk["frame"]})
        with stage("cnmf_spatial_2"):
            A, C, C_chk, b, f = spatial(A, C, C_chk, P["second_spatial"], 2)
        with stage("cnmf_temporal_2"):
            A, C, C_chk, S, b0, c0 = temporal(A, C, C_chk, b, f, P["second_temporal"])

        with stage("export"):
            A_np = A.transpose("unit_id", "height", "width").compute().values
            C_np = C.transpose("unit_id", "frame").compute().values
            S_np = S.sel(unit_id=C.coords["unit_id"].values).transpose("unit_id", "frame").compute().values
            H, W = int(varr.sizes["height"]), int(varr.sizes["width"])
            full = np.zeros((A_np.shape[0], H, W), dtype=np.float32)
            ys = A.coords["height"].values.astype(int)
            xs = A.coords["width"].values.astype(int)
            full[:, ys[:, None], xs[None, :]] = A_np
            save_standard(os.path.join(a.out, "result_standard.npz"), full, C_np, S_np,
                          frame_idx=C.coords["frame"].values.astype(int),
                          meta={"tool": "minian", "params": P, "workers": a.workers,
                                "memory_limit": a.memory})
            with open(os.path.join(a.out, "params_used.json"), "w") as fh:
                json.dump(P, fh, indent=2, default=list)
    finally:
        client.close()
        cluster.close()
        if not a.keep_intermediate:
            shutil.rmtree(intpath, ignore_errors=True)


if __name__ == "__main__":
    main()
