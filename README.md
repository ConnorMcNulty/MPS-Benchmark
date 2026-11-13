# MPS benchmark kit

This kit runs MPS, minian, CaImAn (CNMF-E), and MIN1PIPE on the same synthetic video. It records each run's wall time and peak memory summed over its whole process tree, then scores every output against the ground truth.

The scripts assume each tool is already installed in its own environment on the benchmark PC. The orchestrator itself needs one more Python environment with `numpy scipy pandas matplotlib psutil pyyaml opencv-python`. That can be any of the existing environments, or a small new one.

## Layout

```
bench/
  check_video.py    sanity check of the AVI (codec is lossless, 3000 frames, 600x600, greyscale)
  monitor.py        runs one command; samples RSS and USS over the process tree every 0.2 s
  run_benchmark.py  runs every pipeline N times in rotating order, writes runs.csv
  converters.py     loads MPS native output, the wrapper outputs, and the ground truth
  metrics.py        Hungarian matching on footprint IoU; precision, recall, F1, trace r, spike r
  evaluate.py       scores every run, writes summary.csv and figures
pipelines/
  benchio.py        stage timer + common output writer (numpy/scipy only), used by the wrappers
  run_minian.py     minian notebook steps as a script
  run_caiman.py     CaImAn CNMF-E as a script
  run_min1pipe.m    MATLAB wrapper around min1pipe_HPC
params/             per-tool parameter files referenced by config.yaml
config.example.yaml
```

## Steps on the Windows PC

1. **Check the video.** Run `python -m bench.check_video D:\bench\data\synthetic_5min.avi --gt-frames 3000 --dims 600 600`. If it reports a lossy codec (MJPG, XVID, H.264), ask Ari to re-export the video with FFV1 or uncompressed before going further.
2. **Make MIN1PIPE's TIFF copy.** Run `python -m bench.make_tiff <video.avi> <CM_benchmark>/min1pipe_input/synthetic_5min.tif`. MIN1PIPE's AVI reader only handles uncompressed UCLA Miniscope files and calls `aviinfo`, which current MATLAB no longer has, so it gets a TIFF with identical pixels instead. The script checks every frame against the AVI. Keep the TIFF out of the AVI's folder.
3. **Put the AVI alone in its folder.** MPS step 2a and minian both pick up video files by pattern, so a second AVI in the same folder would be loaded too.
4. **Write the config.** Copy `config.example.yaml` to `config.yaml`, then fill in the environment python paths, the MATLAB and MIN1PIPE paths, and the ground-truth file and keys.
5. **Write the parameter files** in `params/`:
   - `mps_params.json` holds the MPS defaults from the repo README and its Windows sample file, with one change: the step 3a crop is set to the full 600×600 frame (factor 1.0, offsets 0, square). The sample's offsets were for a real rig and would cut off part of the synthetic field. The Dask worker count and memory limit are set on the command line in `config.yaml`. `mps_params_SAMPLE_from_repo.json` is the unmodified original, kept for reference.
   - `minian_params.json` and `caiman_params.json` contain overrides of the defaults at the top of each wrapper script. An empty `{}` runs the defaults.
6. **Check MIN1PIPE's entry point.** `run_min1pipe.m` calls `min1pipe_HPC(Fsi, Fsi_new, spatialr, se, ismc, flag, path_name, file_name)`, the signature in the current JinghaoLu/MIN1PIPE master. If your copy is older, compare the first line of its `min1pipe_HPC.m`. MIN1PIPE also needs CVX in `utilities\cvx`, which it downloads the first time you run it interactively.
7. **Do a dry run, then one real run of each tool.** `python -m bench.run_benchmark config.yaml --dry-run` prints the commands without running anything. Then run `python -m bench.run_benchmark config.yaml --repeats 1` and check that every run exits with code 0 and wrote its output.
8. **Tune parameters, then run the full set.** Use `python -m bench.run_benchmark config.yaml`. Close other programs during this run and turn off Windows Update and antivirus scans if you're allowed.
9. **Score the runs** with `python -m bench.evaluate config.yaml`. The results go to `<work_dir>\evaluation\`.

## What gets measured

- **Wall time** covers process start to exit, which includes imports, cluster startup, and file writes. Per-stage times come from the `stage()` markers in the wrappers, and from MPS's own per-step lines in `mps_transcript.log`.
- **Peak memory** is the highest total over the whole process tree at any sample. USS (private memory) is the number to report. RSS counts shared pages once per process, so with Dask workers it overstates usage.
- **Accuracy:** each footprint is binarized at 20% of its maximum, and Hungarian assignment then matches detected units to ground-truth units by IoU. A pair counts as a match if IoU ≥ 0.3 and the centroids are within 10 px. For matched units, the script reports trace and spike Pearson r over the frames each tool kept, so frames MPS drops in step 2d are excluded from the comparison. Before matching, a constant offset left by motion-correction templates is estimated and removed, and that offset is written to the output.

## Fairness settings to keep identical

- All tools get the same worker count. The two Dask tools (MPS and minian) also get the same per-worker memory limit.
- Every tool starts from its documented 1P defaults, and each one gets a similar, fixed amount of tuning on the synthetic data. Commit the final parameter files and put them in the supplement.
- Pin and record the versions: the MPS commit hash, `minian.__version__`, `caiman.__version__`, the MIN1PIPE commit, and the MATLAB release.
- The scripts don't currently cap MATLAB's thread count. Either add `maxNumCompThreads(8)` to `run_min1pipe.m`, or say in the paper that it was left uncapped.

## Tested here

The harness, converters, metrics, and evaluation were tested end to end on a toy 600×600 synthetic video with fake pipelines that spawn worker processes and write each output format. The tests covered a cropped MPS output with dropped frames, a shifted standard output, and a MIN1PIPE .mat file. The tool wrappers themselves (`run_minian.py`, `run_caiman.py`, `run_min1pipe.m`) could not be run here because the tools aren't installed in this environment, so the first single-repeat run on the PC is their real test.
