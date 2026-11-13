"""
monitor.py -- run one command as a subprocess and record wall time plus
memory summed over its whole process tree (parent + every child/grandchild).

Why the whole tree: MPS and minian run Dask worker processes, CaImAn uses a
multiprocessing pool, and MATLAB's launcher spawns MATLAB.exe. Reading only the
top process (or using a tool that reports the single largest child) badly
under-counts all of these.

Two memory measures are recorded on every sample:
  rss_sum  -- sum of resident set size over the tree. Shared pages (DLLs,
              shared-memory blocks) are counted once per process, so this is
              an upper bound.
  uss_sum  -- sum of unique set size (private memory). Closer to the true
              footprint, but slower to read; enabled with --uss.
Report the same measure for every pipeline. USS is the better number for the
paper if sampling at your interval stays cheap enough on the benchmark PC.

Usage (standalone):
  python -m bench.monitor --out runs/minian_rep1 -- C:\\envs\\minian\\python.exe pipelines\\run_minian.py --config cfg.yaml
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

import psutil


def _tree(proc):
    try:
        return [proc] + proc.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return [proc]


def _sample(root, use_uss):
    rss = uss = 0
    n = 0
    cpu_times = 0.0
    for p in _tree(root):
        try:
            if use_uss:
                mi = p.memory_full_info()
                uss += mi.uss
                rss += mi.rss
            else:
                rss += p.memory_info().rss
            ct = p.cpu_times()
            cpu_times += ct.user + ct.system
            n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return rss, (uss if use_uss else None), n, cpu_times


def run_monitored(cmd, out_dir, cwd=None, env=None, interval=0.2, use_uss=False,
                  timeout_s=None, label=None):
    """Run cmd, sample memory every `interval` seconds, write results to out_dir.

    Returns a dict with wall_s, peak_rss_gb, peak_uss_gb, exit_code, etc.
    Writes:
      out_dir/memory_timeline.csv   t_s, rss_gb, uss_gb, n_procs, cpu_s
      out_dir/stdout.log            combined stdout/stderr of the run
      out_dir/run_summary.json      the returned dict
    """
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "stdout.log")
    tl_path = os.path.join(out_dir, "memory_timeline.csv")

    full_env = dict(os.environ)
    if env:
        # "${PATH}" (or any ${VAR}) inside a value is replaced by the current value,
        # so a config can prepend folders: PATH: "C:/env;C:/env/Library/bin;${PATH}"
        for k, v in env.items():
            v = str(v)
            for name, cur in os.environ.items():
                v = v.replace("${%s}" % name, cur)
            full_env[k] = v
    # Tell wrapper scripts where to write stage markers (see bench/stages.py).
    full_env["BENCH_STAGE_FILE"] = os.path.join(out_dir, "stages.jsonl")
    if os.path.exists(full_env["BENCH_STAGE_FILE"]):
        os.remove(full_env["BENCH_STAGE_FILE"])

    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    t0 = time.perf_counter()
    wall_start = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, env=full_env, stdout=logf,
                            stderr=subprocess.STDOUT)
    root = psutil.Process(proc.pid)

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            t = time.perf_counter() - t0
            rss, uss, n, cpu = _sample(root, use_uss)
            samples.append((t, rss, uss, n, cpu))
            stop.wait(interval)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    timed_out = False
    try:
        exit_code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        for p in reversed(_tree(root)):
            try:
                p.kill()
            except psutil.Error:
                pass
        exit_code = proc.wait()
    wall_s = time.perf_counter() - t0
    stop.set()
    th.join()
    logf.close()

    with open(tl_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "rss_gb", "uss_gb", "n_procs", "cpu_s"])
        for t, rss, uss, n, cpu in samples:
            w.writerow([f"{t:.3f}", f"{rss / 1e9:.4f}",
                        "" if uss is None else f"{uss / 1e9:.4f}", n, f"{cpu:.2f}"])

    peak_rss = max((s[1] for s in samples), default=0)
    peak_uss = max((s[2] for s in samples if s[2] is not None), default=None)
    summary = {
        "label": label,
        "cmd": cmd,
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_start)),
        "wall_s": round(wall_s, 3),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "peak_rss_gb": round(peak_rss / 1e9, 4),
        "peak_uss_gb": None if peak_uss is None else round(peak_uss / 1e9, 4),
        "max_procs": max((s[3] for s in samples), default=0),
        "n_samples": len(samples),
        "interval_s": interval,
        "machine": machine_info(),
    }
    summary["stages"] = read_stages(full_env["BENCH_STAGE_FILE"])
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def read_stages(path):
    """Stage markers written by wrapper scripts -> list of {stage, start, end, dur_s}."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def machine_info():
    vm = psutil.virtual_memory()
    import platform
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(vm.total / 1e9, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--uss", action="store_true")
    ap.add_argument("--timeout-min", type=float, default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("no command given")
    s = run_monitored(cmd, a.out, interval=a.interval, use_uss=a.uss,
                      timeout_s=None if a.timeout_min is None else a.timeout_min * 60)
    print(json.dumps({k: v for k, v in s.items() if k != "machine"}, indent=2))


if __name__ == "__main__":
    main()
