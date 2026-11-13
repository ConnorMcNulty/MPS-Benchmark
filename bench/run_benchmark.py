"""
run_benchmark.py -- run every pipeline in config.yaml N times under the
process-tree monitor, and collect timing + memory into runs.csv.

  python -m bench.run_benchmark config.yaml
  python -m bench.run_benchmark config.yaml --only MPS minian --repeats 1

Each run gets its own folder:  <work_dir>/<pipeline>/rep<k>/
  stdout.log, memory_timeline.csv, stages.jsonl, run_summary.json, plus the
  pipeline's own output (the wrappers are told where via {run_dir}).

Run order is rotated each repeat (A B C D, then B C D A, ...) so slow drift on
the machine (thermal throttling, background updates, disk cache) does not land
on one pipeline. `cooldown_s` pauses between runs so memory is released and
the CPU cools before the next one starts.

Optional per-pipeline keys: env (extra environment variables), mkdirs (folders to
create), copy (list of [src, dst] files copied before the timer starts).

Placeholders available in cmd / output / mkdirs / copy / env values:
  {run_dir} {video} {video_dir} {video_name} {fps} {height} {width} {repo}
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time

import yaml

from bench.monitor import run_monitored


def expand(obj, ctx):
    """Fill {placeholders} from ctx. Unknown names and ${VAR} are left untouched."""
    if isinstance(obj, str):
        return re.sub(r"(?<!\$)\{(\w+)\}",
                      lambda m: str(ctx[m.group(1)]) if m.group(1) in ctx else m.group(0), obj)
    if isinstance(obj, list):
        return [expand(o, ctx) for o in obj]
    if isinstance(obj, dict):
        return {k: expand(v, ctx) for k, v in obj.items()}
    return obj


def base_ctx(cfg, cfg_path):
    video = os.path.abspath(cfg["video"])
    return {
        "video": video,
        "video_dir": os.path.dirname(video),
        "video_name": os.path.basename(video),
        "fps": cfg["fps"],
        "height": cfg["dims"][0],
        "width": cfg["dims"][1],
        "repo": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config_dir": os.path.dirname(os.path.abspath(cfg_path)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", nargs="*", help="pipeline names to run")
    ap.add_argument("--repeats", type=int, help="override repeats")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    mon = cfg.get("monitor", {})
    repeats = a.repeats or cfg.get("repeats", 3)
    work_dir = os.path.abspath(cfg["work_dir"])
    os.makedirs(work_dir, exist_ok=True)
    pipes = [p for p in cfg["pipelines"] if p.get("enabled", True)]
    if a.only:
        pipes = [p for p in pipes if p["name"] in a.only]
    if not pipes:
        sys.exit("no pipelines selected")

    ctx0 = base_ctx(cfg, a.config)
    rows = []
    runs_csv = os.path.join(work_dir, "runs.csv")

    for rep in range(1, repeats + 1):
        k = (rep - 1) % len(pipes)
        order = pipes[k:] + pipes[:k]
        for p in order:
            run_dir = os.path.join(work_dir, p["name"], "rep%d" % rep)
            if os.path.isdir(run_dir) and not a.dry_run:
                shutil.rmtree(run_dir)
            ctx = dict(ctx0, run_dir=run_dir)
            cmd = expand(p["cmd"], ctx)
            env = expand(p.get("env", {}), ctx)
            if a.dry_run:
                print("[%s rep%d] %s" % (p["name"], rep, " ".join(cmd)))
                continue
            os.makedirs(run_dir, exist_ok=True)
            for d in expand(p.get("mkdirs", []), ctx):
                os.makedirs(d, exist_ok=True)
            # untimed input copies, e.g. MIN1PIPE writes next to its input file
            for src, dst in expand(p.get("copy", []), ctx):
                os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
                shutil.copyfile(src, dst)
            print("=== %s  rep %d/%d  ->  %s" % (p["name"], rep, repeats, run_dir), flush=True)
            s = run_monitored(cmd, run_dir, cwd=expand(p.get("cwd"), ctx) if p.get("cwd") else None,
                              env=env, interval=mon.get("interval_s", 0.2),
                              use_uss=mon.get("uss", False),
                              timeout_s=(mon["timeout_min"] * 60) if mon.get("timeout_min") else None,
                              label=p["name"])
            stage_tot = {}
            for st in s["stages"]:
                stage_tot[st["stage"]] = stage_tot.get(st["stage"], 0) + st["dur_s"]
            row = {
                "pipeline": p["name"], "rep": rep, "wall_s": s["wall_s"],
                "peak_rss_gb": s["peak_rss_gb"], "peak_uss_gb": s["peak_uss_gb"],
                "exit_code": s["exit_code"], "timed_out": s["timed_out"],
                "max_procs": s["max_procs"], "run_dir": run_dir,
                "output": expand(p.get("output", ""), ctx),
                "stages_json": json.dumps(stage_tot),
            }
            rows.append(row)
            print("    wall %.1fs  peak RSS %.2f GB  peak USS %s GB  exit %s"
                  % (s["wall_s"], s["peak_rss_gb"], s["peak_uss_gb"], s["exit_code"]), flush=True)
            with open(runs_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            time.sleep(mon.get("cooldown_s", 30))

    if not a.dry_run:
        with open(os.path.join(work_dir, "machine.json"), "w") as f:
            from bench.monitor import machine_info
            json.dump(machine_info(), f, indent=2)
        print("runs ->", runs_csv)


if __name__ == "__main__":
    main()
