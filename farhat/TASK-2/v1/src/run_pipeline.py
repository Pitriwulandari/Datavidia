"""Orkestrasi seluruh tahap pipeline Task-2.

Contoh:
  python -m src.run_pipeline --stage all
  python -m src.run_pipeline --stage cluster --encoders dinov2_s --runs 16
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from src import config as C
from src.utils import get_logger

log = get_logger("pipeline")

STAGES = {
    "prepare": lambda a: ["-m", "src.prepare"],
    "embed": lambda a: ["-m", "src.embed", "--encoder", a.encoder, "--size", str(a.size)]
                       + (["--limit", str(a.limit)] if a.limit else []),
    "cluster": lambda a: ["-m", "src.features", "--encoders", *a.encoders,
                          "--runs", str(a.runs), "--tag", a.tag, "--state-tag", a.tag],
    "montage": lambda a: ["-m", "src.cluster", "--clusters", a.tag],
    "train": lambda a: ["-m", "src.train", "--clusters", a.tag, "--tag", a.tag],
    "predict": lambda a: ["-m", "src.predict", "--clusters", a.tag, "--probe", a.tag,
                          "--tag", a.subtag],
}


def run_stage(name: str, args) -> float:
    cmd = [sys.executable, *STAGES[name](args)]
    log.info(">>> %s", " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=C.ROOT, check=True)
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    ap.add_argument("--encoders", nargs="+", default=["dinov2_s"])
    ap.add_argument("--encoder", default="dinov2_s")
    ap.add_argument("--size", type=int, default=196)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--runs", type=int, default=C.CONSENSUS_RUNS)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--subtag", default="final")
    args = ap.parse_args()

    order = (["prepare", "embed", "cluster", "montage", "train", "predict"]
             if args.stage == "all" else [args.stage])
    timings = {}
    for stage in order:
        timings[stage] = round(run_stage(stage, args), 1)
        log.info("selesai %s (%.1fs)", stage, timings[stage])
    log_file = C.CACHE / "pipeline_timings.json"
    log_file.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    log.info("waktu: %s -> %s", timings, log_file)


if __name__ == "__main__":
    main()
