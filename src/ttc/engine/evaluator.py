from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch

from ttc import methods  # noqa: F401
from ttc.config import Config
from ttc.data.features import load_stream
from ttc.registry import get_method

FIELDS = ["method", "topk", "memory_size", "lam", "backbone", "task", "subtask", "dataset", "shots", "coop_seed",
          "seed", "aug", "accuracy"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def task_runs(cfg: Config) -> list[tuple[str, int]]:
    task = cfg.data.task
    if task in ("zeroshot", "domain_generalization"):
        return [("zeroshot", 0)]
    if task == "fewshot":
        return [("fewshot", int(s)) for s in cfg.data.shots]
    if task == "base_to_novel":
        return [("base", 16), ("new", 0)]
    if task == "cross_dataset":
        return [("cross", 0)]
    raise ValueError(f"unknown task '{task}'")


def run_one(cfg: Config, subtask: str, dataset: str, shots: int, coop_seed: int, seed: int) -> float:
    train_features, train_labels, test_features, test_labels, clip_weights = load_stream(
        cfg.data, subtask, dataset, shots, coop_seed)

    set_seed(seed)
    perm = torch.from_numpy(np.random.permutation(test_features.shape[0]))
    test_features, test_labels = test_features[perm], test_labels[perm]

    method_cfg = {"dataset": dataset, "backbone": cfg.data.backbone, "shots": shots, "lam": cfg.method.lam,
                  "attempts": cfg.method.topk, "memory_size": cfg.method.memory_size}
    return float(get_method(cfg.method.name)(method_cfg, train_features, train_labels, test_features, test_labels,
                                             clip_weights))


def _read_completed_runs(out: Path) -> set[tuple[str, ...]]:
    identity_fields = FIELDS[:-1]
    done = set()
    with out.open() as f:
        reader = csv.DictReader(f, strict=True)
        row_number = 1
        try:
            header = reader.fieldnames
            if header != FIELDS:
                raise SystemExit(
                    f"{out}: row 1 has columns {header}, expected {FIELDS}; use another --run.output_dir"
                )
            row_number = 2
            for row in reader:
                if None in row or None in row.values():
                    raise SystemExit(f"{out}: row {row_number}: expected {len(FIELDS)} columns")
                # Zero-shot runs intentionally serialize an empty CoOp seed.
                for field in identity_fields:
                    if field != "coop_seed" and not row[field].strip():
                        raise SystemExit(f"{out}: row {row_number}: missing {field!r}")
                try:
                    accuracy = float(row["accuracy"])
                except ValueError:
                    accuracy = math.nan
                if not math.isfinite(accuracy):
                    raise SystemExit(f"{out}: row {row_number}: accuracy must be a finite number")
                done.add(tuple(row[field] for field in identity_fields))
                row_number += 1
        except csv.Error as exc:
            raise SystemExit(f"{out}: row {row_number}: invalid CSV: {exc}") from exc
    return done


def evaluate(cfg: Config) -> Path:
    out = Path(cfg.run.output_dir) / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _read_completed_runs(out) if out.exists() and out.stat().st_size else set()

    coop_seeds = [None] if cfg.data.task in ("zeroshot", "domain_generalization") else cfg.data.coop_seeds
    for subtask, shots in task_runs(cfg):
        for dataset in cfg.data.datasets:
            for coop_seed in coop_seeds:
                for seed in cfg.run.seeds:
                    row = {"method": cfg.method.name, "topk": cfg.method.topk, "memory_size": cfg.method.memory_size,
                           "lam": cfg.method.lam, "backbone": cfg.data.backbone, "task": cfg.data.task,
                           "subtask": subtask, "dataset": dataset, "shots": shots,
                           "coop_seed": "" if coop_seed is None else coop_seed, "seed": seed, "aug": cfg.data.aug}
                    if tuple(str(row[k]) for k in FIELDS[:-1]) in done:
                        continue
                    row["accuracy"] = f"{run_one(cfg, subtask, dataset, shots, coop_seed, seed):.4f}"
                    print(f"[{cfg.method.name}] {subtask} {dataset} shots={shots} coop_seed={row['coop_seed']} "
                          f"seed={seed}: {row['accuracy']}", flush=True)
                    new_file = not out.exists() or not out.stat().st_size
                    with out.open("a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=FIELDS)
                        if new_file:
                            writer.writeheader()
                        writer.writerow(row)
    return out
