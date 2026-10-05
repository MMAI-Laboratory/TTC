"""Summarize results.csv files into the table format of the paper.

  python -m scripts.summarize outputs                   # every outputs/<run>/results.csv
  python -m scripts.summarize outputs/zeroshot          # one run
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import pandas as pd
from rich.console import Console
from rich.table import Table

CONSOLE = Console(width=max(Console().width, 160))
ZS = ["caltech101", "dtd", "eurosat", "fgvc", "flowers102", "food101", "imagenet", "oxfordiiitpet", "stanfordcars",
      "sun397", "ucf101"]
DG = ["ImageNetA", "ImageNetR", "ImageNetSketch", "ImageNetV2"]
SHORT = {"caltech101": "Caltech", "dtd": "DTD", "eurosat": "ESAT", "fgvc": "Aircraft", "flowers102": "Flower",
         "food101": "Food", "imagenet": "INet", "oxfordiiitpet": "Pets", "stanfordcars": "Cars", "sun397": "SUN",
         "ucf101": "UCF", "ImageNetA": "A", "ImageNetR": "R", "ImageNetSketch": "Ske.", "ImageNetV2": "V2"}


def harmonic_mean(a, b):
    return 2 * a * b / (a + b) if (a + b) else 0.0


def _table(title, datasets, rows, extra):
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("")
    for c in [SHORT[d] for d in datasets] + [name for name, _ in extra]:
        table.add_column(c, justify="center")
    for label, acc, *override in rows:
        override = override[0] if override else {}
        complete = all(d in acc for d in datasets)
        cells = [f"{acc[d]:.2f}" if d in acc else "-" for d in datasets]
        cells += [f"{override[n]:.2f}" if n in override else f"{fn(acc):.2f}" if complete else "-" for n, fn in extra]
        table.add_row(label, *cells)
    CONSOLE.print(table)


def print_run(path: Path | str) -> None:
    df = pd.read_csv(path)
    run = Path(path).parent.name
    df["aug"] = df["aug"].astype(str)
    params = [c for c in ("topk", "memory_size", "lam") if c in df.columns]
    keys = ["method", *params, "backbone", "task", "aug"]
    acc = df.groupby(keys + ["subtask", "shots", "dataset"])["accuracy"].mean()
    n_settings = df.groupby("method")[params].nunique().max(axis=1) if params else None
    avg = ("Avg.", lambda a: mean(a[d] for d in ZS))
    for key, g in acc.groupby(level=keys):
        info = dict(zip(keys, key))
        method, backbone, task, aug = info["method"], info["backbone"], info["task"], info["aug"]
        name = f"{run}: {method} {backbone}"
        if params and n_settings[method] > 1:
            name += " (" + ", ".join(f"{p}={info[p]}" for p in params) + ")"
        name += f" ({aug}aug)" if aug != "0" else ""
        by = g.droplevel(keys)
        if task == "zeroshot":
            _table(name, ZS, [("Acc.", by.loc[("zeroshot", 0)].to_dict())], [avg])
        elif task == "domain_generalization":
            _table(name, DG, [("Acc.", by.loc[("zeroshot", 0)].to_dict())], [("Dist.", lambda a: mean(a[d] for d in DG))])
        elif task == "fewshot":
            rows = [(f"{s} shots", by.loc[("fewshot", s)].to_dict()) for s in sorted(by.loc["fewshot"].index.unique(0))]
            _table(name, ZS, rows, [avg])
        elif task == "base_to_novel":
            base, novel = by.loc[("base", 16)].to_dict(), by.loc[("new", 0)].to_dict()
            hms = {d: harmonic_mean(base[d], novel[d]) for d in ZS if d in base and d in novel}
            avg_hm = {"Avg.": harmonic_mean(mean(base[d] for d in ZS), mean(novel[d] for d in ZS))} \
                if len(hms) == len(ZS) else {}
            _table(name, ZS, [("Base", base), ("Novel", novel), ("H.M.", hms, avg_hm)], [avg])
        elif task == "cross_dataset":
            _table(name, ZS, [("Acc.", by.loc[("cross", 0)].to_dict())], [avg])


def main():
    parser = argparse.ArgumentParser(description="Summarize TTC results")
    parser.add_argument("root", nargs="?", default="outputs", help="a run directory or a directory of runs")
    args = parser.parse_args()
    root = Path(args.root)
    paths = [root / "results.csv"] if (root / "results.csv").exists() else sorted(root.glob("*/results.csv"))
    if not paths:
        raise SystemExit(f"no results.csv found under {root}")
    for path in paths:
        print_run(path)


if __name__ == "__main__":
    main()
