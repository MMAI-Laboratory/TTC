from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

DG_DATASETS = ["ImageNetA", "ImageNetR", "ImageNetSketch", "ImageNetV2"]


def _load(directory: Path, name: str) -> torch.Tensor:
    return load_file(directory / f"{name}.safetensors")[name].float()


def load_stream(data, task: str, dataset: str, shots: int, coop_seed: int):
    root = Path(data.feature_root)
    backbone = data.backbone.replace("/", "")
    # stream_dir holds the test stream; run_dir the text classifier and few-shot cache of one CoOp run
    if task == "zeroshot" and dataset in DG_DATASETS:
        run_dir = root / "domain_gen" / backbone / dataset
        stream_dir = run_dir / f"{data.aug}aug"
    elif task == "zeroshot":
        run_dir = stream_dir = root / "zeroshot" / backbone / dataset
    elif task == "fewshot":
        stream_dir = root / "fewshot" / dataset
        run_dir = stream_dir / f"{shots}shots" / f"seed{coop_seed}"
    elif task in ("base", "new"):
        stream_dir = root / "base2new" / dataset / task
        run_dir = stream_dir / f"seed{coop_seed}"
    elif task == "cross":
        if dataset == "imagenet":
            # the ImageNet source column is the 16-shot few-shot run
            shots = 16
        stream_dir = root / "cross" / dataset
        run_dir = stream_dir / f"seed{coop_seed}"
    else:
        raise ValueError(f"unknown task '{task}'")

    clip_weights = _load(run_dir, "text_weights_cupl" if task == "zeroshot" else "text_weights")
    test_features, test_labels = _load(stream_dir, "test_f"), _load(stream_dir, "test_l")
    train_features = train_labels = None
    if shots:
        train_features, train_labels = _load(run_dir, f"keys_{shots}shots"), _load(run_dir, f"values_{shots}shots")
    return train_features, train_labels, test_features, test_labels, clip_weights
