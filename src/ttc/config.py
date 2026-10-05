from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_type_hints

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


ZS_DATASETS = ["caltech101", "dtd", "eurosat", "fgvc", "flowers102", "food101", "imagenet", "oxfordiiitpet",
               "stanfordcars", "sun397", "ucf101"]


@dataclass
class MethodArgs:
    name: str = "ttc"
    topk: int = 3  # number of candidate labels
    memory_size: int = 16  # per-class memory size
    lam: float = 0.03


@dataclass
class DataArgs:
    task: str = "zeroshot"  # zeroshot, domain_generalization, fewshot, base_to_novel, cross_dataset
    backbone: str = "ViT-B16"  # RN50, ViT-B16, ViT-B32, ViT-L14
    datasets: list = field(default_factory=lambda: list(ZS_DATASETS))
    feature_root: str = "./features"  # zeroshot/, domain_gen/, fewshot/, cross/, base2new/
    shots: list = field(default_factory=lambda: [16])  # fewshot only
    coop_seeds: list = field(default_factory=lambda: [1, 2, 3])  # fewshot, base_to_novel, cross_dataset
    aug: int = 0  # views of the domain_generalization features: 0 or 10


@dataclass
class RunArgs:
    seeds: list = field(default_factory=lambda: [3407, 3408, 3409])  # test-stream permutations
    output_dir: str = "outputs/run"


@dataclass
class Config:
    method: MethodArgs = field(default_factory=MethodArgs)
    data: DataArgs = field(default_factory=DataArgs)
    run: RunArgs = field(default_factory=RunArgs)

    @classmethod
    def from_name(cls, name: str, configs_dir: Path | str = CONFIGS_DIR) -> "Config":
        p = Path(name)
        if not p.suffix:
            p = Path(configs_dir) / f"{name}.yaml"
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        return cls.from_yaml(p)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return _from_dict(cls, raw)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path: Path | str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def apply_overrides(self, overrides: dict[str, str]) -> "Config":
        for dotted, value in overrides.items():
            _set_dotted(self, dotted, value)
        return self


def _from_dict(dc_type: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(dc_type)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown config key(s) {sorted(unknown)} in section '{dc_type.__name__}'; "
                       f"valid keys: {sorted(known)}")

    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(dc_type)
    for f in fields(dc_type):
        if f.name not in data:
            continue
        val = data[f.name]
        ftype = type_hints[f.name]
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(ftype, val)
        else:
            kwargs[f.name] = val
    return dc_type(**kwargs)


def _optional(hint: Any) -> bool:
    return type(None) in get_args(hint) if hint is not None else False


def _coerce(current: Any, raw: Any, hint: Any = None) -> Any:
    if isinstance(raw, str) and raw.lower() in ("none", "null") and _optional(hint):
        return None
    target = type(current) if current is not None else next(
        (t for t in get_args(hint) if t is not type(None)), None)
    if target is bool:
        return str(raw).lower() in ("1", "true", "yes", "y")
    if target is int:
        return int(float(raw))
    if target is float:
        return float(raw)
    if target is list:
        items = [x.strip() for x in str(raw).split(",") if x.strip()]
        elem = current[0] if current else None
        return [_coerce(elem, it) for it in items] if elem is not None else items
    return raw


def _set_dotted(cfg: Any, dotted: str, value: str) -> None:
    parts = dotted.split(".")
    obj = cfg
    for i, p in enumerate(parts[:-1]):
        if not hasattr(obj, p):
            raise AttributeError(f"unknown config override '--{dotted}': no section '{'.'.join(parts[:i + 1])}'")
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        avail = sorted(f.name for f in fields(type(obj))) if is_dataclass(obj) else []
        raise AttributeError(f"unknown config override '--{dotted}': no key '{leaf}' (available: {avail})")
    hint = get_type_hints(type(obj)).get(leaf)
    setattr(obj, leaf, _coerce(getattr(obj, leaf), value, hint))


def parse_cli_overrides(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            key = tok[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                out[k] = v
                i += 1
            else:
                if i + 1 >= len(argv):
                    raise SystemExit(f"error: --{key} expects a value (use --{key}=<value> or provide one)")
                out[key] = argv[i + 1]
                i += 2
        else:
            i += 1
    return out
