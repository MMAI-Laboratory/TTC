from __future__ import annotations

import argparse
from pathlib import Path

from scripts.summarize import print_run
from ttc.config import Config, parse_cli_overrides
from ttc.engine import evaluate


def main():
    parser = argparse.ArgumentParser(description="Run TTC over the datasets and seeds of a config")
    parser.add_argument("--config", required=True, help="short name or path to a YAML config")
    args, rest = parser.parse_known_args()

    cfg = Config.from_name(args.config).apply_overrides(parse_cli_overrides(rest))
    Path(cfg.run.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(Path(cfg.run.output_dir) / "config.yaml")

    print_run(evaluate(cfg))


if __name__ == "__main__":
    main()
