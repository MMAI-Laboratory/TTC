from __future__ import annotations

from ttc.registry import register_method

from .ttc.ttc import TTC


@register_method("ttc")
def run_ttc(cfg, train_features, train_labels, test_features, test_labels, clip_weights):
    return TTC(cfg, train_features, train_labels, test_features, test_labels, clip_weights)[0]
