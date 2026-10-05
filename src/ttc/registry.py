from __future__ import annotations

from typing import Callable

METHOD_REGISTRY: dict[str, Callable] = {}


def register_method(name: str) -> Callable[[Callable], Callable]:
    def deco(fn):
        if name in METHOD_REGISTRY:
            raise KeyError(f"method '{name}' already registered")
        METHOD_REGISTRY[name] = fn
        return fn
    return deco


def get_method(name: str) -> Callable:
    if name not in METHOD_REGISTRY:
        raise KeyError(f"unknown method '{name}'. registered: {sorted(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name]
