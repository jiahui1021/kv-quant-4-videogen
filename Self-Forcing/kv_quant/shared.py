"""Load the repository-level baseline quantizers for Self-Forcing."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any


_SHARED_PACKAGE = "_workspace_kv_quant"


def _shared_factory():
    if _SHARED_PACKAGE not in sys.modules:
        package_dir = Path(__file__).resolve().parents[2] / "kv_quant"
        spec = importlib.util.spec_from_file_location(
            _SHARED_PACKAGE,
            package_dir / "__init__.py",
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load shared quantizers from {package_dir}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[_SHARED_PACKAGE] = package
        spec.loader.exec_module(package)
    return importlib.import_module(f"{_SHARED_PACKAGE}.factory")


def shared_quantizer_class(method: str):
    class_names = {
        "RTN": ("rtn", "RTNQuantizer"),
        "KIVI": ("kivi", "KIVIQuantizer"),
        "QUAROT_KV": ("quarot_kv", "QuaRotKVQuantizer"),
    }
    try:
        module_name, class_name = class_names[method.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported shared quantizer method={method}") from exc
    _shared_factory()
    module = importlib.import_module(f"{_SHARED_PACKAGE}.{module_name}")
    return getattr(module, class_name)


def create_shared_quantizer(
    method: str,
    bits: int,
    block_size: int = 16,
    key_bits: int | None = None,
    value_bits: int | None = None,
    name: str | None = None,
    **kwargs: Any,
):
    """Create RTN/KIVI/QuaRot from the repository-level implementation."""
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported shared quantizer arguments: {unsupported}")
    factory = _shared_factory()
    return factory.create_quantizer(
        method,
        bits=bits,
        block_size=block_size,
        key_bits=key_bits,
        value_bits=value_bits,
        name=name,
    )
