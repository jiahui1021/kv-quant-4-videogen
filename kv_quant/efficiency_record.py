"""Compatibility import for the Self-Forcing efficiency contract.

The repository-level quantizers and the Self-Forcing adapter are both named
``kv_quant``.  Test runners put the repository root on ``sys.path`` before the
adapter directory, so this small loader keeps ``kv_quant.efficiency_record``
available without creating a second contract copy.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "Self-Forcing"
    / "kv_quant"
    / "efficiency_record.py"
)
_SPEC = spec_from_file_location("_self_forcing_efficiency_record", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load efficiency contract: {_SOURCE}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in _MODULE.__all__:
    globals()[_name] = getattr(_MODULE, _name)

__all__ = list(_MODULE.__all__)
