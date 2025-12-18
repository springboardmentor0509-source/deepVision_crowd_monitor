# Minimal local stub for `torch` to satisfy Pylance diagnostics.
from typing import Any as _Any

# Common symbols used in this repo — treat as opaque Any types.
Tensor: _Any
Module: _Any

def __getattr__(name: str) -> _Any: ...
