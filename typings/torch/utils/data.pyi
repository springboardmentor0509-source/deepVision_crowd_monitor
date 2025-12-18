from typing import Any as _Any

# Minimal symbols that are commonly imported from torch.utils.data
DataLoader: _Any
Dataset: _Any

def __getattr__(name: str) -> _Any: ...
