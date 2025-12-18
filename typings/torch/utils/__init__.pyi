from typing import Any as _Any

# Expose `data` submodule as Any so imports like `import torch.utils.data` resolve.
data: _Any

def __getattr__(name: str) -> _Any: ...
