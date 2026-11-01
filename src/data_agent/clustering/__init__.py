"""M0 input and configuration shared with downstream experiments."""

from .config import ClusterConfig
from .dataset_adapter import load_input

# Core functions are imported directly from .core to keep M0 usable without
# optional M1 dependencies.
__all__ = ["ClusterConfig", "load_input"]
