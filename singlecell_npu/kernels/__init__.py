"""Optional NPU kernels. Importing this module does not require TileLang."""

from .hvg import hvg_to_dense

__all__ = ["hvg_to_dense"]
