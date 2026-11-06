"""Single-cell analysis with explicit Ascend/CPU execution boundaries."""

__version__ = "0.1.0"


def __getattr__(name):
    # CPU-only imports must never require a functioning Ascend installation.
    if name == "pp":
        import importlib
        return importlib.import_module(".pp", __name__)
    if name == "DeviceSession":
        from .session import DeviceSession
        return DeviceSession
    raise AttributeError(name)
