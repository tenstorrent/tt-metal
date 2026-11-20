"""Import a conv2d_3x3 program descriptor without ttnn (host-only plan math; importing ttnn starts the UMD cluster)."""
import importlib.util
import sys
import types


class _D:
    def __init__(self, n="x"):
        self.n = n

    def __getattr__(self, k):
        return _D(self.n + "." + k)

    def __call__(self, *a, **k):
        return _D(self.n + "()")

    def __or__(self, o):
        return self

    def __hash__(self):
        return hash(self.n)

    def __eq__(self, o):
        return isinstance(o, _D) and o.n == self.n


def load(path, name="pd"):
    if "ttnn" not in sys.modules:
        m = types.ModuleType("ttnn")
        m.__getattr__ = lambda k: _D("ttnn." + k)
        sys.modules["ttnn"] = m
    spec = importlib.util.spec_from_file_location(name, path)
    pd = importlib.util.module_from_spec(spec)
    sys.modules[name] = pd
    spec.loader.exec_module(pd)
    return pd
