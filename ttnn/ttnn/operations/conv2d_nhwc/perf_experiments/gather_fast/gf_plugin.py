"""pytest plugin (-p gf_plugin, PYTHONPATH=<this dir>): alias ttnn.operations.conv2d_nhwc to variant/ so
the op's own unit tests run against the gather_fast variant (knobs GF_MODE / GF_PAD from env)."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_v = os.path.join(HERE, "variant")
_spec = importlib.util.spec_from_file_location(
    "gf_variant", os.path.join(_v, "__init__.py"), submodule_search_locations=[_v]
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["gf_variant"] = _pkg
_spec.loader.exec_module(_pkg)
import ttnn.operations  # noqa: E402

sys.modules["ttnn.operations.conv2d_nhwc"] = _pkg
for _sub in ("conv2d_nhwc", "conv2d_nhwc_program_descriptor"):
    sys.modules[f"ttnn.operations.conv2d_nhwc.{_sub}"] = sys.modules[f"gf_variant.{_sub}"]
setattr(ttnn.operations, "conv2d_nhwc", _pkg)
print(
    f"GF_PLUGIN: conv2d_nhwc -> variant (GF_MODE={os.environ.get('GF_MODE','0')} GF_PAD={os.environ.get('GF_PAD','0')})"
)
