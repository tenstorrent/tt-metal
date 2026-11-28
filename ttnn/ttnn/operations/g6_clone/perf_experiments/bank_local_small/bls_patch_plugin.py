"""pytest plugin: run a suite on the exact rule.patch code without touching the op's files. The patched
descriptor source (rule_patched/) is executed in the op's own descriptor module namespace, so tests that
monkeypatch its knobs still reach the patched functions, and the kernels come from rule_patched/kernels.
Use: PYTHONPATH=<this dir>:$PYTHONPATH scripts/run_safe_pytest.sh <tests> -p bls_patch_plugin
(bench: BLS_VARIANT=patch in bench_cases.py, which loads this module the same way)."""
import importlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install():
    desc = importlib.import_module("ttnn.operations.g6_clone.g6_clone_program_descriptor")
    op = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
    src = HERE / "rule_patched" / "g6_clone_program_descriptor.py"
    exec(compile(src.read_text(), str(src), "exec"), desc.__dict__)
    desc.KERNEL_DIR = HERE / "rule_patched" / "kernels"
    for name in ("compute_plan", "create_program_descriptor", "patch_addresses", "tile_geometry", "zone_build"):
        setattr(op, name, getattr(desc, name))
    return "patch"


print(f"BLS_INSTALLED {install()}", flush=True)
