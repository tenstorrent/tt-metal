"""lean_compute_startup bench: the campaign case bench (generated/meta/g6_typecast/bench/bench_cases.py)
with the op's kernel directory swapped for kernels_<LCS_VARIANT>/ of this experiment dir (in-process
monkeypatch of g6_typecast_program_descriptor.KERNEL_DIR; the op's files are untouched).
LCS_VARIANT unset or 'op' = the op's own kernels."""
import importlib.util
import os
from pathlib import Path

from ttnn.operations.g6_typecast import g6_typecast_program_descriptor as _pd

_HERE = Path(__file__).resolve().parent
_V = os.environ.get("LCS_VARIANT", "op")
if _V not in ("op", "stock"):
    _kd = _HERE / f"kernels_{_V}"
    assert _kd.is_dir(), _kd
    _pd.KERNEL_DIR = _kd
# LCS_NOC_SWAP=1: reader on NoC1 / writer on NoC0 (same RISCs) — probe of the read-latency NoC choice.
if os.environ.get("LCS_NOC_SWAP", "0") == "1":
    import ttnn as _ttnn

    _ttnn.ReaderConfigDescriptor = lambda: _ttnn.DataMovementConfigDescriptor(
        processor=_ttnn.DataMovementProcessor.RISCV_1, noc=_ttnn.NOC.NOC_1
    )
    _ttnn.WriterConfigDescriptor = lambda: _ttnn.DataMovementConfigDescriptor(
        processor=_ttnn.DataMovementProcessor.RISCV_0, noc=_ttnn.NOC.NOC_0
    )
print(f"LCS_NOC_SWAP={os.environ.get('LCS_NOC_SWAP', '0')}")
print(f"LCS_VARIANT={_V} KERNEL_DIR={_pd.KERNEL_DIR}")

_ROOT = _HERE.parents[5]
_spec = importlib.util.spec_from_file_location(
    "g6_bench_cases", _ROOT / "generated/meta/g6_typecast/bench/bench_cases.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

pytestmark = _mod.pytestmark
test_bench = _mod.test_bench
