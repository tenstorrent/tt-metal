"""pytest plugin: PYTHONPATH=<this dir> LN_VARIANT=<v> run_safe_pytest.sh ... -p swap_kernel -> op kernels = op_kernels/<v>."""

import importlib
import os
from pathlib import Path

_pd = importlib.import_module("ttnn.operations.layer_norm_tile.layer_norm_tile_program_descriptor")
_pd.KERNEL_DIR = Path(__file__).parent / "op_kernels" / os.environ.get("LN_VARIANT", "base")
print(f"swap_kernel: KERNEL_DIR={_pd.KERNEL_DIR}")
