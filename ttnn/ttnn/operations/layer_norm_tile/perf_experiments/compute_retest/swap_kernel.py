"""pytest plugin: PYTHONPATH=<this dir> LN_VARIANT=<v> run_safe_pytest.sh ... -p swap_kernel -> op compute = kernels/<v>."""

import os
from pathlib import Path

from ttnn.operations.layer_norm_tile import layer_norm_tile_program_descriptor as _pd

_pd.KERNEL_DIR = Path(__file__).parent / "kernels" / os.environ.get("LN_VARIANT", "base")
print(f"swap_kernel: KERNEL_DIR={_pd.KERNEL_DIR}")
