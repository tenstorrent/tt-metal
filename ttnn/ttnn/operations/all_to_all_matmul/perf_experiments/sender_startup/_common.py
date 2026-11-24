# sender_startup bake-off: point the op at this dir's kernel copy (never edits the real op). Variants are selected by
# A2AMM_ABLATE tokens (ss*) which both compile ABLATE_SS* defines into the copy and change the program hash.
from pathlib import Path

import ttnn.operations.all_to_all_matmul.all_to_all_matmul_program_descriptor as _pd

import os

KERNELS = Path(__file__).parent / os.environ.get("SS_KERNELS", "kernels")
_pd.KERNEL_DIR = KERNELS
