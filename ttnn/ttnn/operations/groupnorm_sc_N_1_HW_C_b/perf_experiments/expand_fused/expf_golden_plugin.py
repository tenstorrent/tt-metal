# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: run the op's golden suite on an expand_fused kernel variant (kernels + its descriptor_patch.py).

  PYTHONPATH=<this dir>:$PYTHONPATH EXPF_GOLDEN_VARIANT=v3s [EXPF_GOLDEN_CHUNK=i/n] \
      scripts/run_safe_pytest.sh --run-all eval/golden_tests/groupnorm_sc_N_1_HW_C_b/ -p expf_golden_plugin

EXPF_GOLDEN_CHUNK=i/n keeps only the i-th of n slices of the test_op cells the op SUPPORTS (bf16 input, bf16 RM /
no affine, tile aligned, shard fit none/even); EXPF_GOLDEN_CHUNK=extras keeps every non-test_op test (regression,
loose, distributions, eps, program cache). Deselected test_op cells are refusal checks the kernels never see.
"""
import importlib.util
import os
from pathlib import Path

import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd

HERE = Path(__file__).parent
VARIANT = os.environ.get("EXPF_GOLDEN_VARIANT", "v3s")
KDIR = HERE / VARIANT
pd.KERNEL_DIR = KDIR

_orig_cpd, _orig_fp = pd.create_program_descriptor, pd.footprint_bytes
_patch = None
if (KDIR / "descriptor_patch.py").exists():
    spec = importlib.util.spec_from_file_location(f"expf_golden_patch_{VARIANT}", KDIR / "descriptor_patch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _patch = mod.patch

    def _cpd(plan, input_tensor, output_tensor, **kw):
        return _patch(_orig_cpd(plan, input_tensor, output_tensor, **kw), plan, input_tensor, output_tensor, **kw)

    def _fp(**kw):
        # CB 25 ([mr, e], 2 Float32 pages) lives in alias group B: only a T = 1 group B (4 KB) grows (to 8 KB).
        tile_shard = kw.get("sharded", False) and not kw["is_rm"]
        group_b = pd.F32_TILE_BYTES * kw["T"] if tile_shard else pd.group_b_bytes(kw["T"])
        return _orig_fp(**kw) + max(0, 2 * pd.F32_TILE_BYTES - group_b)

    pd.create_program_descriptor, pd.footprint_bytes = _cpd, _fp
print(f"EXPF_GOLDEN_PLUGIN: kernel dir {pd.KERNEL_DIR}, descriptor patch {_patch is not None}")


def _supported(nodeid):
    return (
        "dtype=BFLOAT16-in_place" in nodeid
        and "alignment=tile_aligned" in nodeid
        and ("affine_dtype=BFLOAT16" in nodeid or "affine_dtype=none" in nodeid)
        and ("affine_layout=ROW_MAJOR" in nodeid or "affine_layout=none" in nodeid)
        and ("shard_fit=none" in nodeid or "shard_fit=even" in nodeid)
    )


def pytest_collection_modifyitems(config, items):
    chunk = os.environ.get("EXPF_GOLDEN_CHUNK")
    if not chunk:
        return
    extras = chunk == "extras"
    i, n = (0, 1) if extras else (int(x) for x in chunk.split("/"))
    keep, drop, k = [], [], 0
    for it in items:
        if it.originalname == "test_op" or it.name.startswith("test_op["):
            if not extras and _supported(it.nodeid):
                (keep if k % n == i else drop).append(it)
                k += 1
            else:
                drop.append(it)
        else:
            (keep if extras else drop).append(it)
    config.hook.pytest_deselected(items=drop)
    items[:] = keep
