# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""host_move_list bake-off.

test_bench[<variant>-<case>]: LAUNCHES dispatches on the same input (fresh output each launch, so the cached
descriptor is re-patched with new addresses every launch); first and last results checked bit-exact against
torch.reshape with the suite's exact gate. Variant "base" = the real g6_reshape op (unmodified).
test_correct[<variant>-<case>]: NaN-poisoned input tile padding, then a second input at another address
through the cached descriptor; both exact.

Run one (variant, case) per process (bench.sh):
  scripts/run_safe_pytest.sh --profile --no-py-zones <this>::test_bench -k "hml_bank-q553"
"""

import os
import sys
from pathlib import Path

import pytest

torch = __import__("torch")  # experiment bench: global torch import is fine here (not part of ttnn)
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import hml  # noqa: E402
from eval.golden_tests.g6_reshape import helpers as H  # noqa: E402
from ttnn.operations.g6_reshape import g6_reshape as base_op  # noqa: E402

CASES = {
    "q553": ((1, 1, 6, 256), (1, 1, 1536), ttnn.bfloat16),
    "q508": ((1, 8, 64), (8, 1, 64), ttnn.bfloat16),
    "q494": ((8, 1, 64), (1, 1, 8, 64), ttnn.bfloat16),
    "q490": ((1, 8, 6, 256), (1, 1, 8, 1536), ttnn.bfloat16),
    "f120": ((1, 256), (1, 1, 256, 1), ttnn.float32),
    "q349": ((1, 16, 6, 256), (1, 1, 16, 1536), ttnn.bfloat16),
}
VARIANTS = ["base", "base_nz", *hml.VARIANTS]
LAUNCHES = int(os.environ.get("HML_LAUNCHES", "12"))


def _op(variant):
    if variant == "base":
        return base_op
    if variant == "base_nz":  # the op's own host + kernel, zones compiled out (in-process path patch only)
        hml.BASE.KERNEL = Path(__file__).parent / "base_nz" / "g6_reshape_mover.cpp"
        return base_op
    return lambda x, s: hml.hml_reshape(x, s, variant)


def _input(case, device, poison=None):
    in_shape, out_shape, dtype = CASES[case]
    t = H.make_torch_input(
        dtype, in_shape, "special_values" if poison is None and dtype in (ttnn.bfloat16, ttnn.float32) else None
    )
    x = H.create_ttnn_input_tensor(t, device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    if poison is not None:
        x = ttnn.fill_implicit_tile_padding(x, poison)
    return t, x


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("variant", VARIANTS)
def test_bench(variant, case, device):
    torch.manual_seed(0)
    in_shape, out_shape, dtype = CASES[case]
    t, x = _input(case, device)
    exp = H.pytorch_g6_reshape(t, out_shape)
    op = _op(variant)
    first = op(x, list(out_shape))
    ttnn.synchronize_device(device)
    last = None
    for _ in range(LAUNCHES - 1):
        if last is not None:
            ttnn.deallocate(last)
        last = op(x, list(out_shape))
        ttnn.synchronize_device(device)
    if variant == "null":
        return
    H.check_exact(first, exp, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    if last is not None:
        H.check_exact(last, exp, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    key = next((k for k in hml.LAST_PLAN if k[1] == tuple(out_shape)), None)
    if key is not None:
        hp = hml.LAST_PLAN[key]
        print(
            f"\nHML_PLAN {variant} {case} max_rt_words={hp.max_rt_words()} "
            f"xfers(mover0,core0)={[(len(p.reads), len(p.zc), len(p.copies), len(p.imgw)) for (m, xy), p in list(hp.progs.items())[:2]]}"
        )
    print(f"HML_OK {variant} {case}")


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("variant", [v for v in VARIANTS if v != "null"])
def test_correct(variant, case, device):
    torch.manual_seed(1)
    in_shape, out_shape, dtype = CASES[case]
    op = _op(variant)
    t, x = _input(case, device, poison=float("nan"))
    H.check_exact(op(x, list(out_shape)), H.pytorch_g6_reshape(t, out_shape), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    # Shift the allocator, then a second input at a new address through the cached descriptor.
    pad = ttnn.from_torch(torch.zeros(3, 64, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    t2, x2 = _input(case, device, poison=float("inf"))
    assert x2.buffer_address() != x.buffer_address()
    H.check_exact(op(x2, list(out_shape)), H.pytorch_g6_reshape(t2, out_shape), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    ttnn.deallocate(pad)
