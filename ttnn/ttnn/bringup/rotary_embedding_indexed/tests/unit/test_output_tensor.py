# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.rotary_embedding_indexed(..., output_tensor=) (deepseek_v41_flash_d_p P.16): the op writes the caller's
buffer instead of allocating its output (uniform program and the per-coordinate one). On the box's mesh (FABRIC_2D),
DeepSeek-V4.1's rope_stock shapes (test_uniform.CASES):
  * the result is byte-identical on every chip to the call without the option, and lives in the given buffer;
  * the buffer reused by a second call (a program-cache hit) holds the new result; its marker contents are overwritten;
  * refusals (miss and hit): a buffer of another spec, the input itself as the output.
The warm host time a call with and without the option (the allocating side freeing its output, as the model does) is
printed, not gated."""

import importlib.util
import time
from pathlib import Path

import pytest
import torch

import ttnn

_spec = importlib.util.spec_from_file_location(
    "bringup_rotary_unit_test_uniform", Path(__file__).resolve().parent / "test_uniform.py"
)
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)


def _marker(mesh, like):
    t = torch.full([mesh.get_num_devices(), *list(like.shape)[1:]], 1.0e4)
    return U._put(mesh, t, True)


@pytest.mark.parametrize("mesh_device", U._mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", U.DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "per_coord"])
@pytest.mark.parametrize("rows, heads, width, prefix_w, inverse", U.CASES)
def test_output_tensor(mesh_device, uniform, rows, heads, width, prefix_w, inverse):
    mesh = mesh_device
    _, d = U._inputs(mesh, rows, heads, width, prefix_w, inverse)
    want = U._call(ttnn.bringup.rotary_embedding_indexed, d, uniform=uniform)
    out = _marker(mesh, want)
    for rep in range(2):
        got = U._call(ttnn.bringup.rotary_embedding_indexed, d, uniform=uniform, output_tensor=out)
        assert got.buffer_address() == out.buffer_address()
        for i, (a, b) in enumerate(zip(U._chips(mesh, out), U._chips(mesh, want))):
            assert torch.equal(a, b), f"chip {i} rep {rep}: output_tensor differs from the allocating call"
        if rep == 0:  # other inputs into the same buffer (the program's cache hit)
            d["x"] = U._put(mesh, torch.randn(mesh.get_num_devices(), heads, rows, width) * 2, True)
            want = U._call(ttnn.bringup.rotary_embedding_indexed, d, uniform=uniform)


@pytest.mark.parametrize("mesh_device", U._mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", U.DEVICE_PARAMS, indirect=True)
def test_output_tensor_refusals(mesh_device, expect_error):
    mesh = mesh_device
    _, d = U._inputs(mesh, 160, 1, 512, 0, False)
    bad = U._put(mesh, torch.zeros(mesh.get_num_devices(), 1, 160, 544), True)
    for _ in range(2):  # a miss, then a hit
        with expect_error(RuntimeError, "is not the op's output spec"):
            U._call(ttnn.bringup.rotary_embedding_indexed, d, uniform=True, output_tensor=bad)
        ttnn.deallocate(U._call(ttnn.bringup.rotary_embedding_indexed, d, uniform=True))
    with expect_error(RuntimeError, "must not be the input"):
        U._call(ttnn.bringup.rotary_embedding_indexed, d, uniform=True, output_tensor=d["x"])


@pytest.mark.parametrize("mesh_device", U._mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", U.DEVICE_PARAMS, indirect=True)
def test_output_tensor_host_time(mesh_device):
    """Warm host us a call (DeepSeek-V4.1's q rope): allocate + free vs a given output, printed."""
    mesh = mesh_device
    _, d = U._inputs(mesh, 160, 64, 64, 448, False)
    out = U._call(ttnn.bringup.rotary_embedding_indexed, d, uniform=True)
    res = {}
    for name in ("alloc", "given", "alloc", "given"):
        n = 50
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(n):
            if name == "alloc":
                ttnn.deallocate(U._call(ttnn.bringup.rotary_embedding_indexed, d, uniform=True))
            else:
                U._call(ttnn.bringup.rotary_embedding_indexed, d, uniform=True, output_tensor=out)
        res[name] = (time.perf_counter() - t0) / n * 1e6
        ttnn.synchronize_device(mesh)
    print(
        f"\nrotary_embedding_indexed host us a call on {tuple(mesh.shape)}: "
        f"allocate + free {res['alloc']:.1f}, output_tensor {res['given']:.1f}"
    )
