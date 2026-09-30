# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The C++ host side of ttnn.bringup.mhc_post against the Python builder it was ported from.

`ttnn.bringup.mhc_post` builds its program in C++ (device/mhc_post_ttnn_program_factory.cpp); the Python builder
(mhc_post_program_descriptor.py) stays as the reference. Two checks, as rms_norm_ttnn's parity test:

  1. PROGRAM PARITY on the host, no dispatch: both builders on the same tensors, the ProgramDescriptors compared field
     by field (kernel sources, cores, configs incl. NoC mode, compile-time and per-core runtime args, CBs,
     semaphores), over dtype pairs, aligned / ragged T, widths from 1 to 896 column tiles, n 1..5, ranks 2..4, the
     read help on and off, and GLM-5.3's shape.
  2. OUTPUT PARITY on the device: the C++ op and the Python op give bit-identical outputs, and so does a second C++
     call on fresh buffers (a program-cache hit, which only re-patches addresses).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ttnn

from ttnn.bringup.mhc_post_ttnn import mhc_post as mhc_post_python
from ttnn.bringup.mhc_post_ttnn.mhc_post import default_compute_kernel_config
from ttnn.bringup.mhc_post_ttnn.mhc_post_program_descriptor import create_program_descriptor as python_descriptor

_REPO = Path(__file__).resolve().parents[6]
cpp_descriptor = ttnn._ttnn.operations.bringup._mhc_post_ttnn_program_descriptor

F32, BF16 = ttnn.float32, ttnn.bfloat16

# (id, lead dims, T, C, n, X dtype, F dtype)
CASES = [
    ("glm_5120x4096_n4_bf16", (), 5120, 4096, 4, BF16, BF16),
    ("glm_2048x4096_n4_bf16", (), 2048, 4096, 4, BF16, BF16),
    ("t640_c1792_bf16", (), 640, 1792, 4, BF16, BF16),
    ("t1280_c4096_bf16_help", (), 1280, 4096, 4, BF16, BF16),
    ("t1000_c7168_x32_f16", (), 1000, 7168, 4, F32, BF16),
    ("t1000_c7168_f32", (), 1000, 7168, 4, F32, F32),
    ("t640_c7168_x16_f32", (), 640, 7168, 4, BF16, F32),
    ("t64_c256_n1", (), 64, 256, 1, BF16, BF16),
    ("t96_c512_n2", (), 96, 512, 2, BF16, BF16),
    ("t100_c1024_n3_ragged", (), 100, 1024, 3, BF16, BF16),
    ("t33_c32_n5_tiny", (), 33, 32, 5, F32, F32),
    ("t1_c64_n4_one_row", (), 1, 64, 4, BF16, BF16),
    ("rank3_2x300_c768", (2,), 300, 768, 4, BF16, BF16),
    ("rank4_2x3x64_c512", (2, 3), 64, 512, 4, F32, BF16),
    ("t4096_c28672_wide", (), 256, 28672, 4, BF16, BF16),
]
_IDS = [c[0] for c in CASES]


def _torch(dtype):
    return torch.float32 if dtype == F32 else torch.bfloat16


def _inputs(device, case, seed=0):
    _id, lead, T, C, n, x_dtype, f_dtype = case
    g = torch.Generator().manual_seed(seed)
    shape = lambda last: [*lead, T, last]  # noqa: E731
    host = {
        "f": torch.randn(shape(C), generator=g).to(_torch(f_dtype)),
        "x": torch.randn(shape(n * C), generator=g).to(_torch(x_dtype)),
        "post": torch.rand(shape(n), generator=g),
        "comb": torch.rand(shape(n * n), generator=g),
    }
    to = lambda t, dt: ttnn.from_torch(  # noqa: E731
        t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return (to(host["f"], f_dtype), to(host["x"], x_dtype), to(host["post"], F32), to(host["comb"], F32))


def _cores(crs):
    return sorted((c.x, c.y) for c in ttnn.corerange_to_cores(crs, None, True))


def _kernel_file(path: str) -> Path:
    p = Path(path)
    return (p if p.is_absolute() else _REPO / p).resolve()


def _config_of(cfg):
    if isinstance(cfg, ttnn.ComputeConfigDescriptor):
        return (
            "compute",
            str(cfg.math_fidelity),
            cfg.fp32_dest_acc_en,
            cfg.dst_full_sync_en,
            [str(m) for m in cfg.unpack_to_dest_mode],
            cfg.bfp8_pack_precise,
            cfg.math_approx_mode,
            cfg.enable_trisc2_rvv,
        )
    if isinstance(cfg, ttnn.DataMovementConfigDescriptor):
        return ("dm", str(cfg.processor), cfg.noc.value, str(cfg.noc_mode))
    return (type(cfg).__name__,)


def _kernel_of(k):
    view = k.runtime_args
    rt = {(x, y): list(view[x][y]) for x, y in _cores(k.core_ranges)}
    return {
        "source": _kernel_file(k.kernel_source),
        "source_type": str(k.source_type),
        "cores": _cores(k.core_ranges),
        "config": _config_of(k.config),
        "defines": sorted((str(a), str(b)) for a, b in k.defines),
        "ct": list(k.compile_time_args),
        "common_rt": list(k.common_runtime_args),
        "rt": rt,
    }


def _describe(desc):
    return {
        "kernels": [_kernel_of(k) for k in desc.kernels],
        "cbs": [
            (
                [(fd.buffer_index, fd.data_format_as_uint8, fd.page_size) for fd in cb.format_descriptors],
                cb.total_size,
                _cores(cb.core_ranges),
            )
            for cb in desc.cbs
        ],
        "semaphores": [(s.id, str(s.core_type), _cores(s.core_ranges), s.initial_value) for s in desc.semaphores],
    }


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_program_descriptor_parity(device, case):
    f, x, post, comb = _inputs(device, case)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)), x.dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    py = _describe(python_descriptor(f, x, post, comb, out, default_compute_kernel_config()))
    cpp = _describe(cpp_descriptor(f, x, post, comb, out))
    assert len(py["kernels"]) == len(cpp["kernels"]) == 3
    for name, a, b in zip(("reader", "writer", "compute"), py["kernels"], cpp["kernels"]):
        for field in a:
            assert a[field] == b[field], f"{name}.{field} differs"
    assert py["cbs"] == cpp["cbs"], f"CBs: python={py['cbs']} cpp={cpp['cbs']}"
    assert py["semaphores"] == cpp["semaphores"], "semaphores differ"


_DEVICE_CASES = [
    c
    for c in CASES
    if c[0]
    in ("glm_5120x4096_n4_bf16", "t1000_c7168_x32_f16", "t100_c1024_n3_ragged", "rank4_2x3x64_c512", "t33_c32_n5_tiny")
]


def _bits(t):
    t = t.contiguous()
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


@pytest.mark.parametrize("case", _DEVICE_CASES, ids=[c[0] for c in _DEVICE_CASES])
def test_device_output_is_bit_identical(device, case):
    entries, addresses, spacers = [], [], []
    for seed in (0, 1):  # the second C++ call is a program-cache HIT on fresh buffers
        if seed == 1:
            spacers = [
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, 32, 32 * 7]), BF16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
                )
            ]
        args = _inputs(device, case, seed)
        expected = ttnn.to_torch(mhc_post_python(*args))
        before = device.num_program_cache_entries()
        actual_t = ttnn.bringup.mhc_post(*args)
        entries.append(device.num_program_cache_entries() - before)
        addresses.append([t.buffer_address() for t in args])
        actual = ttnn.to_torch(actual_t)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(_bits(actual), _bits(expected)), f"seed {seed}: C++ and Python outputs differ"
    assert entries[1] == 0, "the second C++ call missed the program cache"
    assert addresses[0] != addresses[1], "the second round reused the first round's buffers"
    del spacers


def test_binding_is_the_cpp_op():
    op = ttnn.bringup.mhc_post
    assert op.is_cpp_operation
    assert op.python_fully_qualified_name == "ttnn.bringup.mhc_post"
