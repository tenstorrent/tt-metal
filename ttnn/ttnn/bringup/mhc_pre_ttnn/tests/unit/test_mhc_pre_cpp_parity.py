# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The C++ host side of ttnn.bringup.mhc_pre against the Python builder it was ported from.

`ttnn.bringup.mhc_pre` builds its program in C++ (device/mhc_pre_ttnn_program_factory.cpp); the Python builder
(mhc_pre_program_descriptor.py) stays as the reference. Two checks, as rms_norm_ttnn's parity test:

  1. PROGRAM PARITY on the host, no dispatch: both builders on the same tensors, the ProgramDescriptors compared field
     by field (kernel sources, cores, configs, defines, compile-time and per-core runtime args, CBs incl. their alias
     formats, semaphores), over every dtype pair (the W hi/lo split, the fp32-X pieces), narrow one-row groups (the
     cost-model width choice, the W column all-gather, the NoC flip) and taller groups (short T), decode, ragged T,
     n 1..4, ranks 2..4, and GLM-5.3's shape.
  2. OUTPUT PARITY on the device: the C++ op and the Python op give bit-identical (y, post, comb), and so does a second
     C++ call on fresh buffers (a program-cache hit, which only re-patches addresses).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ttnn

from ttnn.bringup.mhc_pre_ttnn import mhc_pre as mhc_pre_python
from ttnn.bringup.mhc_pre_ttnn.mhc_pre import default_compute_kernel_config
from ttnn.bringup.mhc_pre_ttnn.mhc_pre_program_descriptor import create_program_descriptor as python_descriptor

_REPO = Path(__file__).resolve().parents[6]
cpp_descriptor = ttnn._ttnn.operations.bringup._mhc_pre_ttnn_program_descriptor

F32, BF16 = ttnn.float32, ttnn.bfloat16
SCALE = (0.8, 1.3, 0.6)

# (id, lead dims, T, C, n, X dtype, W dtype)
CASES = [
    ("glm_5120x4096_n4_bf16_bf16", (), 5120, 4096, 4, BF16, BF16),
    ("glm_2048x4096_n4_bf16_bf16", (), 2048, 4096, 4, BF16, BF16),
    ("t1280_c4096_bf16_f32w", (), 1280, 4096, 4, BF16, F32),
    ("t640_c7168_bf16_f32w", (), 640, 7168, 4, BF16, F32),
    ("t640_c1792_f32_f32w", (), 640, 1792, 4, F32, F32),
    ("t640_c7168_f32_bf16w", (), 640, 7168, 4, F32, BF16),
    ("t1024_c1792_bf16_bf16", (), 1024, 1792, 4, BF16, BF16),
    ("t4096_c1792_bf16_f32w", (), 4096, 1792, 4, BF16, F32),
    ("t64_c256_short_groups", (), 64, 256, 4, BF16, F32),
    ("t100_c512_ragged", (), 100, 512, 4, BF16, BF16),
    ("t1_c7168_decode", (), 1, 7168, 4, BF16, F32),
    ("t1_c1792_decode_bf16w", (), 1, 1792, 4, F32, BF16),
    ("n1_t256_c512", (), 256, 512, 1, BF16, BF16),
    ("n2_t256_c1024", (), 256, 1024, 2, F32, F32),
    ("n3_t300_c768", (), 300, 768, 3, BF16, F32),
    ("rank3_2x320_c1024", (2,), 320, 1024, 4, BF16, BF16),
    ("rank4_2x2x96_c512", (2, 2), 96, 512, 4, F32, BF16),
]
_IDS = [c[0] for c in CASES]


def _torch(dtype):
    return torch.float32 if dtype == F32 else torch.bfloat16


def _inputs(device, case, seed=0):
    _id, lead, T, C, n, x_dtype, w_dtype = case
    g = torch.Generator().manual_seed(seed)
    mix = n * (n + 2)
    host_x = torch.randn([*lead, T, n * C], generator=g)
    host_w = torch.randn(n * C, mix, generator=g) * (n * C) ** -0.5
    host_b = torch.randn(1, mix, generator=g) * 0.1
    to = lambda t, dt: ttnn.from_torch(  # noqa: E731
        t.to(_torch(dt)), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return to(host_x, x_dtype), to(host_w, w_dtype), to(host_b, F32), n


def _outputs(device, x, n):
    lead = list(x.shape)[:-1]
    C = x.shape[-1] // n

    def alloc(last, dtype):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(lead + [last]), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )

    return alloc(C, x.dtype), alloc(n, F32), alloc(n * n, F32)


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
    return {
        "source": _kernel_file(k.kernel_source),
        "source_type": str(k.source_type),
        "cores": _cores(k.core_ranges),
        "config": _config_of(k.config),
        "defines": sorted((str(a), str(b)) for a, b in k.defines),
        "ct": list(k.compile_time_args),
        "common_rt": list(k.common_runtime_args),
        "rt": {(x, y): list(view[x][y]) for x, y in _cores(k.core_ranges)},
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
    x, w, b, n = _inputs(device, case)
    y, post, comb = _outputs(device, x, n)
    kw = dict(n=n, scale=SCALE, sinkhorn_iters=20, eps=1e-6, norm_eps=1e-5)
    py_desc, _plan = python_descriptor(x, w, b, y, post, comb, cfg=default_compute_kernel_config(), **kw)
    py = _describe(py_desc)
    cpp = _describe(cpp_descriptor(x, w, b, y, post, comb, **kw))
    assert len(py["kernels"]) == len(cpp["kernels"]), "kernel count (reader / writer sets) differs"
    for i, (a, c) in enumerate(zip(py["kernels"], cpp["kernels"])):
        for field in a:
            assert a[field] == c[field], f"kernel {i}.{field} differs"
    assert py["cbs"] == cpp["cbs"], f"CBs: python={py['cbs']} cpp={cpp['cbs']}"
    assert py["semaphores"] == cpp["semaphores"], "semaphores differ"


_DEVICE_CASES = [
    c
    for c in CASES
    if c[0]
    in (
        "glm_5120x4096_n4_bf16_bf16",
        "t640_c7168_bf16_f32w",
        "t640_c1792_f32_f32w",
        "t64_c256_short_groups",
        "t100_c512_ragged",
        "rank4_2x2x96_c512",
    )
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
        x, w, b, _n = _inputs(device, case, seed)
        kw = dict(scale=SCALE, sinkhorn_iters=20, eps=1e-6, norm_eps=1e-5)
        expected = [ttnn.to_torch(t) for t in mhc_pre_python(x, w, b, **kw)]
        before = device.num_program_cache_entries()
        actual_t = ttnn.bringup.mhc_pre(x, w, b, **kw)
        entries.append(device.num_program_cache_entries() - before)
        addresses.append([t.buffer_address() for t in (x, w, b)])
        for name, a, e in zip(("y", "post", "comb"), [ttnn.to_torch(t) for t in actual_t], expected):
            assert a.shape == e.shape and a.dtype == e.dtype, name
            assert torch.equal(_bits(a), _bits(e)), f"seed {seed}: C++ and Python {name} differ"
    assert entries[1] == 0, "the second C++ call missed the program cache"
    assert addresses[0] != addresses[1], "the second round reused the first round's buffers"
    del spacers


def test_binding_is_the_cpp_op():
    op = ttnn.bringup.mhc_pre
    assert op.is_cpp_operation
    assert op.python_fully_qualified_name == "ttnn.bringup.mhc_pre"
