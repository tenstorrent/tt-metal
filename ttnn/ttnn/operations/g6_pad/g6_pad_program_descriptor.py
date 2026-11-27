# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_pad: plan (regime, work split, block knobs) and ProgramDescriptor.

Every shape-, padding- or value-dependent quantity is a runtime argument; compile-time
arguments are limited to CB ids, element/page size, FILL_IS_ZERO and the TensorAccessorArgs,
so one binary per regime and dtype serves every shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
FACE = 16

# Smallest per-core assignment; num_cores = ceil(work_units / MIN_PAGES_PER_CORE), capped by the grid.
MIN_PAGES_PER_CORE = 1
# L1 the model leaves to the op on every core (BB-05).
L1_BUDGET_BYTES = 1347776

# CB slots (op_design.md, Circular Buffers).
CB_STAGED_TILES = 0
CB_FILL_PAGE_READER = 1
CB_FILL_PAGE_WRITER = 2


def high_padding(rank, padding):
    """Per-dim high padding, the pairs aligned to the trailing dims."""
    return [0] * (rank - len(padding)) + [int(pair[1]) for pair in padding]


def fill_bits(value, dtype):
    """Raw bit pattern of c: torch's own conversion of a Python number when it fills a tensor."""
    if dtype in (ttnn.int32, ttnn.uint32):
        return int(value) & 0xFFFFFFFF
    import torch

    torch_dtype, int_view, mask = {
        ttnn.bfloat16: (torch.bfloat16, torch.int16, 0xFFFF),
        ttnn.float32: (torch.float32, torch.int32, 0xFFFFFFFF),
    }[dtype]
    return int(torch.full((1,), value, dtype=torch_dtype).view(int_view)[0]) & mask


def _ceil_div(a, b):
    return -(-a // b)


def _balanced(total, num_cores, k):
    """split_work_to_cores' balanced formula: (start, count) of core k."""
    base, extra = divmod(total, num_cores)
    return k * base + min(k, extra), base + (1 if k < extra else 0)


def _lead(dims):
    """(lead0, lead1): the dims before -2, rank 4 -> (d0, d1), rank 3 -> (1, d0), rank 2 -> (1, 1)."""
    lead = [1, 1] + list(dims[:-2])
    return lead[-2], lead[-1]


def band_runs(a, b, es):
    """Byte runs of tile rows [a, b) in face layout, adjacent runs merged (at most 3)."""
    row_bytes = FACE * es
    face_bytes = FACE * row_bytes
    runs = []
    for face in range(4):
        face_row0 = FACE * (face >> 1)
        lo, hi = max(a, face_row0), min(b, face_row0 + FACE)
        if lo >= hi:
            continue
        off = face * face_bytes + (lo - face_row0) * row_bytes
        length = (hi - lo) * row_bytes
        if runs and runs[-1][0] + runs[-1][1] == off:
            runs[-1][1] += length
        else:
            runs.append([off, length])
    assert len(runs) <= 3, runs
    return [tuple(run) for run in runs]


@dataclass
class G6PadPlan:
    regime: str
    num_cores: int
    core_ranges: ttnn.CoreRangeSet
    block_pages: int
    staging_depth: int
    roles: list
    out_shape: list
    c_bits: int
    es: int
    page_bytes: int
    # Geometry shared by both regimes (per slice, in tiles).
    lead_in: tuple
    lead_out: tuple
    ht_in: int
    wt_in: int
    ht_out: int
    wt_out: int
    # extension_band
    band: list = field(default_factory=list)
    # new_storage_tiles
    row_mask: int = TILE
    col_mask: int = TILE
    # Per-core runtime arguments, in core order.
    per_core_args: list = field(default_factory=list)
    cores: list = field(default_factory=list)
    cache_key: tuple = ()


_PLAN_CACHE = {}
_DESCRIPTOR_CACHE = {}


def _cache_key(input_tensor, highs, c_bits):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    return (
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        tuple(highs),
        c_bits,
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.memory_config().buffer_type,
        device.id(),
        (grid.x, grid.y),
    )


def build_plan(input_tensor, highs, c_bits) -> G6PadPlan:
    key = _cache_key(input_tensor, highs, c_bits)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_plan(input_tensor, highs, c_bits)
        plan.cache_key = key
        _PLAN_CACHE[key] = plan
    return plan


def _build_plan(input_tensor, highs, c_bits):
    shape = list(input_tensor.shape)
    rank = len(shape)
    out_shape = [d + h for d, h in zip(shape, highs)]
    es = input_tensor.element_size()
    page_bytes = input_tensor.buffer_page_size()

    lead_in, lead_out = _lead(shape), _lead(out_shape)
    ht_in, wt_in = _ceil_div(shape[-2], TILE), _ceil_div(shape[-1], TILE)
    ht_out, wt_out = _ceil_div(out_shape[-2], TILE), _ceil_div(out_shape[-1], TILE)
    slices_in = lead_in[0] * lead_in[1]
    slices_out = lead_out[0] * lead_out[1]

    grid = input_tensor.device().compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y

    is_extension = (
        input_tensor.layout == ttnn.TILE_LAYOUT
        and highs[rank - 2] > 0
        and all(h == 0 for i, h in enumerate(highs) if i != rank - 2)
        and ht_out == ht_in
    )

    common = dict(
        out_shape=out_shape,
        c_bits=c_bits,
        es=es,
        page_bytes=page_bytes,
        lead_in=lead_in,
        lead_out=lead_out,
        ht_in=ht_in,
        wt_in=wt_in,
        ht_out=ht_out,
        wt_out=wt_out,
    )

    if is_extension:
        units = slices_in * wt_in
        num_cores = min(grid_cores, _ceil_div(units, MIN_PAGES_PER_CORE))
        last_row0 = TILE * (ht_in - 1)
        band = band_runs(shape[-2] - last_row0, out_shape[-2] - last_row0, es)
        per_core, roles = [], []
        for k in range(num_cores):
            start, count = _balanced(units, num_cores, k)
            per_core.append([start, count])
            roles.append({"core": k, "role": "band_writer", "units": count})
        return G6PadPlan(
            regime="extension_band",
            num_cores=num_cores,
            core_ranges=ttnn.num_cores_to_corerangeset(num_cores, grid, True),
            block_pages=0,
            staging_depth=0,
            roles=roles,
            band=band,
            per_core_args=per_core,
            cores=[ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(num_cores)],
            **common,
        )

    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise NotImplementedError("g6_pad: the new_storage_rows regime (ROW_MAJOR) is not built")

    staged = slices_in * ht_in * wt_in
    total = slices_out * ht_out * wt_out
    fill = total - staged
    num_cores = min(grid_cores, _ceil_div(max(staged, fill, 1), MIN_PAGES_PER_CORE))

    # Block knobs: one block holds the core's whole staged assignment when it fits the budget,
    # otherwise the largest block that double-buffers inside it.
    num_fill_pages = 2
    staging_budget = L1_BUDGET_BYTES - num_fill_pages * page_bytes
    r_max = max(1, _ceil_div(staged, num_cores))
    staging_depth = 1 if r_max * page_bytes <= staging_budget else 2
    block_pages_cap = staging_budget // (staging_depth * page_bytes)
    block_pages = min(r_max, block_pages_cap)

    row_mask = shape[-2] - TILE * (ht_in - 1) if shape[-2] % TILE != 0 and out_shape[-2] > shape[-2] else TILE
    col_mask = shape[-1] - TILE * (wt_in - 1) if shape[-1] % TILE != 0 and out_shape[-1] > shape[-1] else TILE

    fill_coords = _FillEnumeration(lead_in, lead_out, ht_in, wt_in, ht_out, wt_out)
    per_core, roles = [], []
    for k in range(num_cores):
        s_start, s_count = _balanced(staged, num_cores, k)
        num_blocks = _ceil_div(s_count, block_pages)
        f_start, f_count = _balanced(fill, num_cores, num_cores - 1 - k)
        f_slice, f_ht, f_wt = fill_coords.coords(f_start) if f_count > 0 else (0, 0, 0)
        per_core.append([s_start, s_count, num_blocks, f_slice, f_ht, f_wt, f_count])
        roles.append({"core": k, "role": "reader+writer", "staged_pages": s_count, "fill_pages": f_count})

    return G6PadPlan(
        regime="new_storage_tiles",
        num_cores=num_cores,
        core_ranges=ttnn.num_cores_to_corerangeset(num_cores, grid, True),
        block_pages=block_pages,
        staging_depth=staging_depth,
        roles=roles,
        row_mask=row_mask,
        col_mask=col_mask,
        per_core_args=per_core,
        cores=[ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(num_cores)],
        **common,
    )


class _FillEnumeration:
    """Fill pages in output page order -> (output slice, tile row, tile col).

    In an image slice (o0 < L0 and o1 < L1) rows ht < Ht_in hold cols [Wt_in, Wt_out) and rows
    ht >= Ht_in hold every col; a slice outside the input holds every page.
    """

    def __init__(self, lead_in, lead_out, ht_in, wt_in, ht_out, wt_out):
        self.l0, self.l1 = lead_in
        self.m1 = lead_out[1]
        self.ht_in, self.wt_in, self.wt_out = ht_in, wt_in, wt_out
        self.tail_cols = wt_out - wt_in
        self.per_image = ht_in * self.tail_cols + (ht_out - ht_in) * wt_out
        self.per_fill = ht_out * wt_out
        self.per_image_group = self.l1 * self.per_image + (self.m1 - self.l1) * self.per_fill
        self.per_fill_group = self.m1 * self.per_fill

    def coords(self, f):
        image_groups_total = self.l0 * self.per_image_group
        if f < image_groups_total:
            o0, r = divmod(f, self.per_image_group)
            if r < self.l1 * self.per_image:
                o1, r = divmod(r, self.per_image)
                ht, wt = self._image_slice(r)
            else:
                o1, r = divmod(r - self.l1 * self.per_image, self.per_fill)
                o1 += self.l1
                ht, wt = divmod(r, self.wt_out)
        else:
            o0, r = divmod(f - image_groups_total, self.per_fill_group)
            o0 += self.l0
            o1, r = divmod(r, self.per_fill)
            ht, wt = divmod(r, self.wt_out)
        return o0 * self.m1 + o1, ht, wt

    def _image_slice(self, r):
        if r < self.ht_in * self.tail_cols:
            ht, c = divmod(r, self.tail_cols)
            return ht, self.wt_in + c
        ht, wt = divmod(r - self.ht_in * self.tail_cols, self.wt_out)
        return self.ht_in + ht, wt


def _cb(index, pages, plan, dtype):
    return ttnn.CBDescriptor(
        total_size=pages * plan.page_bytes,
        core_ranges=plan.core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=plan.page_bytes)],
    )


def _runtime_args(plan, per_core):
    rt = ttnn.RuntimeArgs()
    for core, args in zip(plan.cores, per_core):
        rt[core.x][core.y] = args
    return rt


def _band_descriptor(plan, input_tensor, output_tensor):
    fill_is_zero = int(plan.c_bits == 0)
    ct_args = [CB_FILL_PAGE_WRITER, plan.es, plan.page_bytes, fill_is_zero]
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    runs = list(plan.band) + [(0, 0)] * (3 - len(plan.band))
    common_args = [output_tensor.buffer_address(), plan.c_bits, plan.ht_in, plan.wt_in, len(plan.band)]
    for off, length in runs:
        common_args += [off, length]

    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_pad_band_writer.cpp"),
        core_ranges=plan.core_ranges,
        compile_time_args=ct_args,
        runtime_args=_runtime_args(plan, plan.per_core_args),
        common_runtime_args=common_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    descriptor = ttnn.ProgramDescriptor(
        kernels=[writer],
        semaphores=[],
        cbs=[_cb(CB_FILL_PAGE_WRITER, 1, plan, input_tensor.dtype)],
    )
    # (kernel index, tensor position in generic_op's list) of every buffer-address common arg.
    return descriptor, [(0, 1)]


def _new_storage_descriptor(plan, input_tensor, output_tensor):
    fill_is_zero = int(plan.c_bits == 0)
    lead_in, lead_out = plan.lead_in, plan.lead_out

    reader_ct = [CB_STAGED_TILES, CB_FILL_PAGE_READER, plan.es, plan.page_bytes, fill_is_zero]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_common = [
        input_tensor.buffer_address(),
        plan.c_bits,
        plan.ht_in,
        plan.wt_in,
        plan.row_mask,
        plan.col_mask,
        plan.block_pages,
    ]
    reader_per_core = [[a[0], a[1], a[2]] for a in plan.per_core_args]

    writer_ct = [CB_STAGED_TILES, CB_FILL_PAGE_WRITER, plan.es, plan.page_bytes, fill_is_zero]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_common = [
        output_tensor.buffer_address(),
        plan.c_bits,
        plan.ht_in,
        plan.wt_in,
        lead_in[0],
        lead_in[1],
        plan.ht_out,
        plan.wt_out,
        lead_out[1],
        plan.block_pages,
    ]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_pad_reader.cpp"),
        core_ranges=plan.core_ranges,
        compile_time_args=reader_ct,
        runtime_args=_runtime_args(plan, reader_per_core),
        common_runtime_args=reader_common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_pad_writer.cpp"),
        core_ranges=plan.core_ranges,
        compile_time_args=writer_ct,
        runtime_args=_runtime_args(plan, plan.per_core_args),
        common_runtime_args=writer_common,
        config=ttnn.WriterConfigDescriptor(),
    )
    dtype = input_tensor.dtype
    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader, writer],
        semaphores=[],
        cbs=[
            _cb(CB_STAGED_TILES, plan.staging_depth * plan.block_pages, plan, dtype),
            _cb(CB_FILL_PAGE_READER, 1, plan, dtype),
            _cb(CB_FILL_PAGE_WRITER, 1, plan, dtype),
        ],
    )
    return descriptor, [(0, 0), (1, 1)]


def get_program_descriptor(plan, input_tensor, output_tensor):
    """Cached per plan key (BB-07); a hit patches only the buffer-address common arguments."""
    tensors = (input_tensor, output_tensor)
    entry = _DESCRIPTOR_CACHE.get(plan.cache_key)
    if entry is None:
        build = _band_descriptor if plan.regime == "extension_band" else _new_storage_descriptor
        entry = build(plan, input_tensor, output_tensor)
        _DESCRIPTOR_CACHE[plan.cache_key] = entry
        return entry[0]
    descriptor, address_args = entry
    for kernel_index, tensor_index in address_args:
        descriptor.kernels[kernel_index].common_runtime_args[0] = tensors[tensor_index].buffer_address()
    return descriptor
