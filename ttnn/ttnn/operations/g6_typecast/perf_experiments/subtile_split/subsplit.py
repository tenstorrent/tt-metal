# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""subtile_split: TILE-path descriptor that splits a tile across cores (each core owns NF contiguous faces).

Unit = (tile, first_face) with NF faces: NF = 2 -> halves {0,1} / {2,3} (16x32 tiny-tile CBs),
NF = 1 -> one face (16x16 tiny-tile CBs). Units whose faces are all padding (live-face rule of the op,
g6_typecast_tile_faces.hpp) are dropped. One unit per core; the core order is the op's own
_tile_core_order on the units' tiles (DRAM bank rows), row-major otherwise.

Modes: "op" (the op's descriptor untouched), "half", "quarter", "auto" (finest split whose unit count fits
the grid; "op" when even whole tiles do not, or when the split would not add cores).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import ttnn

_pd = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast_program_descriptor")
_op = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast")

KDIR = Path(__file__).parent / "kernels"
_ORIG_TILE = _pd._create_tile_program_descriptor
MODE = {"value": "op"}
LAST = {}


def _live_faces(t, Wt, Ht, w_tail, h_tail):
    col, row = t % Wt, (t // Wt) % Ht
    dead_r = 1 <= w_tail <= 16 and col == Wt - 1
    dead_b = 1 <= h_tail <= 16 and row == Ht - 1
    live = []
    for f in range(4):
        if (f & 1) and dead_r:
            continue
        if (f & 2) and dead_b:
            continue
        live.append(f)
    return live


def units_for(nf, tensor_tiles, Wt, Ht, w_tail, h_tail):
    units = []
    for t in range(tensor_tiles):
        live = _live_faces(t, Wt, Ht, w_tail, h_tail)
        for ff in range(0, 4, nf):
            if any(f in live for f in range(ff, ff + nf)):
                units.append((t, ff))
    return units


def _split_descriptor(input_tensor, output_tensor, nf):
    device = input_tensor.device()
    logical = tuple(input_tensor.shape)
    out_dtype = output_tensor.dtype
    in_page = input_tensor.buffer_page_size()
    out_page = output_tensor.buffer_page_size()
    tensor_tiles = _pd._tensor_tiles(logical)
    w = logical[-1]
    h = logical[-2] if len(logical) >= 2 else 1
    w_tail, h_tail = w % 32, h % 32
    Wt, Ht = _pd._div_up(w, 32), _pd._div_up(h, 32)
    in_hdr, in_face = _pd._tile_geometry(input_tensor.dtype, in_page)
    out_hdr, out_face = _pd._tile_geometry(out_dtype, out_page)
    assert in_hdr == 0
    grid = device.compute_with_storage_grid_size()
    units = units_for(nf, tensor_tiles, Wt, Ht, w_tail, h_tail)
    assert len(units) <= grid.x * grid.y
    ordered = _pd._tile_core_order(device, grid, [t for t, _ in units], input_tensor.memory_config().buffer_type)
    if ordered is None:
        ordered = [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(len(units))]
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in ordered])

    tile_desc = ttnn.TileDescriptor(*HALF_TILE) if nf == 2 else ttnn.TileDescriptor(16, 16)
    in_unit = nf * in_face
    exp_sec = _pd._div_up(16 * nf, 16) * 16 if out_hdr else 0
    out_unit = exp_sec + nf * out_face
    cb_in = ttnn.CBDescriptor(
        total_size=in_unit,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_pd.CB_INPUT_TILES, data_format=input_tensor.dtype, page_size=in_unit, tile=tile_desc
            )
        ],
    )
    cb_out = ttnn.CBDescriptor(
        total_size=out_unit,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_pd.CB_OUTPUT_TILES, data_format=out_dtype, page_size=out_unit, tile=tile_desc
            )
        ],
    )

    mask = out_dtype == ttnn.bfloat8_b and w % 16 != 0
    col0 = w_tail % 16 if mask else 0
    face_a = 0 if (w_tail != 0 and w_tail < 16) else 1
    rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, (t, ff) in zip(ordered, units):
        bits = 0
        if mask and t % Wt == Wt - 1:
            for g in (face_a, face_a + 2):
                if ff <= g < ff + nf:
                    bits |= 1 << (g - ff)
        rrt[core.x][core.y] = [t, ff * in_face, bits]
        wrt[core.x][core.y] = [t, ff]
        crt[core.x][core.y] = [1]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "split_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=[_pd.CB_INPUT_TILES, in_page, in_unit, in_face, col0]
        + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()),
        runtime_args=rrt,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "split_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=[_pd.CB_OUTPUT_TILES, out_page, out_hdr, out_face, nf, exp_sec]
        + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()),
        runtime_args=wrt,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "split_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=[_pd.CB_INPUT_TILES, _pd.CB_OUTPUT_TILES, 1, nf],
        runtime_args=crt,
        config=_pd._compute_config(input_tensor.dtype, out_dtype),
    )
    desc = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=[cb_in, cb_out])
    LAST["placed"] = [((c.x, c.y), u) for c, u in zip(ordered, units)]
    return desc, {"regime": "tile_split", "nf": nf, "num_cores": len(units)}


# Arms whose compute runs an SFPU pass that TypecastNF / Int32ToBf16RneNF restrict to the unit's faces. Measured:
# these win from the split (math is the long pole); the packer-only arms (bf16 -> fp32 / bf8b) do not
# (flat to +6%: Q2221 half +3.7%, Q2194 quarter +5.8%, Q2227/Q2195 half +5..7%), so "auto" leaves them whole.
SFPU_SPLIT_ARMS = {
    (ttnn.int32, ttnn.bfloat16),
    (ttnn.bfloat16, ttnn.int32),
    (ttnn.uint16, ttnn.int32),
    (ttnn.float32, ttnn.bfloat16),
}


def choose_nf(input_tensor, mode, out_dtype=None):
    """auto = RECOMMENDED host rule: SFPU arm and the unit count of the finest split (quarter, then half) fits the
    grid -> that split (one unit per core); otherwise the op's whole-tile path."""
    if mode == "half":
        return 2
    if mode == "quarter":
        return 1
    if mode != "auto" or (input_tensor.dtype, out_dtype) not in SFPU_SPLIT_ARMS:
        return 4
    logical = tuple(input_tensor.shape)
    grid = input_tensor.device().compute_with_storage_grid_size()
    n = grid.x * grid.y
    w = logical[-1]
    h = logical[-2] if len(logical) >= 2 else 1
    tiles = _pd._tensor_tiles(logical)
    Wt, Ht = _pd._div_up(w, 32), _pd._div_up(h, 32)
    for nf in AUTO_ORDER:
        if len(units_for(nf, tiles, Wt, Ht, w % 32, h % 32)) <= n:
            return nf
    return 4


AUTO_ORDER = (1, 2)
HALF_TILE = (32, 16)  # (height, width) of the 2-face tiny tile. 16x32 packs bf8b wrong (row 15 of each face)


def _tile_descriptor(input_tensor, output_tensor):
    nf = choose_nf(input_tensor, MODE["value"], output_tensor.dtype)
    LAST["nf"] = nf
    if nf == 4:
        return _ORIG_TILE(input_tensor, output_tensor)
    return _split_descriptor(input_tensor, output_tensor, nf)


def set_mode(mode):
    MODE["value"] = mode
    _op._PROGRAM_CACHE.clear()


def install():
    _pd._create_tile_program_descriptor = _tile_descriptor
    _op._PROGRAM_CACHE.clear()
