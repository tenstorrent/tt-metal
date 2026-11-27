# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshape plan builder and ProgramDescriptor.

The op is a bijection between the valid face-row segments of the input and the output tile
tensors. One tensor (the tile side) is split into strips over all lanes (two data-movement
RISC-Vs per core); the other (the map side) is addressed segment by segment through the index
map. Every knob below is read as a module global at call time and is part of the memo key.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
LANE_KERNEL = KERNEL_DIR / "g6_reshape_lane.cpp"

# ---------------------------------------------------------------------------
# Blocking knobs (single source of truth)
# ---------------------------------------------------------------------------
LANES_PER_CORE = 2
MIN_STRIPS_PER_LANE = 1
BLOCK_SLOT_BYTES_CAP = 131072
SLOT_DEPTH = 1
STAGING_SLOTS_CAP = 512
COPY_ENTRIES_CAP = 1024
REGIME_OVERRIDE = None

GATHER_ALIGNED = "gather_aligned"
SCATTER_ALIGNED = "scatter_aligned"
GATHER_GENERAL = "gather_general"
REGIME_DEFINES = {
    GATHER_ALIGNED: "G6_REGIME_GATHER_ALIGNED",
    SCATTER_ALIGNED: "G6_REGIME_SCATTER_ALIGNED",
    GATHER_GENERAL: "G6_REGIME_GATHER_GENERAL",
}

# Lane-private scratch; no CB credit ever passes through them.
CB_LANE_TILE_IMAGES = (0, 1)
CB_LANE_STAGING = (2, 3)

TILE = 32
FACE = 16
COPY_ENTRY_BYTES = 8
# The copy table stores the source as a 16-bit offset into the staging ring.
STAGING_OFFSET_LIMIT = 1 << 16
MAX_RANK = 4


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def geometry(shape):
    """W, H, IMG, Wt, Ht and the tile count of a TILE tensor with logical shape `shape`."""
    shape = tuple(int(s) for s in shape)
    W = shape[-1]
    H = shape[-2] if len(shape) >= 2 else 1
    IMG = math.prod(shape[:-2]) if len(shape) > 2 else 1
    Wt = -(-W // TILE)
    Ht = -(-H // TILE)
    return {"W": W, "H": H, "IMG": IMG, "Wt": Wt, "Ht": Ht, "num_tiles": IMG * Ht * Wt}


def valid_rows(geo, tr):
    return min(TILE, geo["H"] - TILE * tr)


def valid_cols(geo, tc):
    return min(TILE, geo["W"] - TILE * tc)


def tile_runs(r_lo, r_hi, vc, face_row_bytes):
    """Merged byte runs (offset, bytes) that move rows [r_lo, r_hi) of one tile; same rule as the kernel."""
    face_bytes = FACE * face_row_bytes
    runs = []
    for f in range(4):
        if (f & 1) and vc <= FACE:
            continue
        f_r0 = (f >> 1) * FACE
        lo, hi = max(r_lo, f_r0), min(r_hi, f_r0 + FACE)
        if lo >= hi:
            continue
        off, nbytes = f * face_bytes + (lo - f_r0) * face_row_bytes, (hi - lo) * face_row_bytes
        if runs and runs[-1][0] + runs[-1][1] == off:
            runs[-1] = (runs[-1][0], runs[-1][1] + nbytes)
        else:
            runs.append((off, nbytes))
    return runs


def tile_side_transfers(geo, face_row_bytes):
    """Transfers needed to move every valid row of every tile of `geo` as merged face runs."""
    total = 0
    for tr_class, tr_count in ((0, geo["Ht"] - 1), (geo["Ht"] - 1, 1)):
        if tr_count <= 0:
            continue
        v = valid_rows(geo, tr_class)
        for tc_class, tc_count in ((0, geo["Wt"] - 1), (geo["Wt"] - 1, 1)):
            if tc_count <= 0:
                continue
            total += tr_count * tc_count * len(tile_runs(0, v, valid_cols(geo, tc_class), face_row_bytes))
    return geo["IMG"] * total


class StripMap:
    """Strip index <-> (tile, p) for the tile-side tensor (closed form of op_design.md)."""

    def __init__(self, geo, strip_rows):
        self.geo = geo
        self.strip_rows = strip_rows
        self.sf = TILE // strip_rows
        self.last_strips = self.strips(geo["Ht"] - 1)
        self.s_img = geo["Wt"] * (self.sf * (geo["Ht"] - 1) + self.last_strips)

    def strips(self, tr):
        return -(-valid_rows(self.geo, tr) // self.strip_rows)

    @property
    def count(self):
        return self.geo["IMG"] * self.s_img

    def locate(self, s):
        Wt, Ht = self.geo["Wt"], self.geo["Ht"]
        img, s1 = divmod(s, self.s_img)
        tr = min(s1 // (self.sf * Wt), Ht - 1)
        s2 = s1 - tr * self.sf * Wt
        tc, p = divmod(s2, self.strips(tr))
        return (img * Ht + tr) * Wt + tc, p


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _select_regime(aligned, tst_in, tst_out):
    if REGIME_OVERRIDE is not None:
        if REGIME_OVERRIDE not in REGIME_DEFINES:
            raise ValueError(f"g6_reshape: unknown REGIME_OVERRIDE {REGIME_OVERRIDE!r}")
        if REGIME_OVERRIDE != GATHER_GENERAL and not aligned:
            raise ValueError(f"g6_reshape: REGIME_OVERRIDE {REGIME_OVERRIDE!r} requires W_in, W_out multiples of 16")
        return REGIME_OVERRIDE
    if not aligned:
        return GATHER_GENERAL
    return SCATTER_ALIGNED if tst_in <= tst_out else GATHER_ALIGNED


def _knobs():
    return (
        LANES_PER_CORE,
        MIN_STRIPS_PER_LANE,
        BLOCK_SLOT_BYTES_CAP,
        SLOT_DEPTH,
        STAGING_SLOTS_CAP,
        COPY_ENTRIES_CAP,
        REGIME_OVERRIDE,
    )


def build_plan(in_shape, out_shape, esize, grid):
    """Everything the descriptor needs, derived from the shapes, element size, grid and knobs."""
    if LANES_PER_CORE not in (1, 2):
        raise ValueError(f"g6_reshape: LANES_PER_CORE must be 1 or 2, got {LANES_PER_CORE}")
    if SLOT_DEPTH not in (1, 2):
        raise ValueError(f"g6_reshape: SLOT_DEPTH must be 1 or 2, got {SLOT_DEPTH}")
    if MIN_STRIPS_PER_LANE < 1:
        raise ValueError("g6_reshape: MIN_STRIPS_PER_LANE must be >= 1")

    geo_in, geo_out = geometry(in_shape), geometry(out_shape)
    face_row_bytes = FACE * esize
    tile_bytes = TILE * TILE * esize
    granule_bytes = max(ttnn.get_dram_alignment(), face_row_bytes)
    if granule_bytes % face_row_bytes or TILE % (granule_bytes // face_row_bytes):
        raise RuntimeError(f"g6_reshape: granule of {granule_bytes} B does not tile the {face_row_bytes}-B face rows")
    strip_rows = granule_bytes // face_row_bytes

    aligned = geo_in["W"] % FACE == 0 and geo_out["W"] % FACE == 0
    tst_in = tile_side_transfers(geo_in, face_row_bytes)
    tst_out = tile_side_transfers(geo_out, face_row_bytes)
    regime = _select_regime(aligned, tst_in, tst_out)
    tile_side = "input" if regime == SCATTER_ALIGNED else "output"
    geo_ts = geo_in if tile_side == "input" else geo_out

    strip_map = StripMap(geo_ts, strip_rows)
    U = strip_map.count

    grid_cores = grid[0] * grid[1]
    n_lanes = min(LANES_PER_CORE * grid_cores, -(-U // MIN_STRIPS_PER_LANE))
    n_cores = -(-n_lanes // LANES_PER_CORE)

    base, extra = divmod(U, n_lanes)
    lanes = []
    s = 0
    for lane in range(n_lanes):
        num = base + (1 if lane < extra else 0)
        first_tile, p = strip_map.locate(s)
        last_tile, _ = strip_map.locate(s + num - 1)
        lanes.append(
            {"start_tile": first_tile, "start_strip": p, "num_strips": num, "tiles": last_tile - first_tile + 1}
        )
        s += num

    block_tiles_cap = max(1, BLOCK_SLOT_BYTES_CAP // tile_bytes)
    max_lane_tiles = max(lane["tiles"] for lane in lanes)
    block_tiles = min(max_lane_tiles, block_tiles_cap)
    for lane in lanes:
        lane["num_blocks"] = -(-lane["tiles"] // block_tiles)
    max_blocks_per_lane = max(lane["num_blocks"] for lane in lanes)
    slot_depth = min(SLOT_DEPTH, max_blocks_per_lane)

    max_strips = max(lane["num_strips"] for lane in lanes)
    min_strips = min(lane["num_strips"] for lane in lanes)
    max_block_rows = min(max_strips * strip_rows, TILE * block_tiles)
    if regime == SCATTER_ALIGNED:
        staging_slots = copy_entries = 0
    else:
        pieces = 1 if regime == GATHER_ALIGNED else min(FACE, -(-FACE // geo_in["W"]) + 2)
        staged_bound = 2 * max_block_rows * pieces
        staging_slots = max(1, min(STAGING_SLOTS_CAP, staged_bound))
        copy_entries = max(1, min(COPY_ENTRIES_CAP, staged_bound))
        if staging_slots * granule_bytes > STAGING_OFFSET_LIMIT:
            raise ValueError(
                f"g6_reshape: STAGING_SLOTS_CAP={STAGING_SLOTS_CAP} exceeds the copy table's 16-bit offsets"
            )

    tile_images_pages = slot_depth * block_tiles
    staging_pages = staging_slots + -(-(COPY_ENTRY_BYTES * copy_entries) // granule_bytes) if staging_slots else 0
    lanes_on_core = LANES_PER_CORE
    l1_bytes_per_core = lanes_on_core * (tile_images_pages * tile_bytes + staging_pages * granule_bytes)

    grid_coord = ttnn.CoreCoord(grid[0], grid[1])
    core_set = ttnn.num_cores_to_corerangeset(n_cores, grid_coord, True)
    cores = ttnn.corerange_to_cores(core_set, None, True)
    ranges = core_set.ranges()

    return {
        "regime": regime,
        "tile_side": tile_side,
        "aligned": aligned,
        "tile_side_transfers": {"input": tst_in, "output": tst_out},
        "work_units": U,
        "strip_rows": strip_rows,
        "granule_bytes": granule_bytes,
        "esize": esize,
        "tile_bytes": tile_bytes,
        "lanes_per_core": LANES_PER_CORE,
        "min_strips_per_lane": MIN_STRIPS_PER_LANE,
        "grid_cores": grid_cores,
        "n_lanes": n_lanes,
        "n_cores": n_cores,
        "num_core_ranges": len(ranges),
        "core_ranges": [((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in ranges],
        "block_tiles": block_tiles,
        "max_blocks_per_lane": max_blocks_per_lane,
        "slot_depth": slot_depth,
        "staging_slots": staging_slots,
        "copy_entries": copy_entries,
        "strips_per_lane": (min_strips, max_strips),
        "tile_images_pages": tile_images_pages,
        "staging_pages": staging_pages,
        "l1_bytes_per_core": l1_bytes_per_core,
        "roles": {"NCRISC": "lane0", "BRISC": "lane1"} if LANES_PER_CORE == 2 else {"NCRISC": "lane0"},
        "geometry": {"input": geo_in, "output": geo_out},
        "lanes": lanes,
        "_core_set": core_set,
        "_cores": [(c.x, c.y) for c in cores],
    }


# ---------------------------------------------------------------------------
# ProgramDescriptor
# ---------------------------------------------------------------------------


def _lane_kernel(plan, lane, input_tensor, output_tensor, dtype):
    geo_in, geo_out = plan["geometry"]["input"], plan["geometry"]["output"]
    ct_args = [
        lane,
        plan["esize"],
        geo_in["W"],
        geo_in["H"],
        geo_in["Wt"],
        geo_in["Ht"],
        geo_out["W"],
        geo_out["H"],
        geo_out["Wt"],
        geo_out["Ht"],
        plan["strip_rows"],
        plan["granule_bytes"],
        plan["block_tiles"],
        plan["slot_depth"],
        plan["staging_slots"],
        plan["copy_entries"],
        CB_LANE_TILE_IMAGES[lane],
        CB_LANE_STAGING[lane],
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    rt_args = ttnn.RuntimeArgs()
    lanes, lpc = plan["lanes"], plan["lanes_per_core"]
    for core_idx, (x, y) in enumerate(plan["_cores"]):
        g = lpc * core_idx + lane
        if g < len(lanes):
            ln = lanes[g]
            rt_args[x][y] = [ln["start_tile"], ln["start_strip"], ln["num_strips"], ln["num_blocks"]]
        else:
            rt_args[x][y] = [0, 0, 0, 0]

    config = ttnn.ReaderConfigDescriptor() if lane == 0 else ttnn.WriterConfigDescriptor()
    return ttnn.KernelDescriptor(
        kernel_source=str(LANE_KERNEL),
        core_ranges=plan["_core_set"],
        compile_time_args=ct_args,
        defines=[(REGIME_DEFINES[plan["regime"]], "1")],
        runtime_args=rt_args,
        common_runtime_args=[input_tensor.buffer_address(), output_tensor.buffer_address()],
        config=config,
    )


def _lane_cbs(plan, lane, dtype):
    cbs = [
        ttnn.CBDescriptor(
            total_size=plan["tile_images_pages"] * plan["tile_bytes"],
            core_ranges=plan["_core_set"],
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_LANE_TILE_IMAGES[lane], data_format=dtype, page_size=plan["tile_bytes"]
                )
            ],
        )
    ]
    if plan["staging_pages"]:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=plan["staging_pages"] * plan["granule_bytes"],
                core_ranges=plan["_core_set"],
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_LANE_STAGING[lane], data_format=dtype, page_size=plan["granule_bytes"]
                    )
                ],
            )
        )
    return cbs


def create_program_descriptor(plan, input_tensor, output_tensor):
    for name, tensor in (("input", input_tensor), ("output", output_tensor)):
        expected = plan["geometry"][name]["num_tiles"]
        if tensor.buffer_num_pages() != expected:
            raise RuntimeError(
                f"g6_reshape: {name} buffer has {tensor.buffer_num_pages()} pages, geometry gives {expected}"
            )
    dtype = input_tensor.dtype
    kernels, cbs = [], []
    for lane in range(plan["lanes_per_core"]):
        kernels.append(_lane_kernel(plan, lane, input_tensor, output_tensor, dtype))
        cbs.extend(_lane_cbs(plan, lane, dtype))
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


# ---------------------------------------------------------------------------
# Memo (BB-07): one plan + descriptor per call signature; a hit rewrites only the two addresses
# ---------------------------------------------------------------------------

_PLAN_CACHE = {}
_DESCRIPTOR_CACHE = {}


def _grid(device):
    g = device.compute_with_storage_grid_size()
    return (g.x, g.y)


def _signature(input_tensor, out_shape):
    device = input_tensor.device()
    return (
        tuple(int(s) for s in input_tensor.shape),
        tuple(int(s) for s in out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        repr(input_tensor.memory_config()),
        device.id(),
        _grid(device),
        _knobs(),
    )


def get_plan(input_tensor, out_shape):
    key = _signature(input_tensor, out_shape)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = build_plan(
            tuple(input_tensor.shape), tuple(out_shape), input_tensor.element_size(), _grid(input_tensor.device())
        )
        _PLAN_CACHE[key] = plan
    return plan


def get_program_descriptor(input_tensor, output_tensor, out_shape):
    key = _signature(input_tensor, out_shape)
    entry = _DESCRIPTOR_CACHE.get(key)
    if entry is None:
        plan = get_plan(input_tensor, out_shape)
        descriptor = create_program_descriptor(plan, input_tensor, output_tensor)
        _DESCRIPTOR_CACHE[key] = descriptor
        return descriptor
    addresses = [input_tensor.buffer_address(), output_tensor.buffer_address()]
    kernels = entry.kernels
    for kernel in kernels:
        kernel.common_runtime_args = addresses
    entry.kernels = kernels
    return entry


def public_plan(plan):
    """The plan report: everything except the internal core objects."""
    return {k: v for k, v in plan.items() if not k.startswith("_")}
