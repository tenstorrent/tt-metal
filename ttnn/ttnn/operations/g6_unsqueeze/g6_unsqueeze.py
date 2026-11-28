# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_unsqueeze: torch.unsqueeze(x, -1) for a one-row float32 TILE tensor, as one device program.

Row 0 of every input tile becomes column 0 of the output tile with the same page id.
Each mover core owns a contiguous page range; its two data-movement RISCs each move one
tile half (input face h row 0 -> output face 2h column 0).
"""

from __future__ import annotations

import hashlib
import math
import operator
from dataclasses import dataclass

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_unsqueeze_program_descriptor import build_program_descriptor, patch_buffer_addresses


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs[0] is the scenario dict {"input_shape", "dim", "dtype", "layout"}
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Blocking knobs (host constants; every extent of the plan derives from these)
# ---------------------------------------------------------------------------

# None = the whole compute-with-storage grid.
MAX_MOVER_CORES = None
# 2: tile half 0 on RISCV_1/NoC0 and half 1 on RISCV_0/NoC1; 1: one RISC moves both halves.
MOVERS_PER_CORE = 2
# Slot groups resident per mover; 2 overlaps block i+1's gather with block i's scatter/store.
STAGING_DEPTH = 1
# L1 the model leaves to the op on every core (spec rule BB-05).
OP_L1_BUDGET_BYTES = 1347776

REGIME = "half_split_face_scatter"
_MOVER_PLACEMENT = (
    ("mover0", ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
    ("mover1", ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
)


def _raise(message):
    raise ValueError(f"g6_unsqueeze: {message}")


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, dim):
    scenario = {
        "input_shape": tuple(input_tensor.shape),
        "dim": dim,
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_unsqueeze: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_unsqueeze: unsupported combination (refinement candidate): {exc}")

    if not ttnn.is_tensor_storage_on_device(input_tensor):
        _raise("input_tensor must be on device")
    shape = tuple(input_tensor.shape)
    rank = len(shape)
    if rank not in (1, 2, 3):
        _raise(f"input rank must be 1 to 3, got {rank} (shape {list(shape)})")
    if any(d == 0 for d in shape):
        _raise(f"input has no elements (shape {list(shape)})")
    if any(d != 1 for d in shape[:-1]):
        _raise(f"every dim but the last must be 1: one logical row (shape {list(shape)})")
    memory_config = input_tensor.memory_config()
    if (
        memory_config.buffer_type != ttnn.BufferType.DRAM
        or memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        _raise(f"input must be DRAM interleaved, got {memory_config}")
    try:
        dim = operator.index(dim)
    except TypeError:
        _raise(f"dim must be an integer, got {dim!r}")
    if dim < -(rank + 1) or dim > rank:
        _raise(f"dim {dim} out of range [{-(rank + 1)}, {rank}] for an input of rank {rank}")
    canonical = dim if dim >= 0 else dim + rank + 1
    if canonical != rank:
        _raise(f"only the new axis last is supported (dim {rank} or -1), got dim {dim}")


# ---------------------------------------------------------------------------
# Plan — the only selector of extents; dispatch builds from it and it is reportable (BB-08, BB-21)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileGeometry:
    tile_h: int
    tile_w: int
    face_h: int
    face_w: int
    elem_bytes: int

    @property
    def face_bytes(self):
        return self.face_h * self.face_w * self.elem_bytes

    @property
    def face_row_bytes(self):
        return self.face_w * self.elem_bytes

    @property
    def faces_per_face_row(self):
        return self.tile_w // self.face_w

    @property
    def tile_halves(self):
        return self.tile_w // self.face_w


@dataclass(frozen=True)
class MoverRole:
    name: str
    processor: object
    noc: object
    cb_index: int
    first_half: int
    num_halves: int


@dataclass(frozen=True)
class G6UnsqueezePlan:
    regime: str
    geometry: TileGeometry
    grid: tuple
    tensor_w_tiles: int
    max_mover_cores: int
    num_cores: int
    max_core_w_tiles: int
    movers_per_core: int
    block_tile_halves: int
    staging_depth: int
    block_w_tiles_cap: int
    block_w_tiles: int
    core_ranges: tuple
    # ((x, y), first_tile, core_w_tiles) per mover core, row-major inside the rectangle.
    core_work: tuple
    roles: tuple

    def report(self):
        lines = [
            f"g6_unsqueeze plan: regime={self.regime}",
            f"  tensor_w_tiles={self.tensor_w_tiles} grid={self.grid} max_mover_cores={self.max_mover_cores} "
            f"num_cores={self.num_cores} max_core_w_tiles={self.max_core_w_tiles}",
            f"  movers_per_core={self.movers_per_core} block_tile_halves={self.block_tile_halves} "
            f"block_w_tiles={self.block_w_tiles} (cap {self.block_w_tiles_cap}) staging_depth={self.staging_depth}",
            f"  core_ranges={list(self.core_ranges)}",
        ]
        for (x, y), first_tile, core_w_tiles in self.core_work:
            num_blocks = math.ceil(core_w_tiles / self.block_w_tiles)
            roles = ", ".join(
                f"{r.processor}={r.name} halves [{r.first_half}, {r.first_half + r.num_halves}) {r.noc}"
                for r in self.roles
            )
            lines.append(
                f"  core ({x},{y}): pages [{first_tile}, {first_tile + core_w_tiles}) blocks={num_blocks}; {roles}"
            )
        return "\n".join(lines)


def _tile_geometry(input_tensor):
    tile = input_tensor.tile
    tile_h, tile_w = tile.tile_shape
    face_h, face_w = tile.face_shape
    return TileGeometry(tile_h, tile_w, face_h, face_w, input_tensor.element_size())


def _placement(num_cores, grid_x, grid_y):
    """One rectangle when num_cores factors into it, else full rows plus one partial row."""
    for cols in range(min(num_cores, grid_x), 0, -1):
        if num_cores % cols == 0 and num_cores // cols <= grid_y:
            rows = num_cores // cols
            ranges = (((0, 0), (cols - 1, rows - 1)),)
            cores = tuple((k % cols, k // cols) for k in range(num_cores))
            return ranges, cores
    full_rows, tail = divmod(num_cores, grid_x)
    ranges = (((0, 0), (grid_x - 1, full_rows - 1)),)
    if tail:
        ranges += (((0, full_rows), (tail - 1, full_rows)),)
    cores = tuple((k % grid_x, k // grid_x) for k in range(num_cores))
    return ranges, cores


def g6_unsqueeze_plan(input_tensor):
    geometry = _tile_geometry(input_tensor)
    grid_size = input_tensor.device().compute_with_storage_grid_size()
    grid_x, grid_y = grid_size.x, grid_size.y

    tensor_w_tiles = math.ceil(input_tensor.shape[-1] / geometry.tile_w)
    max_mover_cores = grid_x * grid_y if MAX_MOVER_CORES is None else min(MAX_MOVER_CORES, grid_x * grid_y)
    num_cores = min(tensor_w_tiles, max_mover_cores)
    max_core_w_tiles = math.ceil(tensor_w_tiles / num_cores)
    assert geometry.tile_halves % MOVERS_PER_CORE == 0, "movers must split the tile halves evenly"
    block_tile_halves = geometry.tile_halves // MOVERS_PER_CORE
    block_w_tiles_cap = OP_L1_BUDGET_BYTES // (
        MOVERS_PER_CORE * STAGING_DEPTH * block_tile_halves * geometry.face_bytes
    )
    block_w_tiles = min(max_core_w_tiles, block_w_tiles_cap)

    core_ranges, cores = _placement(num_cores, grid_x, grid_y)
    base, extra = divmod(tensor_w_tiles, num_cores)
    core_work = tuple((core, k * base + min(k, extra), base + (1 if k < extra else 0)) for k, core in enumerate(cores))
    roles = tuple(
        MoverRole(
            name=name,
            processor=processor,
            noc=noc,
            cb_index=m,
            first_half=m * block_tile_halves,
            num_halves=block_tile_halves,
        )
        for m, (name, processor, noc) in enumerate(_MOVER_PLACEMENT[:MOVERS_PER_CORE])
    )
    return G6UnsqueezePlan(
        regime=REGIME,
        geometry=geometry,
        grid=(grid_x, grid_y),
        tensor_w_tiles=tensor_w_tiles,
        max_mover_cores=max_mover_cores,
        num_cores=num_cores,
        max_core_w_tiles=max_core_w_tiles,
        movers_per_core=MOVERS_PER_CORE,
        block_tile_halves=block_tile_halves,
        staging_depth=STAGING_DEPTH,
        block_w_tiles_cap=block_w_tiles_cap,
        block_w_tiles=block_w_tiles,
        core_ranges=core_ranges,
        core_work=core_work,
        roles=roles,
    )


def _check_plan(plan, input_tensor, output_tensor):
    geometry = plan.geometry
    dram_alignment = ttnn.get_dram_alignment()
    assert geometry.face_row_bytes % dram_alignment == 0, "face-row reads must be DRAM-aligned"
    assert geometry.face_bytes % dram_alignment == 0, "slot stride must keep every slot DRAM-aligned"
    assert plan.tensor_w_tiles == input_tensor.buffer_num_pages() == output_tensor.buffer_num_pages()
    l1_bytes = plan.movers_per_core * plan.staging_depth * plan.block_w_tiles * plan.block_tile_halves
    assert l1_bytes * geometry.face_bytes <= OP_L1_BUDGET_BYTES, "face slots exceed the op's L1 budget"


# ---------------------------------------------------------------------------
# Host program cache (BB-07): a hit patches the two buffer addresses and nothing else
# ---------------------------------------------------------------------------

_PROGRAM_CACHE = {}


def _cache_key(input_tensor, device):
    memory_config = input_tensor.memory_config()
    grid = device.compute_with_storage_grid_size()
    return (
        device.id(),
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        str(input_tensor.dtype),
        str(input_tensor.layout),
        str(memory_config.buffer_type),
        str(memory_config.memory_layout),
        (grid.x, grid.y),
        (MAX_MOVER_CORES, MOVERS_PER_CORE, STAGING_DEPTH, OP_L1_BUDGET_BYTES),
    )


def _program_hash(key):
    digest = hashlib.blake2b(repr(("g6_unsqueeze", key)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _program_descriptor(input_tensor, output_tensor, device):
    key = _cache_key(input_tensor, device)
    program_descriptor = _PROGRAM_CACHE.get(key)
    if program_descriptor is not None:
        patch_buffer_addresses(program_descriptor, input_tensor, output_tensor)
        return program_descriptor
    plan = g6_unsqueeze_plan(input_tensor)
    _check_plan(plan, input_tensor, output_tensor)
    program_descriptor = build_program_descriptor(plan, input_tensor, output_tensor)
    program_descriptor.custom_program_hash = _program_hash(key)
    _PROGRAM_CACHE[key] = program_descriptor
    return program_descriptor


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_unsqueeze(input_tensor: ttnn.Tensor, dim: int) -> ttnn.Tensor:
    validate(input_tensor, dim)
    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape) + [1]),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )
    program_descriptor = _program_descriptor(input_tensor, output_tensor, device)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
