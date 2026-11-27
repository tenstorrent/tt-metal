# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for g6_reshard (PULL regime, op_design.md → Blocking Model).

Every core holding an out block runs `issuers_per_core` instances of one
dataflow source. Each instance pulls the pieces of in0 overlapping its row
range of the out block straight into the out shard (no CB, no compute kernel),
enumerating the pieces on device from the two placements' geometry.
"""

from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
ISSUER_KERNEL = KERNEL_DIR / "g6_reshard_reader.cpp"

TILE_HW = 32

# L1 per core the model leaves to the op (BB-05).
L1_FREE_BYTES = 1347776

# Blocking-model knob: how many data-movement RISC-Vs split each out block's
# tile rows. Each issuer slot names its processor and default NoC; the first
# ISSUERS_PER_CORE slots are used.
ISSUERS_PER_CORE = 2
ISSUER_SLOTS = (
    (ttnn.DataMovementProcessor.RISCV_1, 0),
    (ttnn.DataMovementProcessor.RISCV_0, 1),
)

# NoC-assignment knob.
#   "fixed":     every issuer reads on its slot's default NoC (dedicated NoC mode).
#   "wrap_free": when every piece of the placement pair reaches its target without
#                a torus wrap on one NoC, every issuer reads on that NoC; otherwise
#                each keeps its slot default. NoC0 carries data east/south, NoC1
#                west/north. Moving only some issuers piles their bytes onto a NoC
#                that already carries the other issuers' wrapped traffic, so the
#                choice is all-or-nothing. Two issuers sharing a NoC needs dynamic
#                NoC mode.
#   "balanced":  starts from "wrap_free", then moves single issuers between NoCs
#                while that lowers the busiest directed link (NIU ports included)
#                of a dimension-order route model over the device's core coordinates.
NOC_POLICY = "balanced"

# Common RT arg slots, rewritten on every call (both buffers are lockstep-allocated:
# one address serves every core).
COMMON_ARG_IN_ADDR = 0
COMMON_ARG_OUT_ADDR = 1

PLAN_REGIME = "PULL"


@dataclass(frozen=True)
class Placement:
    """Block grid of the padded 2-D view under one BLOCK_SHARDED shard spec."""

    origin_x: int
    origin_y: int
    orientation: object
    block_height_tiles: int
    block_width_tiles: int
    tensor_height_tiles: int
    tensor_width_tiles: int

    @property
    def num_block_rows(self):
        return -(-self.tensor_height_tiles // self.block_height_tiles)

    @property
    def num_block_cols(self):
        return -(-self.tensor_width_tiles // self.block_width_tiles)

    def core_of(self, block_row, block_col):
        if self.orientation == ttnn.ShardOrientation.ROW_MAJOR:
            return self.origin_x + block_col, self.origin_y + block_row
        return self.origin_x + block_row, self.origin_y + block_col

    def kernel_range(self):
        last_x, last_y = self.core_of(self.num_block_rows - 1, self.num_block_cols - 1)
        return ttnn.CoreRange(ttnn.CoreCoord(self.origin_x, self.origin_y), ttnn.CoreCoord(last_x, last_y))


def tensor_tiles(tensor):
    padded = list(tensor.padded_shape)
    height = 1
    for d in padded[:-1]:
        height *= d
    return height // TILE_HW, padded[-1] // TILE_HW


def placement_of(tensor, memory_config):
    spec = memory_config.shard_spec
    start = spec.grid.bounding_box().start
    tensor_height_tiles, tensor_width_tiles = tensor_tiles(tensor)
    return Placement(
        origin_x=start.x,
        origin_y=start.y,
        orientation=spec.orientation,
        block_height_tiles=int(spec.shape[0]) // TILE_HW,
        block_width_tiles=int(spec.shape[1]) // TILE_HW,
        tensor_height_tiles=tensor_height_tiles,
        tensor_width_tiles=tensor_width_tiles,
    )


def issuer_row_ranges(row_begin, row_end, issuers):
    """Contiguous split of [row_begin, row_end): the first issuers take the ceiling share."""
    rows = row_end - row_begin
    ranges = []
    begin = row_begin
    for k in range(issuers):
        end = begin + rows // issuers + (1 if k < rows % issuers else 0)
        ranges.append((begin, end))
        begin = end
    return ranges


def iter_pieces(row_begin, row_end, col_begin, col_end, in_block_height_tiles, in_block_width_tiles):
    """Host twin of the kernel's piece walk: (in block row, in block col, piece rows, piece width)."""
    if row_begin >= row_end:
        return
    for ri in range(row_begin // in_block_height_tiles, (row_end - 1) // in_block_height_tiles + 1):
        piece_rows = min(row_end, (ri + 1) * in_block_height_tiles) - max(row_begin, ri * in_block_height_tiles)
        for ci in range(col_begin // in_block_width_tiles, (col_end - 1) // in_block_width_tiles + 1):
            piece_width = min(col_end, (ci + 1) * in_block_width_tiles) - max(col_begin, ci * in_block_width_tiles)
            yield ri, ci, piece_rows, piece_width


def _wrap_free(noc, src, dst):
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    if noc == 0:
        return dx >= 0 and dy >= 0
    return dx <= 0 and dy <= 0


def choose_nocs(roles, in_p, policy):
    """NoC of every issuer of every role, under `policy` (see NOC_POLICY)."""
    defaults = [[ISSUER_SLOTS[k][1] for k in range(len(role["issuers"]))] for role in roles]
    if policy == "fixed":
        return defaults
    if policy not in ("wrap_free", "balanced"):
        raise ValueError(f"g6_reshard: unknown NoC policy {policy!r}")
    all_free = [True, True]
    for role in roles:
        for issuer in role["issuers"]:
            for ri, ci, _, _ in issuer["piece_list"]:
                src_core = in_p.core_of(ri, ci)
                for noc in (0, 1):
                    all_free[noc] = all_free[noc] and _wrap_free(noc, src_core, role["core"])
    if all_free[0] == all_free[1]:
        return defaults
    shared = 0 if all_free[0] else 1
    return [[shared] * len(role["issuers"]) for role in roles]


# NoC torus extent (columns, rows) in NoC coordinates, per architecture.
NOC_GRID = {"BLACKHOLE": (17, 12), "WORMHOLE_B0": (10, 12)}


def _route_links(noc, src, dst, grid):
    """Directed links a read's data crosses from src to dst: NoC0 +x then +y, NoC1 -y then -x."""
    cols, rows = grid
    x, y = src
    links = [("egress", noc, src)]
    if noc == 0:
        while x != dst[0]:
            nx = (x + 1) % cols
            links.append((noc, x, y, nx, y))
            x = nx
        while y != dst[1]:
            ny = (y + 1) % rows
            links.append((noc, x, y, x, ny))
            y = ny
    else:
        while y != dst[1]:
            ny = (y - 1) % rows
            links.append((noc, x, y, x, ny))
            y = ny
        while x != dst[0]:
            nx = (x - 1) % cols
            links.append((noc, x, y, nx, y))
            x = nx
    links.append(("ingress", noc, dst))
    return links


def balance_nocs(roles, in_p, nocs, device, max_passes=16):
    """Greedy single-issuer NoC moves that lower (busiest link, sum of squared link loads)."""
    grid = NOC_GRID.get(str(device.arch()).split(".")[-1])
    if grid is None:
        return nocs
    coord_cache = {}

    def noc_xy(core):
        if core not in coord_cache:
            v = device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
            coord_cache[core] = (v.x, v.y)
        return coord_cache[core]

    # Per issuer, for each NoC: the (link, tiles) it adds.
    contrib = []
    for role in roles:
        dst = noc_xy(role["core"])
        for issuer in role["issuers"]:
            per_noc = []
            for noc in (0, 1):
                acc = collections.Counter()
                for ri, ci, piece_rows, piece_width in issuer["piece_list"]:
                    for link in _route_links(noc, noc_xy(in_p.core_of(ri, ci)), dst, grid):
                        acc[link] += piece_rows * piece_width
                per_noc.append(acc)
            contrib.append(per_noc)

    flat = [noc for row in nocs for noc in row]
    load = collections.Counter()
    for k, noc in enumerate(flat):
        load.update(contrib[k][noc])

    def score():
        return max(load.values(), default=0), sum(v * v for v in load.values())

    best = score()
    for _ in range(max_passes):
        improved = False
        for k in range(len(flat)):
            old, new = flat[k], 1 - flat[k]
            load.subtract(contrib[k][old])
            load.update(contrib[k][new])
            candidate = score()
            if candidate < best:
                best, flat[k], improved = candidate, new, True
            else:
                load.subtract(contrib[k][new])
                load.update(contrib[k][old])
        if not improved:
            break

    out, k = [], 0
    for row in nocs:
        out.append(flat[k : k + len(row)])
        k += len(row)
    return out


def describe_plan(input_tensor, memory_config):
    """Per out block: holder core, clipped block rectangle, and each issuer's rows, pieces and NoC."""
    issuers_per_core = ISSUERS_PER_CORE
    in_p = placement_of(input_tensor, input_tensor.memory_config())
    out_p = placement_of(input_tensor, memory_config)
    roles = []
    for r in range(out_p.num_block_rows):
        row_begin = r * out_p.block_height_tiles
        row_end = min(out_p.tensor_height_tiles, row_begin + out_p.block_height_tiles)
        for c in range(out_p.num_block_cols):
            col_begin = c * out_p.block_width_tiles
            col_end = min(out_p.tensor_width_tiles, col_begin + out_p.block_width_tiles)
            core = out_p.core_of(r, c)
            issuers = []
            for begin, end in issuer_row_ranges(row_begin, row_end, issuers_per_core):
                pieces = list(
                    iter_pieces(begin, end, col_begin, col_end, in_p.block_height_tiles, in_p.block_width_tiles)
                )
                runs = sum(
                    1 if piece_width == in_p.block_width_tiles == out_p.block_width_tiles else piece_rows
                    for _, _, piece_rows, piece_width in pieces
                )
                issuers.append({"rows": (begin, end), "pieces": len(pieces), "runs": runs, "piece_list": pieces})
            roles.append(
                {
                    "core": core,
                    "block": (r, c),
                    "block_rows": (row_begin, row_end),
                    "block_cols": (col_begin, col_end),
                    "issuers": issuers,
                }
            )
    nocs = choose_nocs(roles, in_p, NOC_POLICY)
    if NOC_POLICY == "balanced":
        nocs = balance_nocs(roles, in_p, nocs, input_tensor.device())
    for role, role_nocs in zip(roles, nocs):
        for issuer, noc in zip(role["issuers"], role_nocs):
            issuer["noc"] = noc
            del issuer["piece_list"]
    return {
        "regime": PLAN_REGIME,
        "issuers_per_core": issuers_per_core,
        "noc_policy": NOC_POLICY,
        "kernel_range": out_p.kernel_range(),
        "roles": roles,
    }


def _placement_key(memory_config):
    spec = memory_config.shard_spec
    ranges = tuple(sorted((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges()))
    return (ranges, tuple(int(d) for d in spec.shape), str(spec.orientation))


def _cache_key(input_tensor, memory_config):
    return (
        "g6_reshard",
        tuple(input_tensor.padded_shape),
        str(input_tensor.dtype),
        str(input_tensor.layout),
        _placement_key(input_tensor.memory_config()),
        _placement_key(memory_config),
        input_tensor.device().id(),
        ISSUERS_PER_CORE,
        NOC_POLICY,
    )


def _program_hash(key):
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "little")


def _build(input_tensor, output_tensor, memory_config):
    plan = describe_plan(input_tensor, memory_config)
    in_p = placement_of(input_tensor, input_tensor.memory_config())
    out_p = placement_of(input_tensor, memory_config)
    aligned_page_bytes = input_tensor.buffer_aligned_page_size()

    core_ranges = ttnn.CoreRangeSet([plan["kernel_range"]])

    compile_time_args = [
        in_p.block_height_tiles,
        in_p.block_width_tiles,
        out_p.block_width_tiles,
        out_p.tensor_width_tiles,
        aligned_page_bytes,
    ]
    compile_time_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    common_runtime_args = [0, 0]
    common_runtime_args[COMMON_ARG_IN_ADDR] = input_tensor.buffer_address()
    common_runtime_args[COMMON_ARG_OUT_ADDR] = output_tensor.buffer_address()

    # A RISC may read on a NoC other than its configured one only in dynamic NoC mode.
    off_default = any(
        issuer["noc"] != ISSUER_SLOTS[k][1] for role in plan["roles"] for k, issuer in enumerate(role["issuers"])
    )
    noc_mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC if off_default else ttnn.NOC_MODE.DM_DEDICATED_NOC

    kernels = []
    for issuer in range(ISSUERS_PER_CORE):
        processor, default_noc = ISSUER_SLOTS[issuer]
        runtime_args = ttnn.RuntimeArgs()
        for role in plan["roles"]:
            x, y = role["core"]
            row_begin, row_end = role["issuers"][issuer]["rows"]
            col_begin, col_end = role["block_cols"]
            runtime_args[x][y] = [
                row_begin,
                row_end,
                role["block_rows"][0],
                col_begin,
                col_end,
                role["issuers"][issuer]["noc"],
            ]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(ISSUER_KERNEL),
                core_ranges=core_ranges,
                compile_time_args=compile_time_args,
                runtime_args=runtime_args,
                common_runtime_args=common_runtime_args,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=processor, noc=ttnn.NOC.NOC_0 if default_noc == 0 else ttnn.NOC.NOC_1, noc_mode=noc_mode
                ),
            )
        )

    program_descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
    return program_descriptor, plan


_plan_cache = {}


def get_program_descriptor(input_tensor, output_tensor, memory_config):
    """Cached descriptor for this placement pair; a hit only rewrites the two buffer addresses (BB-07)."""
    key = _cache_key(input_tensor, memory_config)
    entry = _plan_cache.get(key)
    if entry is None:
        program_descriptor, plan = _build(input_tensor, output_tensor, memory_config)
        program_descriptor.custom_program_hash = _program_hash(key)
        _plan_cache[key] = (program_descriptor, plan)
        return program_descriptor, plan

    program_descriptor, plan = entry
    # The kernels list comes back as a copy: patch it and assign it back.
    kernels = program_descriptor.kernels
    for kernel in kernels:
        args = list(kernel.common_runtime_args)
        args[COMMON_ARG_IN_ADDR] = input_tensor.buffer_address()
        args[COMMON_ARG_OUT_ADDR] = output_tensor.buffer_address()
        kernel.common_runtime_args = args
    program_descriptor.kernels = kernels
    return program_descriptor, plan
