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

import numpy as np

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
#   "split":     starts from "balanced", then searches each issuer's NoC and, per
#                SPLIT_SHAPE, a cut of its sub-block whose far side reads on the other
#                NoC, lowering a p-norm of the same route model's loads with rising p.
#                Keeps the "balanced" plan unless the result lowers the busiest link
#                without raising the busiest NIU port.
NOC_POLICY = "split"

# What the "split" policy may choose per issuer.
#   "issuer": one NoC.
#   "cut":    one NoC, or one tile-row or piece-column cut with a NoC per side. The route
#             model ranks it best, yet on Blackhole it measures slower than "issuer": an
#             issuer's reads on its second NoC take ~300 cycles longer to issue, a cost the
#             model (response bytes per link) does not see. Splitting rows into segments on
#             one NoC costs nothing, so the cross-NoC switch is the cost.
SPLIT_SHAPE = "issuer"
# Candidate cut positions per axis; bounds the host search per issuer.
SPLIT_MAX_ROW_CUTS = 6
SPLIT_MAX_COL_CUTS = 4
# Link-load weight the "split" search gives a read whose route crosses the torus seam;
# 1.0 is the plain link-sum model. Above 1 the search avoids wrapped routes (stops the
# "cut" regression on I5), but it also drops the wrapped single-issuer moves that make
# "issuer" faster on P.
SPLIT_WRAP_WEIGHT = 1.0

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


_NIU = ("egress", "ingress")


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


def _route_model(device):
    """(grid, logical core -> NoC coordinate) of the device's route model, or None if not modeled."""
    grid = NOC_GRID.get(str(device.arch()).split(".")[-1])
    if grid is None:
        return None
    coord_cache = {}

    def noc_xy(core):
        if core not in coord_cache:
            v = device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
            coord_cache[core] = (v.x, v.y)
        return coord_cache[core]

    return grid, noc_xy


def balance_nocs(roles, in_p, nocs, device, max_passes=16):
    """Greedy single-issuer NoC moves that lower (busiest link, sum of squared link loads)."""
    model = _route_model(device)
    if model is None:
        return nocs
    grid, noc_xy = model

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


def _subsample(values, cap):
    """At most `cap` of the sorted `values`, evenly spaced, first and last kept."""
    if len(values) <= cap:
        return list(values)
    n = len(values)
    return sorted({values[round(i * (n - 1) / (cap - 1))] for i in range(cap)})


def _issuer_options(main_default, rows, cols, piece_rows0, piece_rows, piece_cols0, piece_widths, shape):
    """Per-issuer encodings (noc, cut) and, per encoding, the tiles of each piece on NoC0.

    A cut ("row", r) or ("col", c) sends tile rows >= r, or the pieces starting at column >= c,
    to the other NoC. The first option is (main_default, None); options with the same NoC0
    tiles keep the first.
    """
    cuts = [None]
    if shape == "cut":
        row_cuts = _subsample(list(range(rows[0] + 1, rows[1])), SPLIT_MAX_ROW_CUTS)
        col_cuts = _subsample(sorted({c for c in piece_cols0 if c > cols[0]}), SPLIT_MAX_COL_CUTS)
        cuts += [("row", r) for r in row_cuts] + [("col", c) for c in col_cuts]
    elif shape != "issuer":
        raise ValueError(f"g6_reshard: unknown SPLIT_SHAPE {shape!r}")

    p0 = np.asarray(piece_rows0)
    p1 = p0 + np.asarray(piece_rows)
    c0 = np.asarray(piece_cols0)
    w = np.asarray(piece_widths)
    tiles = (p1 - p0) * w
    options, seen, noc0 = [], set(), []
    for main in (main_default, 1 - main_default):
        for cut in cuts:
            if cut is None:
                far_tiles = np.zeros_like(tiles)
            elif cut[0] == "row":
                far_tiles = np.clip(p1 - np.maximum(p0, cut[1]), 0, None) * w
            else:
                far_tiles = tiles * (c0 >= cut[1])
            on_noc0 = tiles - far_tiles if main == 0 else far_tiles
            key = on_noc0.tobytes()
            if key in seen:
                continue
            seen.add(key)
            options.append((main, cut))
            noc0.append(on_noc0)
    return options, np.asarray(noc0, dtype=np.float64), tiles


def issuer_segments(rows, cols, noc, cut):
    """Rectangles (row begin, row end, col begin, col end, NoC) one issuer reads; the first starts at cols[0]."""
    (r0, r1), (c0, c1) = rows, cols
    if cut is None:
        return [(r0, r1, c0, c1, noc)]
    if cut[0] == "row":
        return [(r0, cut[1], c0, c1, noc), (cut[1], r1, c0, c1, 1 - noc)]
    return [(r0, r1, c0, cut[1], noc), (r0, r1, cut[1], c1, 1 - noc)]


def split_nocs(roles, in_p, nocs, device, shape, powers=(4, 8, 16, 32, 64), max_passes=20):
    """Per issuer (NoC, cut) from a p-norm local search over the route model's loads.

    Starts at `nocs` with no cut. Each pass gives every issuer, in turn, the encoding that
    minimizes sum(load ** p) over all links and NIU ports; p rises so the search ends on the busiest
    load. Returns the start unless a result lowers the busiest link without raising the busiest NIU port.
    """
    start = [[(noc, None) for noc in row] for row in nocs]
    model = _route_model(device)
    if model is None:
        return start
    grid, noc_xy = model

    resource_ids = {}
    issuers = []  # per issuer: (role index, issuer index, options, touched resources, load per option)
    for ri_, role in enumerate(roles):
        dst = noc_xy(role["core"])
        for k, issuer in enumerate(role["issuers"]):
            begin, end = issuer["rows"]
            if begin >= end:
                continue
            pr0, prs, pc0, pws, routes = [], [], [], [], []
            for ri, ci, piece_rows, piece_width in issuer["piece_list"]:
                pr0.append(max(begin, ri * in_p.block_height_tiles))
                prs.append(piece_rows)
                pc0.append(max(role["block_cols"][0], ci * in_p.block_width_tiles))
                pws.append(piece_width)
                src = noc_xy(in_p.core_of(ri, ci))
                per_noc = []
                for noc in (0, 1):
                    weight = 1.0 if _wrap_free(noc, src, dst) else SPLIT_WRAP_WEIGHT
                    per_noc.append(
                        [
                            (resource_ids.setdefault(link, len(resource_ids)), 1.0 if link[0] in _NIU else weight)
                            for link in _route_links(noc, src, dst, grid)
                        ]
                    )
                routes.append(per_noc)
            options, noc0, tiles = _issuer_options(
                nocs[ri_][k], (begin, end), role["block_cols"], pr0, prs, pc0, pws, shape
            )
            touched = sorted({r for route in routes for per_noc in route for r, _ in per_noc})
            column = {r: i for i, r in enumerate(touched)}
            on0 = np.zeros((len(routes), len(touched)))
            on1 = np.zeros((len(routes), len(touched)))
            for j, (route0, route1) in enumerate(routes):
                for r, w in route0:
                    on0[j, column[r]] = w
                for r, w in route1:
                    on1[j, column[r]] = w
            load = noc0 @ on0 + (tiles[None, :] - noc0) @ on1
            issuers.append((ri_, k, options, np.asarray(touched), load))
    if not issuers:
        return start

    total = np.zeros(len(resource_ids))
    choice = [0] * len(issuers)
    for (_, _, _, touched, load), j in zip(issuers, choice):
        total[touched] += load[j]

    niu = np.zeros(len(resource_ids), dtype=bool)
    for link, r in resource_ids.items():
        niu[r] = link[0] in _NIU
    # Measured: a NoC's NIU port carrying twice the tiles costs time even with no link traffic,
    # so a plan that only trades link load for NIU-port load is not taken.
    niu_cap = float(total[niu].max())

    def score(loads):
        return float(loads[~niu].max(initial=0.0)), float((loads * loads).sum())

    best_score, best_choice = score(total), list(choice)
    scale = max(float(total.max()), 1.0)
    for p in powers:
        for _ in range(max_passes):
            changed = False
            for i, (_, _, _, touched, load) in enumerate(issuers):
                base = total[touched] - load[choice[i]]
                cost = (((base[None, :] + load) / scale) ** p).sum(axis=1)
                j = int(np.argmin(cost))
                if cost[j] < cost[choice[i]] * (1 - 1e-12):
                    total[touched] = base + load[j]
                    choice[i] = j
                    changed = True
            if not changed:
                break
        candidate = score(total)
        if candidate[0] < best_score[0] and float(total[niu].max()) <= niu_cap:
            best_score, best_choice = candidate, list(choice)

    out = [list(row) for row in start]
    for (ri_, k, options, _, _), j in zip(issuers, best_choice):
        out[ri_][k] = options[j]
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
    nocs = choose_nocs(roles, in_p, "wrap_free" if NOC_POLICY == "split" else NOC_POLICY)
    if NOC_POLICY in ("balanced", "split"):
        nocs = balance_nocs(roles, in_p, nocs, input_tensor.device())
    if NOC_POLICY == "split":
        encodings = split_nocs(roles, in_p, nocs, input_tensor.device(), SPLIT_SHAPE)
    else:
        encodings = [[(noc, None) for noc in row] for row in nocs]
    for role, role_encodings in zip(roles, encodings):
        for issuer, (noc, cut) in zip(role["issuers"], role_encodings):
            issuer["noc"] = noc
            issuer["cut"] = cut
            issuer["segments"] = issuer_segments(issuer["rows"], role["block_cols"], noc, cut)
            del issuer["piece_list"]
    return {
        "regime": PLAN_REGIME,
        "issuers_per_core": issuers_per_core,
        "noc_policy": NOC_POLICY,
        "split_shape": SPLIT_SHAPE if NOC_POLICY == "split" else None,
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
        SPLIT_SHAPE,
        SPLIT_MAX_ROW_CUTS,
        SPLIT_MAX_COL_CUTS,
        SPLIT_WRAP_WEIGHT,
    )


def _program_hash(key):
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "little")


def _build(input_tensor, output_tensor, memory_config):
    plan = describe_plan(input_tensor, memory_config)
    in_p = placement_of(input_tensor, input_tensor.memory_config())
    out_p = placement_of(input_tensor, memory_config)
    aligned_page_bytes = input_tensor.buffer_aligned_page_size()

    core_ranges = ttnn.CoreRangeSet([plan["kernel_range"]])

    max_segments = max(len(issuer["segments"]) for role in plan["roles"] for issuer in role["issuers"])
    compile_time_args = [
        in_p.block_height_tiles,
        in_p.block_width_tiles,
        out_p.block_width_tiles,
        out_p.tensor_width_tiles,
        aligned_page_bytes,
        max_segments,
    ]
    compile_time_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    common_runtime_args = [0, 0]
    common_runtime_args[COMMON_ARG_IN_ADDR] = input_tensor.buffer_address()
    common_runtime_args[COMMON_ARG_OUT_ADDR] = output_tensor.buffer_address()

    # A RISC may read on a NoC other than its configured one only in dynamic NoC mode.
    off_default = any(
        segment[4] != ISSUER_SLOTS[k][1]
        for role in plan["roles"]
        for k, issuer in enumerate(role["issuers"])
        for segment in issuer["segments"]
    )
    noc_mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC if off_default else ttnn.NOC_MODE.DM_DEDICATED_NOC

    kernels = []
    for issuer in range(plan["issuers_per_core"]):
        processor, default_noc = ISSUER_SLOTS[issuer]
        runtime_args = ttnn.RuntimeArgs()
        for role in plan["roles"]:
            x, y = role["core"]
            segments = role["issuers"][issuer]["segments"]
            # Unused segment slots carry an empty row range.
            padding = [(0, 0, 0, 0, segments[0][4])] * (max_segments - len(segments))
            runtime_args[x][y] = [role["block_rows"][0]] + [arg for segment in segments + padding for arg in segment]
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
