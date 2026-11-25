# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Map non-deterministic (ND) output elements back to the Tensix cores that produced them,
and describe the spatial pattern of the failing cores.

Pure torch, no ttnn, so it can be unit tested off-device (test_nd_core_map.py).

A test opts in by telling OpTestBase how its output is laid out across cores:
  nd_core_block      (rows, cols) of the output block each core writes
  nd_core_linear_x   optional: blocks are assigned to cores row-wise over a grid this many cores wide
                     (1D matmul with interleaved output), instead of block (r, c) -> core (c, r)
  nd_core_transpose  optional: block (r, c) -> core (r, c), for ops that put M on cores_x
                     (e.g. minimal_matmul when M > N)

Reading the pattern: in a 2D multicast matmul one input is forwarded core to core along each row
and the other down each column. So a single failing core points at that core, while a run of
cores along one row or column can also be a bad forwarding hop into the run on that axis.
"""

import re

import torch


def per_block_mismatches(ref, out, block_h, block_w):
    """Count mismatching elements in each (block_h x block_w) output block.

    Returns an int tensor [blocks_y, blocks_x]. Partial blocks at the bottom/right edge count what they
    hold, so the last core still counts. Leading (batch) dims are summed.
    Works one block row at a time: a full-size int copy of a 16k x 16k output is gigabytes.
    """
    diff = ref != out
    h, w = diff.shape[-2], diff.shape[-1]
    if diff.numel() != h * w:
        diff = diff.reshape(-1, h, w).sum(dim=0, dtype=torch.int32)
    else:
        diff = diff.reshape(h, w)
    blocks_y, blocks_x = -(-h // block_h), -(-w // block_w)
    pad_w = blocks_x * block_w - w
    counts = torch.zeros(blocks_y, blocks_x, dtype=torch.int64)
    for by in range(blocks_y):
        per_col = diff[by * block_h : (by + 1) * block_h].sum(dim=0, dtype=torch.int32)
        if pad_w:
            per_col = torch.nn.functional.pad(per_col, (0, pad_w))
        counts[by] = per_col.reshape(blocks_x, block_w).sum(dim=1)
    return counts


def block_to_core(by, bx, blocks_x, linear_x=None, transpose=False):
    """Logical core (x, y) that wrote output block (by, bx)."""
    if linear_x:
        idx = by * blocks_x + bx
        return idx % linear_x, idx // linear_x
    return (by, bx) if transpose else (bx, by)


def core_map(ref, out, block, linear_x=None, transpose=False):
    """{(x, y): (mismatching element count, (by, bx) output block)} for every core with a mismatch."""
    counts = per_block_mismatches(ref, out, *block)
    blocks_x = counts.shape[1]
    return {
        block_to_core(by, bx, blocks_x, linear_x, transpose): (int(counts[by, bx]), (by, bx))
        for by, bx in counts.nonzero().tolist()
    }


def core_mismatches(ref, out, block, linear_x=None, transpose=False):
    """{(x, y): mismatching element count} for every core with at least one mismatch."""
    return {core: n for core, (n, _) in core_map(ref, out, block, linear_x, transpose).items()}


TILE = 32
FACE = 16
STRIP_FRAC = 0.5  # a row/column of a core's block counts as a strip when at least this much of it is wrong
DOMINANT_FRAC = 0.8  # a structure label needs this share of the core's wrong elements


def core_blocks(ref, out, block, linear_x=None, transpose=False):
    """{(x, y): (by, bx)} output block index of every failing core."""
    return {core: b for core, (_, b) in core_map(ref, out, block, linear_x, transpose).items()}


def _bits(t):
    """Raw bit pattern as an int tensor, for float formats we can reinterpret."""
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16).to(torch.int32) & 0xFFFF, 16
    if t.dtype == torch.float32:
        return t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF, 32
    return None, 0


def block_structure(ref, out, by, bx, block):
    """Where the wrong elements sit inside one core's output block, to tell which stage corrupted them.

    In a matmul out = in0 x in1:
      row strips     whole output rows wrong       -> a bad in0 value (the input forwarded along rows)
      column strips  whole output columns wrong    -> a bad in1 value (the input forwarded down columns)
      whole tiles    complete 32x32 tiles wrong    -> a tile moved or written wrong (output writer, L1 overwrite)
      face column/row lane  the same position inside every 16x16 face, across the block -> a datapath
                     lane in unpack/math/Dest/pack, not an input value (strips that repeat every 16)
      scattered      isolated elements, often in particular faces or bit lanes -> math/Dest or pack
    Which RISC carries in0/in1 depends on the op: minimal_matmul, not transposed, moves in0 and writes
    output on NCRISC and moves in1 on BRISC; transposed (M > N) swaps them.
    """
    h, w = ref.shape[-2], ref.shape[-1]
    if ref.numel() != h * w:
        return {"label": "n/a (batched output)"}
    bh, bw = block
    r0, c0 = by * bh, bx * bw
    r1, c1 = min(r0 + bh, h), min(c0 + bw, w)
    rb, ob = ref.reshape(h, w)[r0:r1, c0:c1], out.reshape(h, w)[r0:r1, c0:c1]
    d = rb != ob
    n = int(d.sum())
    rows, cols = r1 - r0, c1 - c0
    if n == 0:
        return {"label": "none", "elements": 0}

    strip_rows = d.sum(dim=1) >= STRIP_FRAC * cols
    strip_cols = d.sum(dim=0) >= STRIP_FRAC * rows
    in_row_strips = int(d[strip_rows].sum())
    in_col_strips = int(d[:, strip_cols].sum())

    # Tiles of this block (pad partial edge tiles)
    th, tw = -(-rows // TILE), -(-cols // TILE)
    dt = torch.nn.functional.pad(d, (0, tw * TILE - cols, 0, th * TILE - rows)).reshape(th, TILE, tw, TILE)
    per_tile = dt.sum(dim=(1, 3))
    in_whole_tiles = int(per_tile[per_tile >= 0.9 * TILE * TILE].sum())

    # Position inside a face (0..15), summed over every face of every tile. A bad input value lands on
    # arbitrary rows/columns; a datapath lane defect keeps hitting the same face row or column.
    face_cols = dt.reshape(th, TILE, tw, 2, FACE).sum(dim=(0, 1, 2, 3))
    face_rows = dt.reshape(th, 2, FACE, tw, TILE).sum(dim=(0, 1, 3, 4))
    col_lanes = sorted(((i, int(v) / n) for i, v in enumerate(face_cols.tolist()) if v), key=lambda t: -t[1])
    row_lanes = sorted(((i, int(v) / n) for i, v in enumerate(face_rows.tolist()) if v), key=lambda t: -t[1])

    label = "scattered"
    for name, share in (
        ("whole tiles", in_whole_tiles / n),
        ("row strips", in_row_strips / n),
        ("column strips", in_col_strips / n),
    ):
        if share >= DOMINANT_FRAC:
            label = name
            break
    # Same face column (or row) repeating across most 16-wide face groups: a lane, not an input value.
    # Checked last so it overrides strips, which a lane defect also produces (1/16 of all columns wrong).
    # A single bad row or column also sits on one lane but in only one face group, so it stays a strip.
    col_groups = d.any(dim=0)
    col_groups = torch.nn.functional.pad(col_groups, (0, -(-cols // FACE) * FACE - cols)).reshape(-1, FACE).any(dim=1)
    row_groups = d.any(dim=1)
    row_groups = torch.nn.functional.pad(row_groups, (0, -(-rows // FACE) * FACE - rows)).reshape(-1, FACE).any(dim=1)
    if col_lanes and col_lanes[0][1] >= DOMINANT_FRAC and col_groups.float().mean() >= 0.5:
        label = f"face column lane {col_lanes[0][0]}"
    elif row_lanes and row_lanes[0][1] >= DOMINANT_FRAC and row_groups.float().mean() >= 0.5:
        label = f"face row lane {row_lanes[0][0]}"

    result = {"label": label, "elements": n, "col_lanes": col_lanes}

    # Which bit positions flipped, as a share of all flipped bits
    rbits, nbits = _bits(rb[d])
    obits, _ = _bits(ob[d])
    if nbits:
        x = rbits ^ obits
        flips = [int(((x >> b) & 1).sum()) for b in range(nbits)]
        total = sum(flips)
        if total:
            result["bits"] = sorted(((b, f / total) for b, f in enumerate(flips) if f), key=lambda t: -t[1])
    return result


def summarize_structures(structs):
    """Aggregate one core's block_structure() results over all failing checks into a signature."""
    structs = [s for s in structs if s.get("elements")]
    if not structs:
        return None
    labels = {}
    for s in structs:
        labels[s["label"]] = labels.get(s["label"], 0) + 1
    lanes = {}
    for s in structs:
        if s.get("col_lanes"):
            lane = s["col_lanes"][0][0]
            lanes[lane] = lanes.get(lane, 0) + 1
    bits, with_bits = {}, [s for s in structs if "bits" in s]
    for s in with_bits:
        for b, f in s["bits"]:
            bits[b] = bits.get(b, 0.0) + f / len(with_bits)
    elements = [s["elements"] for s in structs]
    return {
        "checks": len(structs),
        "labels": dict(sorted(labels.items(), key=lambda t: -t[1])),
        "top_col_lane": max(lanes.items(), key=lambda t: t[1]) if lanes else None,
        "elements_min": min(elements),
        "elements_max": max(elements),
        "bits": sorted(bits.items(), key=lambda t: -t[1])[:3],
    }


def describe_summary(s):
    """One-line rendering of summarize_structures()."""
    if not s:
        return "no structure"
    (label, n), *others = s["labels"].items()
    parts = [f"{label} in {n}/{s['checks']} checks"]
    if others:
        parts[0] += " (also " + ", ".join(f"{k} x{v}" for k, v in others) + ")"
    lane = s["top_col_lane"]
    if lane and not label.startswith("face column lane"):
        parts.append(f"most common face column {lane[0]} ({lane[1]}/{s['checks']} checks)")
    parts.append(f"{s['elements_min']}-{s['elements_max']} elements per check")
    if s["bits"]:
        parts.append("bits " + " ".join(f"{b}:{f:.0%}" for b, f in s["bits"]))
    return "; ".join(parts)


def _runs(values):
    """Split sorted ints into contiguous (start, end) runs."""
    runs = []
    for v in sorted(values):
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return [tuple(r) for r in runs]


def _span(name, values, grid_len):
    runs = _runs(values)
    if len(runs) != 1:
        return f"{name}=" + ",".join(f"{a}..{b}" if a != b else f"{a}" for a, b in runs) + " (not contiguous)"
    a, b = runs[0]
    edges = []
    if grid_len is not None:
        if a == 0:
            edges.append(f"{name}=0")
        if b == grid_len - 1:
            edges.append(f"{name}={grid_len - 1}")
    reach = f", reaches edge {' and '.join(edges)}" if edges else ""
    return f"{name}={a}..{b}{reach}"


def classify(cores, grid=None):
    """One-line description of the failing-core pattern.

    cores: iterable of (x, y). grid: optional (grid_x, grid_y) so runs can be reported as reaching an edge.
    Patterns: none, single core, row run, column run, rectangle, whole grid, scattered.
    """
    cores = set(cores)
    gx, gy = grid if grid else (None, None)
    if not cores:
        return "none"
    xs, ys = {x for x, _ in cores}, {y for _, y in cores}
    n = len(cores)
    if n == 1:
        ((x, y),) = cores
        return f"single core ({x},{y})"
    if gx and gy and n == gx * gy:
        return f"whole grid ({n} cores)"
    if len(ys) == 1:
        return f"row y={next(iter(ys))}: {_span('x', xs, gx)} ({n} cores)"
    if len(xs) == 1:
        return f"column x={next(iter(xs))}: {_span('y', ys, gy)} ({n} cores)"
    rect = {(x, y) for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)}
    if cores == rect:
        return f"rectangle x={min(xs)}..{max(xs)} y={min(ys)}..{max(ys)} ({n} cores)"
    listed = " ".join(f"({x},{y})" for x, y in sorted(cores, key=lambda c: (c[1], c[0]))[:12])
    more = f" +{n - 12} more" if n > 12 else ""
    return f"scattered ({n} cores in {len(ys)} rows, {len(xs)} columns): {listed}{more}"


# --- Per-tile owner tables ------------------------------------------------------------------------------------------
#
# For outputs whose core assignment is not one rectangular block per core (SDPA's (batch, head, q-chunk) split,
# contiguous tile ranges that wrap across rows, sharded and sliced conv outputs), the mapping is an explicit table:
#
#   owner[tr, tc] = index into `cores` of the core that computed and wrote output tile (tr, tc), or -1 for none
#
# over the output's physical 2D tile grid: leading dims are flattened (not summed), each leading slice is padded to
# whole tiles, so row tile tr covers physical rows [32 tr, 32 tr + 32) of shape (prod(leading) * pad32(H), pad32(W)).
# A table is built by one of the owner_* functions below from the op's own work split, read from its program factory.


def _pad(n, m=TILE):
    return -(-n // m) * m


def physical_tile_grid(shape):
    """(row tiles, col tiles, leading, padded H, padded W) of the physical 2D tile grid of a logical shape."""
    shape = list(shape)
    lead = 1
    for d in shape[:-2]:
        lead *= d
    h, w = (shape[-2], shape[-1]) if len(shape) >= 2 else (1, shape[-1])
    hp, wp = _pad(h), _pad(w)
    return lead * hp // TILE, wp // TILE, lead, hp, wp


def tile_mismatches(ref, out):
    """int64 [row tiles, col tiles]: mismatching elements per physical output tile (logical padding never counts).
    Works one leading slice and one tile row at a time, so no full-size integer copy of the output is made."""
    diff = ref != out
    h, w = diff.shape[-2], diff.shape[-1]
    diff = diff.reshape(-1, h, w)
    rows_t, cols_t, lead, hp, wp = physical_tile_grid(ref.shape)
    counts = torch.zeros(rows_t, cols_t, dtype=torch.int64)
    tiles_per_slice = hp // TILE
    for l in range(lead):
        for t in range(tiles_per_slice):
            band = diff[l, t * TILE : (t + 1) * TILE]
            if band.numel() == 0:
                continue
            per_col = band.sum(dim=0, dtype=torch.int32)
            if wp != w:
                per_col = torch.nn.functional.pad(per_col, (0, wp - w))
            counts[l * tiles_per_slice + t] = per_col.reshape(cols_t, TILE).sum(dim=1)
    return counts


def owner_core_mismatches(ref, out, owner, cores):
    """({(x, y): mismatching elements}, elements in tiles no core owns). owner/cores come from an owner_* builder."""
    counts = tile_mismatches(ref, out)
    if counts.shape != owner.shape:
        raise ValueError(
            f"owner table {tuple(owner.shape)} does not match the output's tile grid {tuple(counts.shape)}"
        )
    owned = owner >= 0
    per_core = torch.bincount(owner[owned], weights=counts[owned].double(), minlength=len(cores))
    result = {cores[i]: int(n) for i, n in enumerate(per_core.tolist()) if n}
    return result, int(counts[~owned].sum())


def contiguous_owner_index(unit, total, num_cores):
    """Index of the core that owns `unit` (tensor of ints) when `total` units are split into contiguous runs over
    `num_cores` cores and the first total % cores cores take one more (tt_metal/common/work_split.cpp
    split_work_to_cores_counts; same rule as SDPA's global_q_range for non-causal attention)."""
    cores_used = min(total, num_cores)
    base, extra = total // cores_used, total % cores_used
    big = extra * (base + 1)
    return torch.where(unit < big, unit // (base + 1), extra + (unit - big) // max(base, 1))


def row_major_cores(grid_x, grid_y, origin=(0, 0)):
    """Cores of a grid in row-major order (y outer, x inner), as tt-metal's grid_to_cores(row_wise=true)."""
    return [(origin[0] + i % grid_x, origin[1] + i // grid_x) for i in range(grid_x * grid_y)]


def owner_contiguous_tiles(shape, grid_x, grid_y):
    """Interleaved eltwise (binary_ng, no sharding): output tiles in row-major page order t = tr * Wt + tc are split
    into contiguous runs over the grid's cores in row-major order (binary_ng_program_factory.cpp: row_major = true,
    interleaved_split, c_start_id += tiles of the previous core)."""
    rows_t, cols_t, *_ = physical_tile_grid(shape)
    t = torch.arange(rows_t * cols_t, dtype=torch.int64).reshape(rows_t, cols_t)
    cores = row_major_cores(grid_x, grid_y)
    return contiguous_owner_index(t, rows_t * cols_t, len(cores)), cores


def owner_sdpa(shape, q_chunk_size, grid_x, grid_y):
    """Non-causal SDPA / flash MLA prefill, output (B, NQH, Sq, Dv): the flat (b, h, q-chunk) space (b-major) is split
    into contiguous runs over the grid's cores in row-major order, the first total % cores taking one more chunk
    (sdpa_program_factory.cpp global_q_range, core = (i % grid_x, i // grid_x)). Causal attention (zigzag, pair
    distribution) maps differently and is not handled here."""
    b, nqh, sq = shape[-4] if len(shape) >= 4 else 1, shape[-3], shape[-2]
    rows_t, cols_t, lead, hp, _ = physical_tile_grid(shape)
    if q_chunk_size % TILE:
        raise ValueError(f"q_chunk_size {q_chunk_size} is not a whole number of tiles")
    qn = _pad(sq, q_chunk_size) // q_chunk_size
    tr = torch.arange(rows_t, dtype=torch.int64)
    lead_idx, row_in_slice = tr // (hp // TILE), (tr % (hp // TILE)) * TILE
    g = lead_idx * qn + row_in_slice // q_chunk_size
    cores = row_major_cores(grid_x, grid_y)
    core_idx = contiguous_owner_index(g, b * nqh * qn, len(cores))
    return core_idx.unsqueeze(1).expand(rows_t, cols_t).contiguous(), cores


def shard_owner_index(rows, cols, layout, shard_shape, grid, orientation_row_major):
    """Core index (into the returned core list) holding element (rows, cols) of a sharded tensor's physical 2D view.

    layout: "HEIGHT" / "WIDTH" / "BLOCK"; shard_shape: (h, w) elements; grid: list of ((x0, y0), (x1, y1)) ranges in
    the shard spec's order. HEIGHT/WIDTH shard i lives on the i-th core of corerange_to_cores(grid, row_major)
    (tt_metal/impl/buffers/buffer.cpp); BLOCK shard (r, c) on (x0 + c, y0 + r) for ROW_MAJOR, (x0 + r, y0 + c) for
    COL_MAJOR (one rectangle)."""
    sh, sw = shard_shape
    cores = []
    for (x0, y0), (x1, y1) in grid:
        if orientation_row_major:
            cores += [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
        else:
            cores += [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    if layout == "HEIGHT":
        idx = rows // sh
    elif layout == "WIDTH":
        idx = cols // sw
    elif layout == "BLOCK":
        if len(grid) != 1:
            raise ValueError("block sharding over more than one core range")
        (x0, y0), (x1, y1) = grid[0]
        r, c = rows // sh, cols // sw
        x, y = (x0 + c, y0 + r) if orientation_row_major else (x0 + r, y0 + c)
        inside = (x <= x1) & (y <= y1)
        # position of (x, y) in `cores`, which lists the rectangle in the same orientation
        idx = (y - y0) * (x1 - x0 + 1) + (x - x0) if orientation_row_major else (x - x0) * (y1 - y0 + 1) + (y - y0)
        idx = torch.where(inside, idx, torch.full_like(idx, len(cores)))
    else:
        raise ValueError(f"unknown shard layout {layout}")
    idx = torch.where(idx < len(cores), idx, torch.full_like(idx, -1))
    return idx, cores


def owner_sharded(shape, layout, shard_shape, grid, orientation_row_major):
    """Sharded output whose shards are computed and written by the core holding them (true for matmul and conv2d
    sharded outputs). The table is in whole tiles, so shard edges must be tile aligned."""
    if shard_shape[0] % TILE or shard_shape[1] % TILE:
        raise ValueError(f"shard shape {shard_shape} is not tile aligned")
    rows_t, cols_t, *_ = physical_tile_grid(shape)
    rows = (torch.arange(rows_t, dtype=torch.int64) * TILE).unsqueeze(1).expand(rows_t, cols_t)
    cols = (torch.arange(cols_t, dtype=torch.int64) * TILE).unsqueeze(0).expand(rows_t, cols_t)
    idx, cores = shard_owner_index(rows, cols, layout, shard_shape, grid, orientation_row_major)
    return idx.contiguous(), cores


def owner_conv_slices(shape, out_h, out_w, slices):
    """conv2d run in DRAM slices (op_slicing.cpp): each slice computes the output box n in [n0, n1), h in [h0, h1),
    w in [w0, w1) into its own sharded L1 tensor (local row = ((n - n0) * (h1 - h0) + (h - h0)) * (w1 - w0) + (w - w0)),
    which slice_write copies from the same cores into the DRAM output.
    `slices` = [(n0, n1, h0, h1, w0, w1, layout, shard_shape, grid, row_major)], as recorded from that call's graph
    capture. The output's rows are (n * out_h + h) * out_w + w.

    The table is in whole tiles, so every tile row must lie in one slice and one shard: out_w, slice widths, width
    offsets and shard shapes must be multiples of 32. Otherwise this raises rather than credit a whole tile to the
    core holding its first row."""
    if out_w % TILE:
        raise ValueError(f"output width {out_w} is not a multiple of {TILE}; a tile row would span two image rows")
    for n0, n1, h0, h1, w0, w1, layout, shard_shape, *_ in slices:
        if w0 % TILE or (w1 - w0) % TILE:
            raise ValueError(f"slice w {w0}-{w1} is not tile aligned; a tile row would span two slices")
        if shard_shape[0] % TILE or shard_shape[1] % TILE:
            raise ValueError(f"slice shard shape {shard_shape} is not tile aligned")
    rows_t, cols_t, *_ = physical_tile_grid(shape)
    r = torch.arange(rows_t, dtype=torch.int64) * TILE
    image = out_h * out_w
    n, rem = r // image, r % image
    h, w = rem // out_w, rem % out_w
    cols = (torch.arange(cols_t, dtype=torch.int64) * TILE).unsqueeze(0).expand(rows_t, cols_t)
    owner = torch.full((rows_t, cols_t), -1, dtype=torch.int64)
    index = {}
    for n0, n1, h0, h1, w0, w1, layout, shard_shape, grid, row_major in slices:
        inside = (n >= n0) & (n < n1) & (h >= h0) & (h < h1) & (w >= w0) & (w < w1)
        if not inside.any():
            continue
        local = (((n - n0) * (h1 - h0) + (h - h0)) * (w1 - w0) + (w - w0)).unsqueeze(1).expand(rows_t, cols_t)
        idx, cores = shard_owner_index(local, cols, layout, shard_shape, grid, row_major)
        remap = torch.tensor([index.setdefault(c, len(index)) for c in cores] + [-1], dtype=torch.int64)
        mask = inside.unsqueeze(1).expand(rows_t, cols_t)
        owner[mask] = remap[idx[mask]]
    all_cores = [None] * len(index)
    for c, i in index.items():
        all_cores[i] = c
    return owner, all_cores


_SHARD_RE = re.compile(
    r"memory_layout=TensorMemoryLayout::(HEIGHT|WIDTH|BLOCK)_SHARDED[^)]*?shard_spec=ShardSpec\{grid=(\[.*?\]), "
    r"shape=\[(\d+), (\d+)\], orientation=ShardOrientation::(ROW_MAJOR|COL_MAJOR)\}"
)
# One core range as the capture prints it: {"start":{"x":0,"y":0},"end":{"x":7,"y":9}. The printed list is not
# valid JSON (a closing brace short per range), so the ranges are read with this pattern instead of a JSON parser.
_RANGE_RE = re.compile(r'"start":\{"x":(\d+),"y":(\d+)\},"end":\{"x":(\d+),"y":(\d+)\}')
_SHAPE_RE = re.compile(r"Shape\(\[([0-9, ]+)\]\)")


def _shard_from_text(text):
    """(layout, (shard h, shard w), [((x0, y0), (x1, y1))], row_major) of the first sharded memory config in text."""
    m = _SHARD_RE.search(text)
    if m is None:
        raise ValueError("no sharded memory config in the captured arguments")
    grid = [((int(a), int(b)), (int(c), int(d))) for a, b, c, d in _RANGE_RE.findall(m.group(2))]
    if not grid:
        raise ValueError(f"no core range in shard grid {m.group(2)!r}")
    return m.group(1), (int(m.group(3)), int(m.group(4))), grid, m.group(5) == "ROW_MAJOR"


def conv_slices_from_graph(graph):
    """Slices of one conv2d call from its RunMode.NORMAL graph capture, in run order:
    [(n0, n1, h0, h1, w0, w1, layout, shard_shape, grid, row_major)] with n/h/w the NHWC output range each slice writes
    (SliceWriteDeviceOperation begins/ends) and the shard spec of that slice's conv output (slice_write's input,
    written from the cores holding it). A conv run in one piece and moved to DRAM by ShardedToInterleaved after the
    conv gives one entry with n0 = ... = w1 = None (the whole output). An empty list when neither appears."""
    starts = sorted((n for n in graph if n.get("node_type") == "function_start"), key=lambda n: n.get("counter", 0))
    slices, after_conv = [], False
    for n in starts:
        name = (n.get("params") or {}).get("name", "")
        args = [str(a) for a in (n.get("arguments") or [])]
        if name == "Conv2dDeviceOperation":
            after_conv = True
        elif name == "SliceWriteDeviceOperation" and len(args) >= 2:
            begins, ends = ([int(v) for v in m.split(",")] for m in _SHAPE_RE.findall(args[0])[:2])
            layout, shard, grid, row_major = _shard_from_text(args[1])
            slices.append((begins[0], ends[0], begins[1], ends[1], begins[2], ends[2], layout, shard, grid, row_major))
        elif name == "ShardedToInterleavedDeviceOperation" and after_conv and len(args) >= 2 and not slices:
            layout, shard, grid, row_major = _shard_from_text(args[1])
            slices.append((None, None, None, None, None, None, layout, shard, grid, row_major))
    return slices
