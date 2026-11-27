# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Run a matmul on a window of its full core grid, each core doing the work it does on the full grid.

For core-exclusion checks (does the workload pass with a suspect core left out?), shrinking the grid of a fixed-shape
matmul gives every remaining core a bigger block: the per-core work changes and the circular buffers can outgrow L1.
Instead, keep the full grid's program config (per_core_M / per_core_N, subblocks, in0_block_w, the whole K) and shrink
the problem to the output blocks the window's cores own on the full grid. Every core in the window then computes the
same block size with the same buffers as on the full grid. What still differs: shorter multicast chains, a bigger
share of DRAM bandwidth per core, and smaller (different) input data.

Block ownership is the matmul factories' (matmul_multicore_reuse_mcast_{2d,1d}_program_factory.cpp):
  2D: core (x, y) of the grid owns output block row y and block column x (swapped with transpose_mcast).
  1D: core i, row-major over the grid, owns block i: a column block with mcast_in0, a row block with mcast_in1.
Only the last block of a row / column may be partial, which a contiguous window keeps last.
"""

import ttnn

TILE = 32


class NotWindowable(ValueError):
    """The config cannot run on a window with the full grid's per-core work."""


class NoBlocksInWindow(NotWindowable):
    """The window's cores own no output blocks on the full grid (they are idle there too): nothing to verify."""


def config_grid(config):
    """(origin, size) of the grid a matmul factory actually uses: the bounding box of allowed_worker_cores when set
    (both mcast factories place and size the grid from it), else (0, 0) and compute_with_storage_grid_size."""
    allowed = getattr(config, "allowed_worker_cores", None)
    if allowed is not None and allowed.ranges():
        rs = allowed.ranges()
        x0, y0 = min(r.start.x for r in rs), min(r.start.y for r in rs)
        x1, y1 = max(r.end.x for r in rs), max(r.end.y for r in rs)
        return (x0, y0), (x1 - x0 + 1, y1 - y0 + 1)
    g = config.compute_with_storage_grid_size
    return (0, 0), (g.x, g.y)


def _blocks(total_tiles, per_core):
    """Block sizes in tiles along one dimension: full per_core blocks, the last one partial."""
    n = -(-total_tiles // per_core)
    return [min(per_core, total_tiles - i * per_core) for i in range(n)]


def _window_crs(origin, size):
    (ox, oy), (wx, wy) = origin, size
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(ox, oy), ttnn.CoreCoord(ox + wx - 1, oy + wy - 1))])


def window_matmul(config, M, N, window_size, window_origin, batch=1):
    """(M', N', windowed config) for running `config` (a matmul program config for an M x N output on its full
    grid, placed at (0, 0)) on the window [origin, origin + size) of that grid with the full grid's per-core work.

    The output of the windowed run is the full output restricted to the window's blocks, so output block (by, bx) of
    the windowed run is computed by window core (origin.x + bx, origin.y + by) (2D), or window core j =
    (origin.x + j % size.x, origin.y + j // size.x) owns windowed block j (1D)."""
    name = type(config).__name__
    start, (fx, fy) = config_grid(config)
    (wx, wy), (ox, oy) = window_size, window_origin
    if start != (0, 0):
        raise NotWindowable(f"the full config is already placed at {start}; window it from a (0, 0) config")
    if ox + wx > fx or oy + wy > fy or wx < 1 or wy < 1:
        raise NotWindowable(f"window {wx}x{wy} at {ox}x{oy} is not inside the config grid {fx}x{fy}")
    if batch > 1 and getattr(config, "fuse_batch", False):
        raise NotWindowable("fuse_batch with batch > 1 folds batches into the row blocks; not windowed")
    Mt, Nt = -(-M // TILE), -(-N // TILE)
    pcM, pcN = config.per_core_M, config.per_core_N

    if name == "MatmulMultiCoreReuseMultiCastProgramConfig":
        rows, cols = _blocks(Mt, pcM), _blocks(Nt, pcN)
        # in0 (rows) follows core.y, in1 (columns) core.x; transpose_mcast swaps them
        row_range, col_range = (
            (range(ox, ox + wx), range(oy, oy + wy))
            if config.transpose_mcast
            else (
                range(oy, oy + wy),
                range(ox, ox + wx),
            )
        )
        rows_in = [rows[i] for i in row_range if i < len(rows)]
        cols_in = [cols[i] for i in col_range if i < len(cols)]
        if not rows_in or not cols_in:
            raise NoBlocksInWindow(f"window {wx}x{wy} at {ox}x{oy} holds no output blocks of this matmul")
        windowed = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(wx, wy),
            in0_block_w=config.in0_block_w,
            out_subblock_h=config.out_subblock_h,
            out_subblock_w=config.out_subblock_w,
            out_block_h=config.out_block_h,
            out_block_w=config.out_block_w,
            per_core_M=pcM,
            per_core_N=pcN,
            transpose_mcast=config.transpose_mcast,
            fused_activation=config.fused_activation,
            fuse_batch=config.fuse_batch,
            allowed_worker_cores=_window_crs((ox, oy), (wx, wy)),
        )
        return TILE * sum(rows_in), TILE * sum(cols_in), windowed

    if name == "MatmulMultiCoreReuseMultiCast1DProgramConfig":
        hop = getattr(config, "hop_cores", None)
        if config.gather_in0 or (hop is not None and hop.ranges()) or config.num_global_cb_receivers > 1:
            raise NotWindowable(f"{name} with gather_in0, hop cores or global CB receivers is not windowed")
        # full-grid core index of each window core, in the window's own row-major order (monotonic, so a partial
        # last block stays last)
        full_index = [y * fx + x for y in range(oy, oy + wy) for x in range(ox, ox + wx)]
        blocks = _blocks(Nt, pcN) if config.mcast_in0 else _blocks(Mt, pcM)
        owned = [blocks[i] for i in full_index if i < len(blocks)]
        if not owned:
            raise NoBlocksInWindow(f"window {wx}x{wy} at {ox}x{oy} holds no output blocks of this matmul")
        windowed = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(wx, wy),
            in0_block_w=config.in0_block_w,
            out_subblock_h=config.out_subblock_h,
            out_subblock_w=config.out_subblock_w,
            out_block_h=config.out_block_h,
            out_block_w=config.out_block_w,
            per_core_M=pcM,
            per_core_N=pcN,
            fuse_batch=config.fuse_batch,
            fused_activation=config.fused_activation,
            mcast_in0=config.mcast_in0,
            gather_in0=False,
            untilize_out=config.untilize_out,
            num_global_cb_receivers=config.num_global_cb_receivers,
            stream_in1=config.stream_in1,
            allowed_worker_cores=_window_crs((ox, oy), (wx, wy)),
        )
        if config.mcast_in0:
            return M, TILE * sum(owned), windowed
        return TILE * sum(owned), N, windowed

    raise NotWindowable(f"matmul program config {name} cannot be windowed")
