# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""E1 bench: make_plan with a pluggable bf16-X group-width selector (host-only experiment).

`make_plan_with(select)` returns a drop-in replacement for pd.make_plan. It is the real make_plan with the
NARROW_GROUPS block replaced by `select(fits, ctx)`, where `fits` = the L1-fitting group_h = 1 candidates for
every group_w in min(grid_x, Ct)..1 (the same `fit` as the op) and ctx the plan quantities. Everything else
(the fallback full-row / group_h growth path, the Plan fields) is identical to the op's.

Selectors:
  sel_default        the op's current rule (fewest blocks, widest among those)
  sel_forced(w, d)   force group_w (and optionally the X block depth); falls back to default if it does not fit
  sel_candidate      the E1 rule (see pick_pipeline_aware)
"""

import math

import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd


def _fits(device, x_tensor, w_tensor, n, depths=None):
    padded = list(x_tensor.padded_shape)
    lead = 1
    for d in padded[:-2]:
        lead *= d
    Mt = lead * (padded[-2] // pd.TILE)
    C = x_tensor.shape[-1] // n
    Ct = C // pd.TILE
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y
    x_tile = x_tensor.buffer_page_size()
    w_tile = w_tensor.buffer_page_size()
    dtypes = dict(x_dtype=x_tensor.dtype, w_dtype=w_tensor.dtype, y_dtype=x_tensor.dtype)
    budget = ttnn_budget()
    depths = depths or (pd.X_BLOCK_DEPTH_DEFAULT, 1)

    def fit(group_w, group_h, depths=depths):
        group_cores = group_w * group_h
        groups_x, groups_y = grid_x // group_w, grid_y // group_h
        core_token_tiles, t_start = pd._split(Mt, groups_x * groups_y)
        ctt_max = max(core_token_tiles)
        c_tiles, c_starts = pd._c_split(Ct, group_cores, owner_fixed=ctt_max <= 1)
        kmax = n * max(c_tiles)
        y_chunk = min(max(c_tiles), pd.Y_CHUNK_TILES_CAP)

        def l1_at(bt_, depth_):
            return pd._l1_bytes(
                bt=bt_,
                depth=depth_,
                kmax=kmax,
                G=group_cores,
                y_chunk=y_chunk,
                n=n,
                x_tile=x_tile,
                w_tile=w_tile,
                y_tile=x_tile,
                **dtypes,
            )

        bt_cap = min(ctt_max, pd.BLOCK_TOKEN_TILES_CAP)
        for depth in depths:
            fixed = l1_at(0, depth)
            per_bt = l1_at(1, depth) - fixed
            bt = min(bt_cap, (budget - fixed) // per_bt) if budget > fixed else 0
            if bt >= 1:
                return dict(
                    group_w=group_w,
                    group_h=group_h,
                    group_cores=group_cores,
                    kmax=kmax,
                    y_chunk=y_chunk,
                    groups_x=groups_x,
                    groups_y=groups_y,
                    core_token_tiles=core_token_tiles,
                    t_start=t_start,
                    c_tiles=c_tiles,
                    c_starts=c_starts,
                    bt=bt,
                    depth=depth,
                    blocks=math.ceil(ctt_max / bt),
                )
        return None

    ctx = dict(
        Mt=Mt,
        C=C,
        Ct=Ct,
        n=n,
        grid_x=grid_x,
        grid_y=grid_y,
        x_tile=x_tile,
        w_tile=w_tile,
        x_dtype=x_tensor.dtype,
        w_dtype=w_tensor.dtype,
        budget=budget,
    )
    return fit, ctx


def ttnn_budget():
    import ttnn

    return ttnn.get_max_worker_l1_unreserved_size() - pd.L1_SAFETY_MARGIN


def make_plan_with(select, depths=None):
    def make_plan(device, x_tensor, w_tensor, n):
        fit, ctx = _fits(device, x_tensor, w_tensor, n, depths)
        Mt, Ct, grid_x, grid_y = ctx["Mt"], ctx["Ct"], ctx["grid_x"], ctx["grid_y"]
        chosen = None
        if Mt >= grid_y and pd.NARROW_GROUPS and pd.x_pieces(x_tensor.dtype) == 1:
            fits = [f for f in (fit(w, 1) for w in range(min(grid_x, Ct), 0, -1)) if f is not None]
            chosen = select(fits, ctx, fit) if fits else None
        if chosen is None:
            group_w = min(grid_x, Ct)
            group_h = 1 if Mt >= grid_y else max(1, min(grid_y // Mt, Ct // group_w, pd.GROUP_CORES_CAP // group_w))
            while True:
                chosen = fit(group_w, group_h)
                if chosen is not None:
                    break
                if (
                    group_h * 2 <= grid_y
                    and group_w * group_h * 2 <= pd.GROUP_CORES_CAP
                    and group_w * group_h * 2 <= Ct
                ):
                    group_h *= 2
                    continue
                raise RuntimeError("mhc_pre (gw bench): no blocking fits L1")
        make_plan.last = chosen
        return pd.Plan(
            n=n,
            Mt=Mt,
            Ct=Ct,
            Kt=n * Ct,
            grid_x=grid_x,
            grid_y=grid_y,
            group_w=chosen["group_w"],
            group_h=chosen["group_h"],
            group_cores=chosen["group_cores"],
            groups_x=chosen["groups_x"],
            groups_y=chosen["groups_y"],
            num_groups=chosen["groups_x"] * chosen["groups_y"],
            core_c_tiles=chosen["c_tiles"],
            c_start=chosen["c_starts"],
            core_token_tiles=chosen["core_token_tiles"],
            t_start=chosen["t_start"],
            core_k_tiles_max=chosen["kmax"],
            block_token_tiles=chosen["bt"],
            x_block_depth=chosen["depth"],
            y_chunk_tiles=chosen["y_chunk"],
            y_depth=pd.Y_DEPTH,
        )

    make_plan.last = None
    return make_plan


def sel_default(fits, ctx, fit):
    chosen = None
    for f in fits:  # widest first
        if chosen is None or f["blocks"] < chosen["blocks"]:
            chosen = f
    return chosen


def sel_forced(group_w, depth=None):
    def select(fits, ctx, fit):
        if depth is not None:
            f = fit(group_w, 1, depths=(depth,))
            return f  # None -> fallback (full row), reported as such
        for f in fits:
            if f["group_w"] == group_w:
                return f
        return None

    return select


def pipelined(f):
    """Whether every block's tail is covered by the cross-block pipeline (the kernels' pipe_at rule):
    pipe_at(b) = b+1 < blocks and (depth >= 3 or b + depth >= blocks); only the last step's tail is exposed."""
    B, d = f["blocks"], f["depth"]
    return all((b + 1 < B and (d >= 3 or b + d >= B)) for b in range(B - 1))


def sel_candidate(fits, ctx, fit):
    return pick_pipeline_aware(fits, ctx)


# E1 cost-model constants (units: one X tile read by one core at full-grid DRAM contention, ~0.55 us on BH p150)
RT_TILES_BASE = 32  # per-block round trip after the rank's last X tile: partial send -> gather -> mcast -> coef/pre,
# plus the owner's Sinkhorn (zones 640x1792: ~7 us Sinkhorn + ~8 us gather/mcast/coef)
RT_TILES_UNSTREAMED_PROJ = 4  # bf16 W: the projection is one whole-block matmul_block window (not streamed under
# the X read), so the block's last projection adds to its round trip
RT_TILES_PER_RANK = 0.75  # the root's rank-ordered fold grows with group_cores (zones: ~0.45 us per partial)
TAIL_FRAC = 0.4  # a block's own tail (coef + y-mix + y write) per K tile, relative to its X read
SINKHORN_TILES = 12  # the owner's Sinkhorn (zones: c_owned ~6.7 us); off the critical path when every group holds
# one token tile-row (owner_fixed: OWNER_C_DISCOUNT moves C tiles off the owner to pay for it)


def block_schedule_cost(f, n, streamed_proj=True):
    """Critical-path estimate of one plan in X-tile-read units (the E1 selection function).

    stream        B * kmax              the rank's X blocks cross the NoC back to back (prefetch, depth >= 2)
    last block    H(G) + kmax / n       its round trip + y-mix / y write (y is 1/n of X) are always exposed
    step b < B-1  pipelined (pipe_at):  max(0, H - TAIL_FRAC*kmax)      round trip of b+1 hides under tail(b)
                  serial:               max(0, H - (1-TAIL_FRAC)*kmax)  round trip of b hides under the reader's
                                                                        prefetch of X(b+1) minus tail(b)
    H(G) = RT_TILES_BASE + RT_TILES_PER_RANK * group_cores; one-row groups (owner discount) drop SINKHORN_TILES
    from the last block's exposure.
    """
    B, d, k, G = f["blocks"], f["depth"], f["kmax"], f["group_cores"]
    H = RT_TILES_BASE + RT_TILES_PER_RANK * G + (0 if streamed_proj else RT_TILES_UNSTREAMED_PROJ)
    cost = B * k + H + k / n - (SINKHORN_TILES if max(f["core_token_tiles"]) <= 1 else 0)
    for b in range(B - 1):
        pipe = b + 1 < B and (d >= 3 or b + d >= B)
        cost += max(0.0, H - TAIL_FRAC * k) if pipe else max(0.0, H - (1.0 - TAIL_FRAC) * k)
    return cost


def pick_pipeline_aware(fits, ctx):
    """Lexicographic: (1) a plan that can prefetch (depth >= 2, or a single block) beats one that cannot -- at
    depth 1 the X read of block b+1 waits for block b's whole tail, so nothing overlaps; (2) the lowest
    block_schedule_cost; (3) the wider group on a tie (the fits are widest first, min() keeps the first)."""
    streamed = pd.w_pieces(ctx["w_dtype"]) > 1  # bf16 X + fp32 W: the streamed projection (X_STREAM_CHUNKS)
    return min(fits, key=lambda f: (f["depth"] < 2 and f["blocks"] > 1, block_schedule_cost(f, ctx["n"], streamed)))
