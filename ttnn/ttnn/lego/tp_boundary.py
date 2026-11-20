# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused tensor-parallel layer boundary, composed from Lego blocks:

    out_proj (row-parallel matmul) -> reduce-scatter + residual add -> all-gather -> RMSNorm -> next matmul

one program per device (ttnn.generic_op), replacing the five or six ops of the composed chain. Per device:

  matmul cores (`mm_origin`, `mm_grid`; LoFi like the model's matmuls)
      MM1   x_in @ W1 -> P                      (L1)
      MM2   H @ W2 -> out, each output row times its RMSNorm scale (epilogue), optional SwiGLU
  aux cores (the rows above; HiFi4 + fp32 accumulation like the model's norm)
      RS    reduce-scatter P (+ residual) -> h  (the new residual stream), also into this device's slot of H
      AG    all-gather h -> H                   (L1), signalling MM2's in0 injectors row group by row group
      stats prefill: sum(h**2) over this device's slice -> all-gather of the partial sums
            decode:  sum(H**2) per slice once the AG is done (no second collective)
            -> scale = 1/sqrt(E[x**2] + eps) per row

The norm is applied as a row scale after MM2 ((r * x) @ W = r * (x @ W)), so MM2 starts on H's slices as they arrive
(K in arrival order) and the statistics run beside it. Stages hand off through sync blocks (all producers' writes
land before consumers start). prefill_config / decode_config hold the tuned placements; README.md describes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import ttnn

from .ccl import GlobalSems, ag_row_group_waits, ccl_layout, place_all_gather, place_mux, place_reduce_scatter, ring_of
from .compose import Block, LegoProgram, mesh_descriptor
from .mm import MatmulConfig, matmul_cb_bytes, place_matmul
from .norm import plan_parts, place_rms_scale, place_rms_stats
from .sync import KDIR as _K, place_handoff_multi, place_local_handoff

SYNC = f"{_K}/sync"

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


@dataclass
class BoundaryConfig:
    mm1: MatmulConfig
    mm2: MatmulConfig
    mm_origin: tuple = (0, 2)
    mm_grid: tuple = (12, 8)
    aux_rows: tuple = (0, 1)
    rs_workers: int = 5  # per (link, direction); > 1 uses a fabric mux (5 fills the 24 aux cores)
    ag_workers: int = 2
    num_links: int = 2
    swiglu: bool = False
    out_mem: object = field(default_factory=lambda: DRAM)
    # compute config of the aux cores (RS reduction, norm statistics): the model's norm runs HiFi4 + fp32
    aux_fidelity: object = field(default_factory=lambda: ttnn.MathFidelity.HiFi4)
    aux_fp32: bool = True
    # L1 arena on the matmul cores: MM1's CBs, then (once MM1 is done) MM2's weight CB over the same bytes, as deep as
    # max(MM1's CBs, arena_cap) allows, so the weights stream in while the collectives run
    arena_cap: int = 520 * 1024
    # MM2 weight CB depth limit (K blocks). Deep prefetch forwards weights down the matmul columns while the
    # collectives run: a win when the weights dominate (short S), NoC contention for the collectives otherwise.
    mm2_in1_depth: int = 2
    # run the statistics all-gather through the AG's muxes (alongside it) instead of after it
    stats_on_ag_mux: bool = False
    # MM1 -> RS pipelining: MM1 computes the rows batch by batch, RS reduces each batch as soon as MM1 has written it
    # (mm1.M_block must then be a core's rows per batch)
    batches: int = 1
    # Each matmul is split by N over mmX_split grids of mm_grid stacked below mm_origin: more cores stream its weights
    # from DRAM in parallel (short S, where the weights dominate)
    mm1_split: int = 1
    mm2_split: int = 1
    # MM1's grid when it differs from MM2's (mm_grid), e.g. flatter grids to split MM1 more ways
    mm1_grid: tuple | None = None
    # MM2's weight CB reuses MM1's arena bytes once MM1 is done (True), or has its own L1 after them and holds all of
    # MM2's weights, so they stream in from the start while MM1 and the collectives run (decode)
    mm2_overlap: bool = True
    # first row of MM2's grids, relative to mm_origin (> 0 puts MM2 on cores MM1 does not use, so its weight reads
    # start at once instead of after MM1's on the same RISC)
    mm2_row: int = 0
    # reduce-scatter chunks between flow-control syncs (None: the op's default)
    rs_chunks_per_sync: int | None = None
    # statistics from the gathered H once the all-gather is done (each device sums every slice's squares itself)
    # instead of from the local slice plus an all-gather of the partial sums: one collective less, but they start
    # after the all-gather (decode, where collective latency dominates)
    stats_from_gathered: bool = False
    # MM2 transposed (mm2_split 1 only): M split over the grid's columns and N over its rows, in0 forwarded down the
    # columns and the weights along the rows. Pads less when the rows split N more evenly (A's gate_up on an 11 x 8
    # grid: 3 x 28 tiles a core instead of 4 x 22)
    mm2_transpose: bool = False
    # the AG readers also signal MM2's injectors while they wait to forward a slice
    # (place_all_gather(signal_while_forwarding=...)): MM2 hears of a neighbour's rows as they land
    ag_early_signal: bool = False
    # MM1 computes its output quarter by quarter in the order the reduce-scatter's ring steps read them (the slice of
    # the device across the ring, both neighbours', its own) and signals each, so the RS starts during MM1
    # (place_matmul(n_groups=...)); mm1_split 1, batches 1. mm1_transpose: MM1 with M over the columns, N over the rows
    mm1_groups: bool = False
    # MM1 split by N (mm1_split a multiple of the ring size): the splits cover whole RS input quarters, and the bottom
    # ones (whose DRAM reads finish first) the quarters the RS reads first; each signals its quarter, so the RS starts
    # during MM1
    mm1_quarter_splits: bool = False
    # the matmuls' weight injectors read the next K block while forwarding the current one (place_matmul)
    mm1_read_ahead: bool = False
    # SwiGLU MM2: read the weights in same-bank page pairs (place_matmul(in1_bank_pairs=...): small bfp4 pages)
    mm2_bank_pairs: bool = False
    # MM2's weight readers start once every MM1 core has written its output (MM1 reads DRAM alone first: it is on the
    # critical path, MM2's weights stream in during the collectives)
    mm2_after_mm1: bool = False
    mm2_read_ahead: bool = False
    # MM1's in0 held for all of K and reused by its quarters (with mm1_groups) instead of streamed once per quarter
    mm1_resident: bool = False
    mm1_transpose: bool = False


def _split_cfg(cfg: MatmulConfig, n_tiles: int, split: int, gx: int, swiglu: bool) -> MatmulConfig:
    """cfg with N_block = one core's N tiles when the matmul's n_tiles weight columns are split over `split` grids
    of gx columns."""
    if split == 1:
        return cfg
    per = 2 * -(-(n_tiles // 2) // (split * gx)) if swiglu else -(-n_tiles // (split * gx))
    sw = max(w for w in range(1, 5) if per % w == 0 and (not swiglu or w % 2 == 0))
    return MatmulConfig(cfg.M_block, cfg.K_block, per, cfg.subblock_h, sw)


def _fit_n_block(cfg: MatmulConfig, n_tiles: int, gx: int, swiglu: bool) -> MatmulConfig:
    """cfg with N_block = one core's N tiles when cfg.N_block does not divide them (place_matmul splits n_tiles over
    gx columns). A remainder block costs a second pass over K, with in0 streamed again: on an 11-wide grid A's gate_up
    gives 22 tiles a core (blocks of 20 + 2) and B's qkv 5 (4 + 1). The subblock is then 1 x the widest divisor of
    the N block that fits DST (even with SwiGLU); one row high measured fastest for these blocks."""
    per = 2 * -(-(n_tiles // 2) // gx) if swiglu else -(-n_tiles // gx)
    if cfg.N_block and per % cfg.N_block == 0:
        return cfg
    sw = max(w for w in range(1, 9) if per % w == 0 and (not swiglu or w % 2 == 0))
    return MatmulConfig(cfg.M_block, cfg.K_block, per, 1, sw)


def decode_config(boundary: str, grid_x: int = 12) -> BoundaryConfig:
    """Decode (one tile of rows): every matmul core reads its own weight columns (grid_x x 1 grids split by N; MM1 on
    all eight, MM2 on the first six (A) or four (B)), MM2's weights prefetched into their own L1, one CCL worker per
    link and direction (direct fabric connections), statistics from the gathered H. A's bfp4 gate_up weights are read
    in same-bank page pairs, and only once every MM1 core is done (MM1, on the critical path, gets DRAM first)."""
    if boundary == "A":  # o_proj (N 128 tiles) -> gate_up SwiGLU (224): 4 columns (one bank-paired quad) a core
        mm1, mm2, s2 = MatmulConfig(1, 16, 2, 1, 2), MatmulConfig(1, 16, 4, 1, 4), 6
    else:  # down (N 128) -> qkv (48)
        mm1, mm2, s2 = MatmulConfig(1, 16, 2, 1, 2), MatmulConfig(1, 16, 1, 1, 1), 4
    return BoundaryConfig(
        mm1=mm1,
        mm2=mm2,
        mm_grid=(grid_x, 1),
        mm1_split=8,
        mm2_split=s2,
        rs_workers=1,
        ag_workers=1,
        swiglu=boundary == "A",
        out_mem=L1,
        mm2_overlap=False,
        stats_from_gathered=True,
        mm2_bank_pairs=boundary == "A",
        mm2_after_mm1=boundary == "A",
    )


def prefill_config(boundary: str, S: int, grid_x: int = 12) -> BoundaryConfig:
    """Prefill. S >= 512: both matmuls on the grid_x x 8 grid, MM2's weights stream through MM1's arena bytes two K
    blocks deep. Shorter S (the weights dominate): MM1 split by N over four grid_x x 2 grids, A's MM2 (7168 weight
    columns) over two grid_x x 4 grids, MM2's weights prefetched as deep as the arena allows.
    Narrower grids (p300c gives fast dispatch 11 columns): the two aux rows hold 4 RS workers per link and direction
    instead of 5 (1, without a mux, at S < 512), and at S >= 512 the AG takes 3 workers (measured faster on 11 columns; 12 columns keep 2) and A's
    MM2 is transposed with its weights four K blocks deep (3 x 28 tiles a core instead of 4 x 22), and hears of the
    AG's rows as they land (ag_early_signal; measured ~2 us slower for B). A's MM1 there (K = 1024: its in0 fits in
    L1) computes its output quarter by quarter in the RS's order with in0 resident, transposed, so the RS overlaps it
    (mm1_groups; B's in0 does not fit, and streaming it once per quarter costs more than the overlap gains)."""
    narrow = grid_x < 12
    rs_workers = min(5, (2 * grid_x) // 4 - 1) if S >= 512 or not narrow else 1
    ccl = dict(rs_workers=rs_workers, ag_workers=3 if narrow and S >= 512 else 2)
    if S >= 512:
        grid, M, depth, s1, s2, g1 = (grid_x, 8), 4, 2, 1, 1, None
    else:
        grid, M, depth, s1, g1 = (grid_x, 4), max(1, S // 32 // 4), 64, 4, (grid_x, 2)
        s2 = 2 if boundary == "A" else 1
    if boundary == "A":  # o_proj -> gate_up (SwiGLU)
        return BoundaryConfig(
            mm1=MatmulConfig(M, 4, 12, min(2, M), 4),
            mm2=MatmulConfig(M, 8, 20, 1, 4),
            mm_grid=grid,
            swiglu=True,
            out_mem=DRAM if S >= 512 else L1,
            mm2_in1_depth=4 if narrow and S >= 512 else depth,
            mm1_split=s1,
            mm2_split=s2,
            mm1_grid=g1,
            mm2_transpose=narrow and S >= 512,
            ag_early_signal=narrow and S >= 512,
            mm1_groups=narrow and S >= 512,
            mm1_resident=narrow and S >= 512,
            mm1_transpose=narrow and S >= 512,
            mm2_read_ahead=S < 512,  # gate_up's bf4 weights stream during the collectives
            mm2_bank_pairs=S < 512,
            **ccl,
        )
    return BoundaryConfig(
        mm1=MatmulConfig(M, 8, 12, min(2, M), 4),
        mm2=MatmulConfig(M, 8, 4, min(2, M), 4),
        mm_grid=grid,
        mm2_in1_depth=depth,
        mm1_split=s1,
        mm2_split=s2,
        mm1_grid=g1,
        mm1_quarter_splits=S < 512,  # each split one RS input quarter, the first-read ones on the bottom rows
        **ccl,
    )


def default_config(boundary: str, S: int, grid_x: int = 12) -> BoundaryConfig:
    """The tuned configuration for boundary "A" (o_proj -> gate_up) or "B" (down -> qkv) at S rows (S == 32: decode)
    on a grid grid_x columns wide."""
    return decode_config(boundary, grid_x) if S <= 32 else prefill_config(boundary, S, grid_x)


class TPBoundary:
    """One fused boundary for fixed shapes. __call__(x, inp, w1, w2) -> (h, out)."""

    def __init__(self, mesh, eps: float, boundary: str, S: int, n_out: int, cfg: BoundaryConfig | None = None):
        self.mesh, self.eps, self.boundary, self.S = mesh, eps, boundary, S
        g = mesh.compute_with_storage_grid_size()
        # a copy: the split adjusts its matmul configs
        self.cfg = replace(cfg or default_config(boundary, S, g.x))
        self.ring = tuple(mesh.shape)[1]
        self.grid = (g.x, g.y)
        self.dram_banks = mesh.dram_grid_size().x * mesh.dram_grid_size().y
        c = self.cfg
        self.aux = [(x, y) for y in c.aux_rows for x in range(g.x)]
        ox, oy = c.mm_origin
        gx_, gy_ = c.mm_grid
        self.mm1_grid = c.mm1_grid or c.mm_grid
        gy1 = self.mm1_grid[1]
        if c.mm1_quarter_splits:  # a split's share of a quarter, on a grid as wide as divides it
            assert c.mm1_split % self.ring == 0 and c.batches == 1 and not c.mm1_groups
            share = 4096 // 32 // self.ring // (c.mm1_split // self.ring)
            self.mm1_grid = (max(d for d in range(1, gx_ + 1) if share % d == 0), gy1)
        assert self.mm1_grid[0] == gx_ or c.mm1_quarter_splits
        assert max(c.mm1_split, c.mm2_split) == 1 or c.batches == 1
        assert c.mm2_row == 0 or not c.mm2_overlap
        self.mm_origins = [(ox, oy + c.mm2_row + s * gy_) for s in range(c.mm2_split)]
        self.mm1_origins = [(ox, oy + s * gy1) for s in range(c.mm1_split)]
        self.mm1_cores = [
            (o[0] + i, o[1] + j) for o in self.mm1_origins for j in range(gy1) for i in range(self.mm1_grid[0])
        ]
        rows = max(c.mm2_row + gy_ * c.mm2_split, gy1 * c.mm1_split)
        self.mm_cores = [(ox + i, oy + j) for j in range(rows) for i in range(gx_)]
        if c.mm1_grid is not None and gy1 != gy_:  # one core's rows on MM1's grid
            mb = -(-(S // 32) // gy1)
            c.mm1 = MatmulConfig(mb, c.mm1.K_block, c.mm1.N_block, min(2, mb), c.mm1.subblock_w)
        H, Hl = 4096, 4096 // self.ring
        n_w2 = 2 * n_out if c.swiglu else n_out
        if c.batches > 1:  # MM1's M block = a core's rows of one batch
            mb = c.mm1.M_block // c.batches
            c.mm1 = MatmulConfig(mb, c.mm1.K_block, c.mm1.N_block, min(c.mm1.subblock_h, mb), c.mm1.subblock_w)
        c.mm1 = _split_cfg(c.mm1, H // 32, c.mm1_split, gx_, False)
        if c.mm1.subblock_h * c.mm1.subblock_w > 8:
            c.mm1 = MatmulConfig(c.mm1.M_block, c.mm1.K_block, c.mm1.N_block, 1, c.mm1.subblock_w)
        c.mm2 = _split_cfg(c.mm2, n_w2 // 32, c.mm2_split, gx_, c.swiglu)
        if c.mm1_groups:  # one N block per quarter: a core's share of it
            assert c.mm1_split == 1 and c.batches == 1
            n_axis, m_axis = (gy1, gx_) if c.mm1_transpose else (gx_, gy1)
            nb, mb = -(-(H // 32 // self.ring) // n_axis), -(-(S // 32) // m_axis)
            sw = max(w for w in range(1, 9) if nb % w == 0)
            sh = max(h for h in range(1, mb + 1) if mb % h == 0 and h * sw <= 8)
            c.mm1 = MatmulConfig(mb, c.mm1.K_block, nb, sh, sw)
        elif c.mm1_quarter_splits:
            nb = 4096 // 32 // self.ring // (c.mm1_split // self.ring) // self.mm1_grid[0]
            sw = max(w for w in range(1, 9) if nb % w == 0)
            sh = max(h for h in range(1, c.mm1.M_block + 1) if c.mm1.M_block % h == 0 and h * sw <= 8)
            c.mm1 = MatmulConfig(c.mm1.M_block, c.mm1.K_block, nb, sh, sw)
        elif c.mm1_split == 1:
            c.mm1 = _fit_n_block(c.mm1, H // 32, gx_, False)
        if c.mm2_transpose:
            assert c.mm2_split == 1
            mb = -(-(S // 32) // gx_)
            c.mm2 = _fit_n_block(MatmulConfig(mb, c.mm2.K_block, 0, 1, 1), n_w2 // 32, gy_, c.swiglu)  # 0: refit
        elif c.mm2_split == 1:
            c.mm2 = _fit_n_block(c.mm2, n_w2 // 32, gx_, c.swiglu)
        self.P = 1  # one partial-sum tile per row tile and device: the stats all-gather stays small

        def alloc(shape, mem, dtype=ttnn.bfloat16):
            return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, mesh, mem)

        self.buf_p = alloc([1, 1, S, H], L1)  # MM1 out / RS in
        self.buf_h = alloc([1, 1, S, H], L1)  # AG out / MM2 in
        self.rs_interm = alloc([1, 1, S, H], L1)
        self.h = alloc([1, 1, S, Hl], DRAM)  # new residual
        sdt = ttnn.float32 if self.cfg.aux_fp32 else ttnn.bfloat16
        self.stats = alloc([1, 1, S, 32 * self.P], L1, sdt)
        self.stats_g = alloc([1, 1, S, 32 * self.P * self.ring], L1, sdt)
        self.scale = alloc([1, 1, S, 32], L1)
        self.out = alloc([1, 1, S, n_out], c.out_mem)
        self.ones = ttnn.ones(
            [1, 1, 32, 32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=L1
        )
        # per matmul core: one 16 B counter per ring slice (the all-gather -> MM2 slice-ready signals)
        from .compose import core_range_set

        n = len(self.mm_cores)
        cmem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_range_set(self.mm_cores), [1, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.counters = ttnn.allocate_tensor_on_device(
            ttnn.Shape([n, 32]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, cmem
        )
        # per aux core: one 16 B counter per batch (MM1 -> RS batch-ready signals)
        amem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_range_set(self.aux), [1, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.rs_counters = ttnn.allocate_tensor_on_device(
            ttnn.Shape([len(self.aux), 32]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, amem
        )
        self.rs_sems = GlobalSems(mesh, 4)
        self.ag_sems = GlobalSems(mesh, 3)
        self.sag_sems = GlobalSems(mesh, 3)
        self._arena_size = None
        self.arena = None
        self._desc = None
        self._key = None

    def _arena_plan(self, inp, w1, w2):
        """MM1's CB offsets in the arena, and MM2's weight CB depth (it starts at offset 0 once MM1 is done)."""
        c = self.cfg
        k1 = (inp.padded_shape[-1] // 32) // c.mm1.K_block
        b1 = matmul_cb_bytes(inp.dtype, w1.dtype, ttnn.bfloat16, c.mm1, in0_cb_blocks=k1 if c.mm1_resident else 2)
        offs, o = {}, 0
        for k in ("in1", "in0", "out", "interm"):
            offs[k] = o
            o += -(-b1[k] // 64) * 64
        r1 = o
        K_blocks = (self.buf_h.padded_shape[-1] // 32) // c.mm2.K_block
        blk = matmul_cb_bytes(ttnn.bfloat16, w2.dtype, ttnn.bfloat16, c.mm2, c.swiglu, in1_cb_blocks=1)["in1"]
        if not c.mm2_overlap:  # all of MM2's weights, after MM1's CBs
            offs["mm2_in1"] = r1
            return offs, K_blocks, r1 + K_blocks * blk
        size = max(r1, min(K_blocks * blk, c.arena_cap))
        depth = max(2, min(K_blocks, size // blk, c.mm2_in1_depth))
        size = max(size, depth * blk)
        return offs, depth, size

    # ------------------------------------------------------------------ composition

    def build_device(self, j: int, x, inp, w1, w2) -> LegoProgram:
        c, mesh = self.cfg, self.mesh
        ring = ring_of(mesh, (0, j))
        prog = LegoProgram()
        aux0 = (0, c.aux_rows[0])
        gx = self.grid[0]

        def wait(cores, risc, sem, count, label):
            prog.place(
                cores,
                Block(
                    cls="lego::sync::Wait",
                    header=f"{SYNC}/wait.hpp",
                    risc=risc,
                    ct=dict(sem=sem, count=count),
                    label=label,
                ),
            )

        rs_workers, rs_muxes = ccl_layout(origin=aux0, num_links=c.num_links, nw=c.rs_workers, grid_x=gx)
        ag_workers, ag_muxes = ccl_layout(origin=aux0, num_links=c.num_links, nw=c.ag_workers, grid_x=gx)
        # the stats all-gather on the last two aux cores (free of RS / AG connection semaphores)
        sag_workers, _ = ccl_layout(origin=(gx - 2, c.aux_rows[-1]), num_links=1, nw=1, grid_x=gx)
        rs_cores = [w for ws in rs_workers.values() for w in ws]
        ag_cores = [w for ws in ag_workers.values() for w in ws]
        sag_cores = [w for ws in sag_workers.values() for w in ws]
        rs_mux_cores = list(rs_muxes.values()) if rs_muxes else []
        ag_mux_cores = list(ag_muxes.values()) if ag_muxes else []

        # MM1: inp @ W1 -> P, its CBs in the arena
        offs, in1_depth, _ = self._arena_plan(inp, w1, w2)
        mm2_in1_off = offs.pop("mm2_in1", 0)
        rs_cores_ = [w for ws in rs_workers.values() for w in ws]
        # the RS reads the input quarters in this order: the device across the ring, the two neighbours, its own
        r = ring.index
        q_order = [(r + 2) % self.ring, (r + 1) % self.ring, (r - 1) % self.ring, r] if self.ring == 4 else None
        assert q_order is not None or not (c.mm1_groups or c.mm1_quarter_splits)
        s1 = c.mm1_split
        n1 = c.mm_grid[0] * c.mm1.N_block  # a split's columns: whole per-core chunks
        q_tiles = 4096 // 32 // self.ring
        per_q = s1 // self.ring  # quarter splits: splits per quarter, ranked bottom first
        for s_, org in enumerate(self.mm1_origins):
            if c.mm1_quarter_splits:
                rank, part = divmod(s1 - 1 - s_, per_q)
                nr = (q_order[rank] * q_tiles + part * (q_tiles // per_q), q_tiles // per_q)
                sig = (self.rs_counters, rs_cores_, rank)
            else:
                nr = (s_ * n1, n1) if s1 > 1 else None
                sig = (self.rs_counters, rs_cores_) if c.mm1_groups else None
            p1 = place_matmul(
                prog,
                mesh,
                origin=org,
                grid=self.mm1_grid,
                in0=inp,
                weight=w1,
                out=self.buf_p,
                cfg=c.mm1,
                label="mm1",
                arena={k: (self.arena, v) for k, v in offs.items()},
                m_batches=c.batches,
                signal_m_batches=(self.rs_counters, rs_cores_) if c.batches > 1 else None,
                n_range=nr,
                transpose=True if c.mm1_transpose else None,
                n_groups=(q_tiles, q_order) if c.mm1_groups else None,
                signal_n_blocks=sig,
                in0_resident=c.mm1_resident,
                in1_read_ahead=c.mm1_read_ahead,
            )
        # MM1 done on a core (its output written) -> that core's MM2 weight reader may reuse MM1's arena bytes
        if c.mm2_overlap:
            # the waiting RISC runs MM2's weight reader; a transposed MM2 reads its weights on MM1's writer RISC, which
            # then waits for MM1's weight reader instead
            other = "dm1" if p1.writer_risc == "dm0" else "dm0"
            prod, cons = (other, p1.writer_risc) if c.mm2_transpose else (p1.writer_risc, other)
            place_local_handoff(prog, mesh, self.mm_cores, prod, cons, label="l_mm1")
        # P complete -> RS readers (batched: per batch, inside the RS reader)
        if c.batches == 1 and not (c.mm1_groups or c.mm1_quarter_splits):
            place_handoff_multi(prog, mesh, [(self.mm1_cores, p1.writer_risc)], rs_cores, "dm0", label="h_mm1")
        if c.mm2_after_mm1:
            mm2_cores = [
                (o[0] + i, o[1] + r) for o in self.mm_origins for r in range(c.mm_grid[1]) for i in range(c.mm_grid[0])
            ]
            place_handoff_multi(
                prog,
                mesh,
                [(self.mm1_cores, p1.writer_risc)],
                mm2_cores,
                "dm1" if c.mm2_transpose else "dm0",
                label="h_m12",
            )
        # MM2's in0 injectors, per row group r: the first core of row r in each N split (transposed: the first core of
        # column r)
        if c.mm2_transpose:
            inj_groups = [[(o[0] + r, o[1]) for o in self.mm_origins] for r in range(c.mm_grid[0])]
        else:
            inj_groups = [[(o[0], o[1] + r) for o in self.mm_origins] for r in range(c.mm_grid[1])]
        injectors = [i for g in inj_groups for i in g]
        group_of = {i: r for r, g in enumerate(inj_groups) for i in g}
        m_per_inj = -(-(self.S // 32) // len(inj_groups))
        batch_rows = (self.S // 32) // c.batches
        own_targets = (
            injectors
            if c.batches == 1
            else [
                [inj for inj in injectors if (group_of[inj] * m_per_inj) // batch_rows == b] for b in range(c.batches)
            ]
        )
        # RS also writes its result straight into this device's slot of H and tells MM2 the slot is ready
        place_reduce_scatter(
            prog,
            mesh,
            ring,
            rs_workers,
            muxes=rs_muxes,
            input=self.buf_p,
            interm=self.rs_interm,
            output=self.h,
            sems=self.rs_sems,
            residual=x,
            reader_risc="dm0",
            writer_risc="dm1",
            label="rs",
            fp32_dest_acc_en=c.aux_fp32,
            dst_full_sync_en=True,  # aux compute config
            math_fidelity=c.aux_fidelity,
            chunks_per_sync=c.rs_chunks_per_sync,
            dual=(self.buf_h, j * (4096 // self.ring) // 32),
            signal_own=(self.counters, own_targets, j),
            batches=c.batches,
            batch_wait=(self.rs_counters, len(self.mm_cores)) if c.batches > 1 else None,
            slice_wait=(
                self.rs_counters,
                len(self.mm1_cores) // (self.ring if c.mm1_quarter_splits else 1),
                [q_order.index(q) for q in range(self.ring)],
            )
            if c.mm1_groups or c.mm1_quarter_splits
            else None,
        )
        # h complete (every RS writer done) -> every aux core: AG readers and writers (peers may only write into H
        # once this device's RS is done) and the statistics readers
        sem, cnt = place_handoff_multi(prog, mesh, [(rs_cores, "dm1")], self.aux, "dm0", label="h_rs")
        wait(ag_cores, "dm1", sem, cnt, "h_rs_w")
        # The RS muxes hold the fabric channels until they terminate: the AG muxes connect after that.
        if rs_mux_cores and ag_mux_cores:
            place_handoff_multi(prog, mesh, [(rs_mux_cores, "dm0")], ag_mux_cores, "dm0", label="h_fab1")
        # AG streams H into MM2: each MM2 in0 injector is signalled as its rows of a slice land, MM2 reads K in arrival
        # order. Injector r owns M rows [r * m_per, (r + 1) * m_per): row group r.
        Hl = 4096 // self.ring
        group = m_per_inj * (Hl // 32)
        # The AG muxes also carry the statistics all-gather (link 0, one worker per direction) as extra clients, so
        # it runs alongside instead of waiting for the AG to release the fabric.
        ag_mux_objs = None
        if ag_muxes:
            ag_mux_objs = {}
            for (l, d), core in ag_muxes.items():
                extra = sag_workers.get((l, d), []) if c.stats_on_ag_mux else []
                dst = ring.fwd if d == 0 else ring.bwd
                mux = place_mux(prog, mesh, ring.coord, dst, l, core, ag_workers[(l, d)] + extra, label="ag_mux")
                ag_mux_objs[(l, d)] = (mux, 0)
        place_all_gather(
            prog,
            mesh,
            ring,
            ag_workers,
            muxes=ag_mux_objs,
            input=self.h,
            output=self.buf_h,
            sems=self.ag_sems,
            reader_risc="dm0",
            writer_risc="dm1",
            label="ag",
            signal=(self.counters, inj_groups),
            write_local=False,
            row_group_tiles=group,
            signal_while_forwarding=c.ag_early_signal,
        )
        near, far = ag_row_group_waits(self.S, Hl, c.num_links, c.ag_workers, group, self.ring)
        k_order = dict(
            first=j,
            slices=self.ring,
            counters=self.counters,
            waits=lambda core: (len(rs_cores), near[group_of[core]], far[group_of[core]])
            if core in group_of
            else (0, 0, 0),
        )
        # statistics on the aux cores the AG does not use, in parallel with it: local sum of squares -> ring
        # all-gather of the partial sums -> 1/rms per row
        st_cores = [c for c in self.aux if c not in ag_cores and c not in (ag_mux_cores or [])] or self.aux
        if c.stats_from_gathered:
            # H complete on this device (every AG reader done; this device's own slot came from the RS) -> partial
            # sums of squares of each slice straight into the gathered statistics -> 1/rms
            place_handoff_multi(prog, mesh, [(ag_cores, "dm0")], st_cores, "dm0", label="h_ag")
            stat_cores = place_rms_stats(
                prog,
                st_cores,
                src=self.buf_h,
                stats=self.stats_g,
                reader="dm0",
                writer="dm1",
                dst_full_sync_en=True,
                fp32_dest_acc_en=c.aux_fp32,
                math_fidelity=c.aux_fidelity,
            )
            place_handoff_multi(prog, mesh, [(stat_cores, "dm1")], st_cores, "dm0", label="h_st")
        else:
            self._stats_allgather(
                prog, c, mesh, ring, st_cores, sag_workers, sag_cores, ag_workers, ag_mux_objs, ag_mux_cores, ag_cores
            )
        scale_cores = place_rms_scale(
            prog,
            st_cores,
            stats=self.stats_g,
            ones=self.ones,
            scale=self.scale,
            width=4096,
            eps=self.eps,
            reader="dm0",
            writer="dm1",
            dst_full_sync_en=True,
            fp32_dest_acc_en=c.aux_fp32,
            math_fidelity=c.aux_fidelity,
        )
        scale_sem, scale_cnt = place_handoff_multi(
            prog, mesh, [(scale_cores, "dm1")], self.mm_cores, None, label="h_scale"
        )
        # MM2: H @ W2, rows scaled by 1/rms
        n2 = c.mm_grid[0] * c.mm2.N_block
        for s_, org in enumerate(self.mm_origins):
            place_matmul(
                prog,
                mesh,
                origin=org,
                grid=c.mm_grid,
                in0=self.buf_h,
                weight=w2,
                out=self.out,
                cfg=c.mm2,
                fuse_swiglu=c.swiglu,
                row_scale=(self.scale, scale_sem, scale_cnt),
                label="mm2",
                transpose=True if c.mm2_transpose else None,
                in1_read_ahead=c.mm2_read_ahead,
                in1_bank_pairs=self.dram_banks if c.mm2_bank_pairs else 0,
                k_order=k_order,
                arena={"in1": (self.arena, mm2_in1_off)},
                in1_cb_blocks=in1_depth,
                n_range=(s_ * n2, n2) if c.mm2_split > 1 else None,
            )
        return prog

    def _stats_allgather(
        self, prog, c, mesh, ring, st_cores, sag_workers, sag_cores, ag_workers, ag_mux_objs, ag_mux_cores, ag_cores
    ):
        """Statistics on the local slice, then a ring all-gather of the partial sums (prefill)."""
        stat_cores = place_rms_stats(
            prog,
            st_cores,
            src=self.h,
            stats=self.stats,
            reader="dm0",
            writer="dm1",
            dst_full_sync_en=True,
            fp32_dest_acc_en=c.aux_fp32,
            math_fidelity=c.aux_fidelity,
        )
        place_handoff_multi(prog, mesh, [(stat_cores, "dm1")], sag_cores, "dm0", label="h_st")
        sag_muxes = None
        if ag_mux_objs and c.stats_on_ag_mux:
            sag_muxes = {k: (ag_mux_objs[k][0], len(ag_workers[k])) for k in sag_workers}
        else:  # direct connections: wait until the AG has released the fabric
            fab_prev, fab_risc = (ag_mux_cores, "dm0") if ag_mux_cores else (ag_cores, "dm1")
            place_handoff_multi(prog, mesh, [(fab_prev, fab_risc)], sag_cores, "dm1", label="h_fab2")
        place_all_gather(
            prog,
            mesh,
            ring,
            sag_workers,
            muxes=sag_muxes,
            input=self.stats,
            output=self.stats_g,
            sems=self.sag_sems,
            reader_risc="dm0",
            writer_risc="dm1",
            label="sag",
        )
        place_handoff_multi(prog, mesh, [(sag_cores, "dm0"), (sag_cores, "dm1")], st_cores, "dm0", label="h_sag")

    def __call__(self, x, inp, w1, w2):
        key = (x.buffer_address(), inp.buffer_address(), w1.buffer_address(), w2.buffer_address())
        if self.arena is None:
            from .compose import Arena

            self.arena = Arena.allocate(self.mesh, self.mm_cores, self._arena_plan(inp, w1, w2)[2])
        if self._key != key:
            progs = {(0, j): self.build_device(j, x, inp, w1, w2).descriptor() for j in range(self.ring)}
            self._desc = mesh_descriptor(progs)
            self._key = key
        io = [
            x,
            inp,
            w1,
            w2,
            self.buf_p,
            self.buf_h,
            self.rs_interm,
            self.stats,
            self.stats_g,
            self.scale,
            self.ones,
            self.counters,
            self.rs_counters,
            self.arena.tensor,
            self.h,
            self.out,
        ]
        ttnn.generic_op(io, self._desc)
        return self.h, self.out
