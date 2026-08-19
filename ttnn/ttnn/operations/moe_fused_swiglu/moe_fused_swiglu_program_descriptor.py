# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program descriptor for quickly iterating on Moe fused-SwiGLU kernels and geometry.

The production operation is implemented in C++ under
``ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu``.
This descriptor realizes the same blocking model on an ``HGROUPS x KGROUPS``
worker grid:

  * **Hn** (hidden, gate/up output)      -> split across grid COLUMNS, ``hn_pad`` tiles each
  * **Kg** (emb, gate/up contraction)    -> split across grid ROWS, reduced by the reduce-scatter
  * **Ne** (emb, ``down`` output)        -> split across ALL cores, ``ec`` tiles each
  * **Kh** (hidden, ``down`` contraction)-> sequential per core, ``HGROUPS`` K-blocks
  * **M**  (tokens)                      -> sequential outer loop; the only RUNTIME extent

``moe_fused_swiglu_geometry.py`` owns the blocking arithmetic and L1 tuning
knobs used here.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_geometry as geo
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_geometry import (  # noqa: F401  (re-exported)
    Blocking,
    MAILBOX_MAGIC,
    MAILBOX_WORDS,
    TILE,
)

KERNEL_DIR = (
    Path(__file__).parents[3] / "cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels"
)

#: Weight dtypes the op can stream. All three weights must share one; the CB format and the tile
#: stride both come from it.
WEIGHT_DTYPES = (ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16)


# --------------------------------------------------------------------------------------------
# Compile-time argument order — mirrors kernels/moe_fused_swiglu_ct_args.hpp exactly.
# `test_moe_fused_swiglu_ct_args.py` parses that header and asserts these agree.
# --------------------------------------------------------------------------------------------
KERNEL_CT_ORDER = {
    "reader": (
        "INPUT_FORMAT",
        "M_T_MAX",
        "LOCAL_EXPERT_ID",
        "EMB_T",
        "HID_T",
        "KR_PAD",
        "HN_PAD",
        "EC_MAX",
        "WD_EC_MAX",
        "EC_GROUP_MAX",
        "M_BLOCK",
        "HGROUPS",
        "KGROUPS",
        "NUM_CORES",
        "SEM_GO",
        "SEM_DATA",
        "SEM_HSLICE",
        "SEM_XSTAGED",
        "SEM_H_RDY_BASE",
        "SEM_H_FREE",
        "SEM_WDSPLIT",
        "SEM_HROW_FREE",
        "SEM_PHASE_FREE",
        "PHASE_CB_ALIAS",
        "H_ROUND_NOC1_MASK",
        "SCATTER_ONE_SIGNAL",
        "X_PAGE",
        "X_SLICE",
        "COUNTS_PAGE",
        "IDX_PAGE",
        "W_TILE_BYTES",
        "BFP8_TILE",
        "MAILBOX_MAGIC",
        "WD_AHEAD",
        "M_EFF_MIN",
        "W_RESIDENT",
        "WD_RESIDENT",
        "WD_MROW_ROUNDS",
        "WD_MGROUPS",
        "MGROUP_ROWS",
        "WD_MGROUP_MIN_BLOCKS",
        "GU_CHUNKS",
        "XPRIO",
        "HACK_AHEAD",
        "DEPTH_H",
        "DEPTH_X",
        "WD_SPLIT",
        "WG_SHARD_W",
        "WD_SHARD_W",
        "GATHER_PAGES",
        "NEED_START",
        "READ_X_AT_OFFSET",
        "START_PAGE",
        "CB_X_IN",
        "CB_X_TILES",
        "CB_X_STAGE",
        "CB_W_GATE",
        "CB_W_DOWN",
        "CB_H",
        "CB_H_LOCAL",
        "CB_IDX_SCRATCH",
        "CB_COUNTS_SCRATCH",
        "CB_GATHER_GATE",
        "CB_GATHER_UP",
        "CB_UP_ACC",
        "CB_MAILBOX_COMPUTE",
        "CB_MAILBOX_WRITER",
    ),
    "writer": (
        "EMB_T",
        "HID_T",
        "KR_PAD",
        "HN_PAD",
        "EC_MAX",
        "WD_EC_MAX",
        "EC_GROUP_MAX",
        "M_BLOCK",
        "HGROUPS",
        "KGROUPS",
        "NUM_CORES",
        "SEM_GO",
        "SEM_DATA",
        "SEM_HSLICE",
        "SEM_XSTAGED",
        "SEM_H_RDY_BASE",
        "SEM_H_FREE",
        "SEM_WDSPLIT",
        "SEM_PHASE_FREE",
        "SEM_HROW_FREE",
        "PHASE_CB_ALIAS",
        "W_TILE_BYTES",
        "BFP8_TILE",
        "OUT_TILE_BYTES",
        "MAILBOX_MAGIC",
        "M_EFF_MIN",
        "W_RESIDENT",
        "WD_RESIDENT",
        "GU_CHUNKS",
        "XPRIO",
        "WD_MROW_ROUNDS",
        "WD_MGROUPS",
        "MGROUP_ROWS",
        "WD_MGROUP_MIN_BLOCKS",
        "DEPTH_H",
        "H_ROUND_NOC1_MASK",
        "SCATTER_ONE_SIGNAL",
        "WD_SPLIT",
        "WG_SHARD_W",
        "WD_SHARD_W",
        "GATHER_PAGES",
        "PHASE_ALIAS_PAGES",
        "DIRECT_WRITE",
        "OUT_M_T",
        "CB_W_UP",
        "CB_W_DOWN",
        "CB_OUT_TILES",
        "CB_GATE_ACC",
        "CB_UP_ACC",
        "CB_GATHER_GATE",
        "CB_GATHER_UP",
        "CB_H_SLICE",
        "CB_H_LOCAL",
        "CB_H",
        "CB_MAILBOX_WRITER",
    ),
    "compute": (
        "M_BLOCK",
        "KR_PAD",
        "HN_PAD",
        "EC_MAX",
        "WD_EC_MAX",
        "EC_GROUP_MAX",
        "HGROUPS",
        "KGROUPS",
        "HID_T",
        "INPUT_FORMAT",
        "OUT_SUBBLOCK_H_GU",
        "OUT_SUBBLOCK_H_DN",
        "OUT_SUBBLOCK_H_DN_MAX",
        "MAILBOX_MAGIC",
        "M_EFF_MIN",
        "DEPTH_X",
        "HN_BLOCK",
        "WD_RESIDENT",
        "WD_MROW_ROUNDS",
        "WD_MGROUPS",
        "MGROUP_ROWS",
        "WD_MGROUP_MIN_BLOCKS",
        "GU_CHUNKS",
        "ELTWISE_BLK",
        "DEST_LIMIT",
        "GATHER_PAGES",
        "CB_X_IN",
        "CB_X_TILES",
        "CB_X_STAGE",
        "CB_MAILBOX_COMPUTE",
        "CB_W_GATE",
        "CB_W_UP",
        "CB_W_DOWN",
        "CB_GATE_ACC",
        "CB_UP_ACC",
        "CB_GATE_SILU",
        "CB_H_LOCAL",
        "CB_H",
        "CB_OUT_INTERM",
        "CB_OUT_TILES",
        "CB_GATHER_GATE",
        "CB_GATHER_UP",
        "CB_SLICE_GATE",
        "CB_SLICE_UP",
        "CB_H_SLICE",
    ),
}

#: Runtime argument order, same contract. The kernels index these positionally off named
#: constants derived from these lengths.
KERNEL_RT_SCALARS = {
    "reader": (
        "MAILBOX",
        "X_ADDR",
        "WG_ADDR",
        "WD_ADDR",
        "COUNTS_ADDR",
        "IDX_ADDR",
        "KR",
        "KSTART",
        "HSTART",
        "HN",
        "EC",
        "JSTART",
        "EC_GROUP",
        "JSTART_GROUP",
        "MY_COL",
        "MY_ROW",
        # `start` (= expert_region_offsets). LAST in the scalar block, so RT_PEERS and every
        # mcast-arg offset derived from it shift together in the kernel and nothing is hand-indexed.
        "START_ADDR",
    ),
    "writer": (
        "MAILBOX",
        "WU_ADDR",
        "OUT_ADDR",
        "WD_ADDR",
        "KR",
        "KSTART",
        "HSTART",
        "HN",
        "EC",
        "JSTART",
        "EC_GROUP",
        "JSTART_GROUP",
        "MY_COL",
        "MY_ROW",
        "ROOT_ROW",
    ),
    "compute": ("MAILBOX", "KR", "HN", "EC", "EC_GROUP", "MY_COL", "MY_ROW"),
}


# --------------------------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------------------------
def worker_grid(device, core_grid=None):
    """The ``(HGROUPS, KGROUPS)`` used by the C++ op and weight-placement helper.

    ``core_grid`` is a ``(x, y)`` pair or ``ttnn.CoreCoord``. By default the complete
    compute-with-storage grid is used. Explicit requests select a rectangular prefix anchored at
    ``(0, 0)`` and are rejected only when they exceed the device.
    """
    grid = device.compute_with_storage_grid_size()
    max_hgroups, max_kgroups = int(grid.x), int(grid.y)
    hgroups, kgroups = max_hgroups, max_kgroups
    if core_grid is not None:
        gx, gy = (
            (int(core_grid.x), int(core_grid.y)) if hasattr(core_grid, "x") else (int(core_grid[0]), int(core_grid[1]))
        )
        if gx < 1 or gy < 1:
            raise ValueError(f"moe_fused_swiglu: core_grid must be positive, got {gx}x{gy}")
        if gx > max_hgroups or gy > max_kgroups:
            raise ValueError(
                f"moe_fused_swiglu: requested grid {gx}x{gy} exceeds device grid " f"{int(grid.x)}x{int(grid.y)}"
            )
        hgroups, kgroups = gx, gy
    return hgroups, kgroups


def nd_shard_n_tiles(t):
    """N-axis TILES per DRAM ND shard of `t`, or 0 if no contiguous run can be proven.

    This is the ONE place the op learns a weight's placement; everything downstream is a run
    length. Inside one shard, consecutive N pages are physically contiguous in one bank
    (`TensorAccessor::get_bank_and_offset`), so a core's slice decomposes into runs that issue as
    single NoC transactions. 0 means one transaction per tile — correct, and slower.

    Interleaved placement returns 0 because it has no contiguous N-axis shard run.
    """
    try:
        mc = t.memory_config()
        spec = mc.nd_shard_spec
        if spec is None or mc.buffer_type != ttnn.BufferType.DRAM:
            return 0
        shape = list(spec.shard_shape)
    except Exception:  # pragma: no cover - a tensor type without the attribute
        return 0
    if len(shape) < 2 or int(shape[-1]) % TILE != 0:
        return 0
    return int(shape[-1]) // TILE


def weight_memory_configs(device, emb, hidden, core_grid=None, shard_height_tiles=1, transpose_grid=False):
    """The op's PREFERRED weight placement, as `(gate_up, down)` memory configs.

    A pure function of K, N and the grid — never of the runtime token count. Each shard is exactly
    the N slice ONE core consumes for ONE K-row, so that read is a single transaction:
      * gate/up — `hn_pad` tiles, the hidden split across grid COLUMNS
      * down    — `ec_max` tiles, the emb-output split across ALL cores

    The default shard is one tile row tall. ``shard_height_tiles`` can be raised to choose a
    different compatible layout.

    Placement is the CALLER's to choose. The op reads whatever it is handed and takes the
    coalesced path whenever `nd_shard_n_tiles` can prove a run.
    """
    physical_columns, physical_rows = worker_grid(device, core_grid)
    hgroups, kgroups = (physical_rows, physical_columns) if transpose_grid else (physical_columns, physical_rows)
    blk = Blocking(hgroups, kgroups, emb, hidden, m_t_max=1)
    dram = device.dram_grid_size()
    banks = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])

    def mc(n_tiles):
        return ttnn.MemoryConfig(
            ttnn.BufferType.DRAM,
            ttnn.NdShardSpec(
                shard_shape=ttnn.Shape([shard_height_tiles * TILE, n_tiles * TILE]),
                grid=banks,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    return mc(blk.hn_pad), mc(blk.wd_ec_max)


def make_mailbox(device, num_cores):
    """One L1 page per core for the runtime token count the reader publishes.

    Zeroed host-side so a stale magic from a previous dispatch can never be read as fresh.
    """
    import torch

    return ttnn.from_torch(
        torch.zeros((1, 1, num_cores, MAILBOX_WORDS), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


# --------------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------------
def _virt(device, x, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return int(c.x), int(c.y)


def _mcast_compile_time_args(data_ready_sem, consumer_ready_sem, receivers, handshake):
    """Encode the multicast compile-time contract consumed by the dataflow kernels."""
    return [int(receivers != 0), data_ready_sem, consumer_ready_sem, receivers, int(handshake)]


def _rotating_mcast_args(device, noc, x0, y0, x1, y1):
    """Encode a logical rectangle as NOC corners followed by row-major senders."""
    senders = [_virt(device, x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    xs, ys = zip(*senders)
    corners = [min(xs), min(ys), max(xs), max(ys)]
    if noc == ttnn.NOC.NOC_1:
        corners = [max(xs), max(ys), min(xs), min(ys)]
    return corners + [coordinate for sender in senders for coordinate in sender]


def _rotating_mcast_args_for_cores(device, noc, cores):
    """Encode one physical rectangle while preserving ``cores`` as the sender rotation order."""
    senders = [_virt(device, x, y) for x, y in cores]
    xs, ys = zip(*senders)
    corners = [min(xs), min(ys), max(xs), max(ys)]
    if noc == ttnn.NOC.NOC_1:
        corners = [max(xs), max(ys), min(xs), min(ys)]
    return corners + [coordinate for sender in senders for coordinate in sender]


def _cb_allocation(total_size, core_ranges, logical_views, formats):
    """One physical allocation, exposed through one or more independent logical CB views."""
    return ttnn.CBDescriptor(
        total_size=total_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=formats[key], page_size=page_size)
            for index, _, page_size, key in logical_views
        ],
    )


def _ct_args(kernel, values):
    """Positional vector in `KERNEL_CT_ORDER[kernel]`, with every name required exactly once."""
    order = KERNEL_CT_ORDER[kernel]
    missing = [n for n in order if n not in values]
    if missing:
        raise RuntimeError(f"moe_fused_swiglu: {kernel} compile-time args missing {missing}")
    return [int(values[n]) for n in order]


# --------------------------------------------------------------------------------------------
def create_program_descriptor(
    input_tensor,
    w_gate,
    w_up,
    w_down,
    counts,
    global_expert_idx_table,
    output_tensor,
    mailbox,
    *,
    local_expert_id,
    input_m_tiles,
    compute_kernel_config,
    core_grid=None,
    expert_region_offsets=None,
    read_x_at_offset=False,
    transpose_grid=False,
    situ_glu=False,
):
    device = input_tensor.device()
    physical_columns, physical_rows = worker_grid(device, core_grid)
    hgroups, kgroups = (physical_rows, physical_columns) if transpose_grid else (physical_columns, physical_rows)
    num_cores = hgroups * kgroups
    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(physical_columns - 1, physical_rows - 1))]
    )

    def physical_core(hgroup, kgroup):
        return (kgroup, hgroup) if transpose_grid else (hgroup, kgroup)

    emb = int(input_tensor.shape[-1])
    hidden = int(w_gate.shape[-1])
    x_is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    bfp8_tile = ttnn.tile_size(ttnn.bfloat8_b)
    requested_out_tile = ttnn.tile_size(output_tensor.dtype)
    # The BF16 phase-scratch alias is valid for either output dtype.  The output/gather alias has
    # the additional same-page-size predicate in Blocking.phase_cb_alias().
    enable_phase_alias = True
    kr_pad = -(-(emb // TILE) // kgroups)
    blk = Blocking(
        hgroups,
        kgroups,
        emb,
        hidden,
        input_m_tiles,
        w_tile=ttnn.tile_size(w_gate.dtype),
        bfp8_tile=bfp8_tile,
        bf16_tile=ttnn.tile_size(ttnn.bfloat16),
        x_stick=(kr_pad * TILE * input_tensor.element_size()) if x_is_rm else bfp8_tile,
        l1_budget=int(ttnn._ttnn.device.get_max_worker_l1_unreserved_size()) - geo.L1_CB_RESERVE,
        out_tile=requested_out_tile,
        enable_phase_alias=enable_phase_alias,
        x_is_rm=x_is_rm,
    )
    idx_page = max(int(global_expert_idx_table.buffer_aligned_page_size()), ttnn.get_dram_alignment())
    counts_page = max(int(counts.buffer_aligned_page_size()), ttnn.get_dram_alignment())

    # ---- shared-buffer region mode (fused extract / insert) --------------------------------
    # `expert_region_offsets` present => the writer places this expert's rows into `output_tensor`
    # (the SHARED destination) at start[global_id], fusing ttnn::insert. `read_x_at_offset` opts the
    # reader into the mirror-image rebase of its x reads, fusing ttnn::extract. Same two knobs, same
    # meanings as `unified_routed_expert_ffn`; `counts` stands in for the accessor + address when no
    # offsets tensor is passed, so the argument streams have ONE length and the kernels compile
    # identically either way.
    direct_write = expert_region_offsets is not None
    start_tensor = expert_region_offsets if direct_write else counts
    start_page = max(int(start_tensor.buffer_aligned_page_size()), ttnn.get_dram_alignment())
    # The reader fetches `start` over cb_counts_scratch's page (dead once `count` is extracted), so
    # that ONE page must hold either vector. Equal by validation; the max is the belt.
    counts_page = max(counts_page, start_page)
    phase_cb_alias = enable_phase_alias and blk.phase_cb_alias(requested_out_tile)
    need = blk.l1_bytes(x_is_rm, requested_out_tile, enable_phase_alias=enable_phase_alias)
    if need > blk.l1_budget:
        raise RuntimeError(
            f"moe_fused_swiglu: needs {need} B of L1 per core, device has {blk.l1_budget} B.\n"
            f"  {blk.describe()}\n"
            f"  W_down residency is already {'off' if not blk.wd_resident else 'on'}. What is left "
            f"scales with hn_pad ({blk.hn_pad} tiles, = hidden / grid COLUMNS) and with the weight "
            f"dtype ({blk.w_tile} B/tile).\n"
            f"  A NARROWER grid makes this WORSE — fewer columns means a wider hn_pad. What helps: "
            f"more grid columns, a smaller hidden, a smaller emb, or a narrower weight dtype."
        )

    # ---- dtypes -----------------------------------------------------------------------------
    # The weight stride and CB format both come from the tensor. All three weights share a dtype
    # (validate() enforces it), so one width serves the gate/up and down streams alike.
    w_dtype = w_gate.dtype
    w_tile = blk.w_tile
    bf16_tile = blk.bf16_tile
    dram_align = ttnn.get_dram_alignment()
    out_dtype = output_tensor.dtype
    out_tile = ttnn.tile_size(out_dtype)

    input_format = 0 if x_is_rm else 1
    x_stick_slice = blk.kr_pad * TILE * input_tensor.element_size() if x_is_rm else bfp8_tile

    # ---- weight placement -------------------------------------------------------------------
    # gate/up resolve to ONE width: the reader and the writer read the same [k, n] slice of two
    # identically-shaped tensors, and a disagreement would give the up matmul a different
    # coalescing from the gate's.
    # ONE width for gate/up, and they must AGREE. Taking min() of two different widths invents
    # shard boundaries that do not subdivide the real ones — with widths 5 and 4, a run over
    # [4, 6) would cross the width-5 tensor's real boundary at 5 and be issued as one transaction
    # across two banks. Disagreement falls back to the uncoalesced stream, which is always correct.
    _wg, _wu = nd_shard_n_tiles(w_gate), nd_shard_n_tiles(w_up)
    wg_shard_w = _wg if _wg == _wu else 0
    wd_shard_w = nd_shard_n_tiles(w_down)

    # ---- collectives ------------------------------------------------------------------------
    # DataReadySignal::Flag, not Counter: the Counter signal is an atomic on a different command
    # buffer, so it cannot terminate the data multicast's NOC_CMD_VC_LINKED chain and the sender
    # hangs. The h all-gather works around Flag's per-round reset chain with per-slot cells of its
    # own (SEM_H_RDY_BASE); see the reader.
    x_mcast_ct = _mcast_compile_time_args(geo.SEM_X_BASE, geo.SEM_X_BASE + 1, hgroups - 1, True)
    h_mcast_ct = _mcast_compile_time_args(geo.SEM_H_BASE, geo.SEM_H_BASE + 1, num_cores - 1, True)
    logical_cores = [physical_core(h, k) for k in range(kgroups) for h in range(hgroups)]
    h_mcast_args = _rotating_mcast_args_for_cores(device, ttnn.NOC.NOC_0, logical_cores)
    h_mcast_noc1_args = _rotating_mcast_args_for_cores(device, ttnn.NOC.NOC_1, logical_cores)[:4]

    h_group_rect_args = []
    if blk.wd_mgroups:
        for k0 in range(0, kgroups, blk.mgroup_rows):
            k1 = min(k0 + blk.mgroup_rows - 1, kgroups - 1)
            group_cores = [physical_core(h, k) for k in range(k0, k1 + 1) for h in range(hgroups)]
            h_group_rect_args.append(_rotating_mcast_args_for_cores(device, ttnn.NOC.NOC_0, group_cores)[:4])
    else:
        h_group_rect_args = [[0, 0, 0, 0] for _ in range((kgroups + blk.mgroup_rows - 1) // blk.mgroup_rows)]

    # ---- circular buffers --------------------------------------------------------------
    # Slice accumulators use bf16 to avoid repeated quantization during the scatter chain.
    fmt = {
        "bfp8": ttnn.bfloat8_b,
        "bf16": ttnn.bfloat16,
        "weight": w_dtype,
        "out": out_dtype,
        "u32": ttnn.uint32,
        "x_in": ttnn.bfloat16 if x_is_rm else ttnn.bfloat8_b,
    }
    cbs = [
        _cb_allocation(physical_bytes, all_cores, logical_views, fmt)
        for physical_bytes, logical_views in blk.cb_allocations(
            x_is_rm,
            out_tile,
            idx_page,
            counts_page,
            enable_phase_alias=enable_phase_alias,
        )
    ]

    # ---- compile-time args --------------------------------------------------------------------
    shared = {
        "EMB_T": blk.emb_t,
        "HID_T": blk.hid_t,
        "KR_PAD": blk.kr_pad,
        "HN_PAD": blk.hn_pad,
        "EC_MAX": blk.ec_max,
        "WD_EC_MAX": blk.wd_ec_max,
        "EC_GROUP_MAX": blk.ec_group_max,
        "M_BLOCK": geo.M_BLOCK,
        "HGROUPS": hgroups,
        "KGROUPS": kgroups,
        "NUM_CORES": num_cores,
        "INPUT_FORMAT": input_format,
        "MAILBOX_MAGIC": MAILBOX_MAGIC,
        "M_EFF_MIN": blk.m_eff_min,
        "W_RESIDENT": int(geo.W_RESIDENT),
        # blk.wd_resident, NOT the module default: the budget can turn residency OFF and shrink
        # depth_wd, and a kernel still told "resident" would skip every W_down read after M-block 0
        # while the shrunk CB no longer holds block r in slot r — stale weights, silently.
        "WD_RESIDENT": int(blk.wd_resident),
        "WD_MROW_ROUNDS": int(blk.wd_mrow_rounds and blk.wd_resident),
        "WD_MGROUPS": int(blk.wd_mgroups),
        "MGROUP_ROWS": blk.mgroup_rows,
        "WD_MGROUP_MIN_BLOCKS": geo.WD_MGROUP_MIN_BLOCKS,
        "GU_CHUNKS": blk.gu_chunks,
        "XPRIO": int(geo.XPRIO),
        "WD_SPLIT": blk.wd_split,
        "WG_SHARD_W": wg_shard_w,
        "WD_SHARD_W": wd_shard_w,
        "GATHER_PAGES": blk.gather_pages,
        "PHASE_ALIAS_PAGES": blk.phase_cb_alias_pages(requested_out_tile) if phase_cb_alias else 0,
        "PHASE_CB_ALIAS": int(phase_cb_alias),
        "H_ROUND_NOC1_MASK": geo.H_ROUND_NOC1_MASK,
        "SCATTER_ONE_SIGNAL": int(geo.SCATTER_ONE_SIGNAL),
        "W_TILE_BYTES": w_tile,
        "BFP8_TILE": bfp8_tile,
        "OUT_TILE_BYTES": out_tile,
        "SEM_GO": geo.SEM_GO,
        "SEM_DATA": geo.SEM_DATA,
        "SEM_HSLICE": geo.SEM_HSLICE,
        "SEM_XSTAGED": geo.SEM_XSTAGED,
        "SEM_H_RDY_BASE": geo.SEM_H_RDY_BASE,
        "SEM_H_FREE": geo.SEM_H_FREE,
        "SEM_WDSPLIT": geo.SEM_WDSPLIT,
        "SEM_PHASE_FREE": geo.SEM_PHASE_FREE,
        "SEM_HROW_FREE": geo.SEM_HROW_FREE,
        "DEPTH_H": blk.depth_h,
        "DEPTH_X": blk.depth_x,
        "HACK_AHEAD": blk.hack_ahead,
        "WD_AHEAD": blk.wd_ahead,
        "OUT_SUBBLOCK_H_GU": geo.OUT_SUBBLOCK_H_GU,
        "OUT_SUBBLOCK_H_DN": blk.out_subblock_h_dn,
        "OUT_SUBBLOCK_H_DN_MAX": geo.OUT_SUBBLOCK_H_DN_MAX,
        "HN_BLOCK": blk.hn_block,
        "ELTWISE_BLK": geo.ELTWISE_BLK,
        "DEST_LIMIT": geo.DEST_AUTO_LIMIT_TILES,
        "M_T_MAX": input_m_tiles,
        "LOCAL_EXPERT_ID": local_expert_id,
        "X_PAGE": int(input_tensor.buffer_page_size()),
        "X_SLICE": x_stick_slice,
        "COUNTS_PAGE": max(int(counts.buffer_aligned_page_size()), dram_align),
        "IDX_PAGE": max(int(global_expert_idx_table.buffer_aligned_page_size()), dram_align),
        # NEED_START is the reader's "fetch and publish start[global_id]" flag, so it is on for
        # EITHER side of the fusion — the writer's half arrives only through the mailbox word.
        "NEED_START": int(direct_write or read_x_at_offset),
        "READ_X_AT_OFFSET": int(read_x_at_offset),
        "START_PAGE": start_page,
        "DIRECT_WRITE": int(direct_write),
        "OUT_M_T": int(output_tensor.padded_shape[-2]) // TILE,
    }
    shared.update({n: getattr(geo, n) for n in dir(geo) if n.startswith("CB_")})

    reader_ct = _ct_args("reader", shared)
    reader_ct.extend(x_mcast_ct)
    reader_ct.extend(h_mcast_ct)
    # `start_tensor` is appended LAST, which is what let it be added without shifting a single
    # existing accessor offset in the kernel.
    for t in (input_tensor, w_gate, w_down, counts, global_expert_idx_table, start_tensor):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    writer_ct = _ct_args("writer", shared)
    for t in (w_up, output_tensor, w_down):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    compute_ct = _ct_args("compute", shared)

    # ---- runtime args -------------------------------------------------------------------------
    mailbox_addr = mailbox.buffer_address()
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for kgroup in range(kgroups):
        for hgroup in range(hgroups):
            physical_x, physical_y = physical_core(hgroup, kgroup)
            core = ttnn.CoreCoord(physical_x, physical_y)
            i = kgroup * hgroups + hgroup
            kr, kstart = blk.kr_sizes[kgroup], blk.kr_starts[kgroup]
            hn, hstart = blk.hn_sizes[hgroup], blk.hn_starts[hgroup]
            ec, jstart = blk.ec_sizes[i], blk.ec_starts[i]
            gi = (kgroup % blk.mgroup_rows) * hgroups + hgroup
            ec_group, jstart_group = blk.ec_group_sizes[gi], blk.ec_group_starts[gi]

            args = [
                mailbox_addr,
                input_tensor.buffer_address(),
                w_gate.buffer_address(),
                w_down.buffer_address(),
                counts.buffer_address(),
                global_expert_idx_table.buffer_address(),
                kr,
                kstart,
                hstart,
                hn,
                ec,
                jstart,
                ec_group,
                jstart_group,
                hgroup,
                kgroup,
                start_tensor.buffer_address(),
            ]
            # The whole COLUMN in virtual coordinates: the scatter's peer list (invite fan-out +
            # gather destinations). Row r is at index r on every core in the column, which is what
            # makes "worker r owns tiles [r*a, (r+1)*a)" agree grid-wide.
            for r in range(kgroups):
                args.extend(_virt(device, *physical_core(hgroup, r)))
            x_group_cores = [physical_core(h, kgroup) for h in range(hgroups)]
            args.extend(_rotating_mcast_args_for_cores(device, ttnn.NOC.NOC_0, x_group_cores))
            args.extend(h_mcast_args)
            args.extend(h_group_rect_args[kgroup // blk.mgroup_rows])
            reader_rt[physical_x][physical_y] = args

            wargs = [
                mailbox_addr,
                w_up.buffer_address(),
                output_tensor.buffer_address(),
                w_down.buffer_address(),
                kr,
                kstart,
                hstart,
                hn,
                ec,
                jstart,
                ec_group,
                jstart_group,
                hgroup,
                kgroup,
                hgroup % kgroups,
            ]
            # Full-M eight-round W_down: row y gathers every hidden-column fragment onto the
            # diagonal core (y, y).  Appended before the existing column peer table.
            diagonal_h = kgroup if kgroup < hgroups else 0
            diagonal_k = kgroup if kgroup < hgroups else 0
            wargs.extend(_virt(device, *physical_core(diagonal_h, diagonal_k)))
            for r in range(kgroups):
                wargs.extend(_virt(device, *physical_core(hgroup, r)))
            wargs.extend(h_mcast_noc1_args)
            writer_rt[physical_x][physical_y] = wargs

            compute_rt[physical_x][physical_y] = [mailbox_addr, kr, hn, ec, ec_group, hgroup, kgroup]

    stage_profile_defines = [("MOE_FUSED_SWIGLU_STAGE_PROFILE", "1")] if geo.STAGE_PROFILE else []
    dm_defines = [("H_MCAST_POSTED", "1" if geo.H_MCAST_POSTED else "0"), *stage_profile_defines]
    compute_defines = list(stage_profile_defines)
    if situ_glu:
        compute_defines.append(("SITU_GLU", "1"))

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "moe_fused_swiglu_reader.cpp"),
            core_ranges=all_cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
            defines=dm_defines,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "moe_fused_swiglu_writer.cpp"),
            core_ranges=all_cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
            defines=dm_defines,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "moe_fused_swiglu_compute.cpp"),
            core_ranges=all_cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_kernel_config,
            defines=compute_defines,
        ),
    ]

    # ---- semaphores ----------------------------------------------------------------------------
    semaphores = [
        ttnn.SemaphoreDescriptor(id=sem, core_ranges=all_cores, initial_value=0) for sem in range(geo.SEM_COUNT)
    ]
    if geo.SEM_COUNT > geo.NUM_DEVICE_SEMAPHORES:
        raise RuntimeError(
            f"moe_fused_swiglu: needs {geo.SEM_COUNT} semaphores, device has {geo.NUM_DEVICE_SEMAPHORES} "
            f"(DEPTH_H {geo.DEPTH_H} is what scales this)"
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
