# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe_fused_swiglu — the program descriptor.

Realises the blocking model of ``op_design.md`` §1 on an HGROUPS x KGROUPS worker grid:

  * **Hn** (hidden, gate/up output)      -> split across grid COLUMNS, ``hn_pad`` tiles each
  * **Kg** (emb, gate/up contraction)    -> split across grid ROWS, reduced by the reduce-scatter
  * **Ne** (emb, ``down`` output)        -> split across ALL cores, ``ec`` tiles each
  * **Kh** (hidden, ``down`` contraction)-> sequential per core, ``HGROUPS`` K-blocks
  * **M**  (tokens)                      -> sequential outer loop; the only RUNTIME extent

This module assembles the program. `moe_fused_swiglu_geometry.py` decides what to assemble —
every block factor, buffer depth and tuning constant lives there, with the measurement that set
it recorded in `perf_experiments/DESIGN_NOTES.md`.
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

KERNEL_DIR = Path(__file__).parent / "kernels"

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
        "GU_CHUNKS",
        "XPRIO",
        "HACK_AHEAD",
        "DEPTH_H",
        "WD_SPLIT",
        "WG_SHARD_W",
        "WD_SHARD_W",
        "SCATTER_NOC_SPLIT",
        "GATHER_PAGES",
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
    ),
    "writer": (
        "EMB_T",
        "HID_T",
        "KR_PAD",
        "HN_PAD",
        "EC_MAX",
        "M_BLOCK",
        "HGROUPS",
        "KGROUPS",
        "SEM_GO",
        "SEM_DATA",
        "SEM_HSLICE",
        "SEM_XSTAGED",
        "SEM_WDSPLIT",
        "W_TILE_BYTES",
        "BFP8_TILE",
        "MAILBOX_MAGIC",
        "M_EFF_MIN",
        "W_RESIDENT",
        "WD_RESIDENT",
        "GU_CHUNKS",
        "XPRIO",
        "WD_SPLIT",
        "WG_SHARD_W",
        "WD_SHARD_W",
        "SCATTER_NOC_SPLIT",
        "CB_W_UP",
        "CB_W_DOWN",
        "CB_OUT_TILES",
        "CB_GATE_ACC",
        "CB_UP_ACC",
        "CB_GATHER_GATE",
        "CB_GATHER_UP",
        "CB_H_SLICE",
        "CB_H_LOCAL",
    ),
    "compute": (
        "M_BLOCK",
        "KR_PAD",
        "HN_PAD",
        "EC_MAX",
        "HGROUPS",
        "KGROUPS",
        "HID_T",
        "INPUT_FORMAT",
        "OUT_SUBBLOCK_H_GU",
        "OUT_SUBBLOCK_H_DN",
        "MAILBOX_MAGIC",
        "M_EFF_MIN",
        "HN_BLOCK",
        "GU_CHUNKS",
        "ELTWISE_BLK",
        "DEST_LIMIT",
        "GATHER_PAGES",
        "CB_X_IN",
        "CB_X_TILES",
        "CB_X_STAGE",
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
        "MY_COL",
        "MY_ROW",
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
        "MY_COL",
        "MY_ROW",
        "ROOT_ROW",
    ),
    "compute": ("MAILBOX", "KR", "HN", "EC", "MY_COL", "MY_ROW"),
}


# --------------------------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------------------------
def worker_grid(device, core_grid=None):
    """The (HGROUPS, KGROUPS) this op will use — the ONE definition, shared by
    `create_program_descriptor` and by `weight_memory_configs`, so a caller's shard width cannot
    drift from the op's own hidden split.

    `core_grid` is a (x, y) pair or a ttnn.CoreCoord; it is CLAMPED to the device grid rather than
    trusted. The op runs on a full rectangle anchored at (0, 0) — every collective's multicast rect
    derives from that — so a harvested or non-contiguous CoreRangeSet is not expressible here.
    """
    grid = device.compute_with_storage_grid_size()
    hgroups, kgroups = int(grid.x), int(grid.y)
    if core_grid is not None:
        gx, gy = (
            (int(core_grid.x), int(core_grid.y)) if hasattr(core_grid, "x") else (int(core_grid[0]), int(core_grid[1]))
        )
        if gx < 1 or gy < 1:
            raise ValueError(f"moe_fused_swiglu: core_grid must be positive, got {gx}x{gy}")
        hgroups, kgroups = min(gx, hgroups), min(gy, kgroups)
    return hgroups, kgroups


def nd_shard_n_tiles(t):
    """N-axis TILES per DRAM ND shard of `t`, or 0 if no contiguous run can be proven.

    This is the ONE place the op learns a weight's placement; everything downstream is a run
    length. Inside one shard, consecutive N pages are physically contiguous in one bank
    (`TensorAccessor::get_bank_and_offset`), so a core's slice decomposes into runs that issue as
    single NoC transactions. 0 means one transaction per tile — correct, and slower.

    Interleaved deliberately returns 0. Its contiguity is the stride-num_banks bank run, and
    exploiting that measured a net negative; see DESIGN_NOTES.md.
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


def weight_memory_configs(device, emb, hidden, core_grid=None, shard_height_tiles=1):
    """The op's PREFERRED weight placement, as `(gate_up, down)` memory configs.

    A pure function of K, N and the grid — never of the runtime token count. Each shard is exactly
    the N slice ONE core consumes for ONE K-row, so that read is a single transaction:
      * gate/up — `hn_pad` tiles, the hidden split across grid COLUMNS
      * down    — `ec_max` tiles, the emb-output split across ALL cores

    ONE TILE-ROW TALL by default, and that is measured rather than preferred: a core pinned to a
    single DRAM bank saturates near 30 GB/s, while the same bytes with the bank rotating across K
    reach ~370 GB/s. `shard_height_tiles` raises it for a bandwidth probe — the kernel is
    height-agnostic by construction, so any height is correct, just possibly slower.

    Placement is the CALLER's to choose. The op reads whatever it is handed and takes the
    coalesced path whenever `nd_shard_n_tiles` can prove a run.
    """
    hgroups, kgroups = worker_grid(device, core_grid)
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

    return mc(blk.hn_pad), mc(blk.ec_max)


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


def _cb(index, core_ranges, num_pages, page_size, data_format):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _ct_args(kernel, values):
    """Positional vector in `KERNEL_CT_ORDER[kernel]`, with every name required exactly once."""
    order = KERNEL_CT_ORDER[kernel]
    missing = [n for n in order if n not in values]
    if missing:
        raise RuntimeError(f"moe_fused_swiglu: {kernel} compile-time args missing {missing}")
    return [int(values[n]) for n in order]


def _ablations(selected, allowed):
    return [a for a in selected.split("+") if a in allowed]


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
):
    device = input_tensor.device()
    hgroups, kgroups = worker_grid(device, core_grid)
    num_cores = hgroups * kgroups
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hgroups - 1, kgroups - 1))])

    emb = int(input_tensor.shape[-1])
    capacity = int(input_tensor.shape[-2])
    hidden = int(w_gate.shape[-1])
    blk = Blocking(hgroups, kgroups, emb, hidden, input_m_tiles)

    # ---- dtypes -----------------------------------------------------------------------------
    # The weight stride and CB format both come from the tensor. All three weights share a dtype
    # (validate() enforces it), so one width serves the gate/up and down streams alike.
    w_dtype = w_gate.dtype
    w_tile = ttnn.tile_size(w_dtype)
    bfp8_tile = ttnn.tile_size(ttnn.bfloat8_b)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    dram_align = ttnn.get_dram_alignment()
    out_dtype = output_tensor.dtype
    out_tile = ttnn.tile_size(out_dtype)

    x_is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    input_format = 0 if x_is_rm else 1
    x_stick_slice = blk.kr_pad * TILE * input_tensor.element_size() if x_is_rm else bfp8_tile

    # ---- weight placement -------------------------------------------------------------------
    # gate/up resolve to ONE width: the reader and the writer read the same [k, n] slice of two
    # identically-shaped tensors, and a disagreement would give the up matmul a different
    # coalescing from the gate's.
    wg_shard_w = min(nd_shard_n_tiles(w_gate), nd_shard_n_tiles(w_up))
    wd_shard_w = nd_shard_n_tiles(w_down)

    # ---- collectives ------------------------------------------------------------------------
    # DataReadySignal::Flag, not Counter: the Counter signal is an atomic on a different command
    # buffer, so it cannot terminate the data multicast's NOC_CMD_VC_LINKED chain and the sender
    # hangs. The h all-gather works around Flag's per-round reset chain with per-slot cells of its
    # own (SEM_H_RDY_BASE); see the reader.
    handshake = geo.ABLATE != "no_handshake"
    flag = ttnn._ttnn.mcast_host.McastDataReady.Flag
    x_mcast = ttnn.Mcast1D(
        device,
        all_cores,
        ttnn.Mcast1DShape.PerRow,
        0,
        ttnn.McastConfig(
            noc=ttnn.NOC.NOC_0, handshake=handshake, data_ready=flag, rotating_sender=True, base_sem_id=geo.SEM_X_BASE
        ),
    )
    h_mcast = ttnn.Mcast2D(
        device,
        all_cores,
        ttnn.CoreCoord(0, 0),
        ttnn.McastConfig(
            noc=ttnn.NOC.NOC_0, handshake=handshake, data_ready=flag, rotating_sender=True, base_sem_id=geo.SEM_H_BASE
        ),
    )
    if x_mcast.num_senders() != hgroups or h_mcast.num_senders() != num_cores:
        raise RuntimeError(
            f"moe_fused_swiglu: mcast sender counts {x_mcast.num_senders()}/{h_mcast.num_senders()} "
            f"do not match the {hgroups}x{kgroups} grid"
        )

    # ---- circular buffers --------------------------------------------------------------------
    pages = blk.cb_pages(x_is_rm)
    # The slice accumulators are bf16, NOT bfp8, and that is a measured CORRECTNESS choice: the
    # scatter chains KGROUPS contributors through one accumulator and bfp8's pack rounding is a
    # biased half-LSB, so the error grows linearly in the chain. bf16 removes the per-step
    # quantisation entirely (max relative error 0.0580 -> 0.0204, matching the tree's).
    cbs = [
        _cb(geo.CB_X_IN, all_cores, pages["x_in"], x_stick_slice, ttnn.bfloat16 if x_is_rm else ttnn.bfloat8_b),
        _cb(geo.CB_X_TILES, all_cores, pages["x_tiles"], bfp8_tile, ttnn.bfloat8_b),
        _cb(geo.CB_X_STAGE, all_cores, pages["x_stage"], bfp8_tile, ttnn.bfloat8_b),
        _cb(geo.CB_W_GATE, all_cores, pages["w_gu"], w_tile, w_dtype),
        _cb(geo.CB_W_UP, all_cores, pages["w_gu"], w_tile, w_dtype),
        _cb(geo.CB_W_DOWN, all_cores, pages["w_down"], w_tile, w_dtype),
        _cb(geo.CB_H, all_cores, pages["h"], bfp8_tile, ttnn.bfloat8_b),
        _cb(
            geo.CB_IDX_SCRATCH,
            all_cores,
            1,
            max(int(global_expert_idx_table.buffer_aligned_page_size()), dram_align),
            ttnn.uint32,
        ),
        _cb(geo.CB_COUNTS_SCRATCH, all_cores, 1, max(int(counts.buffer_aligned_page_size()), dram_align), ttnn.uint32),
        _cb(geo.CB_GATHER_GATE, all_cores, pages["gather"], bfp8_tile, ttnn.bfloat8_b),
        _cb(geo.CB_GATHER_UP, all_cores, pages["gather"], bfp8_tile, ttnn.bfloat8_b),
        _cb(geo.CB_SLICE_GATE, all_cores, pages["slice"], bf16_tile, ttnn.bfloat16),
        _cb(geo.CB_SLICE_UP, all_cores, pages["slice"], bf16_tile, ttnn.bfloat16),
        _cb(geo.CB_H_SLICE, all_cores, pages["slice"], bfp8_tile, ttnn.bfloat8_b),
        _cb(geo.CB_OUT_TILES, all_cores, geo.DEPTH_OUT * pages["out_block"], out_tile, out_dtype),
        _cb(geo.CB_GATE_ACC, all_cores, pages["gu_block"], bfp8_tile, ttnn.bfloat8_b),
        _cb(geo.CB_UP_ACC, all_cores, pages["gu_block"], bfp8_tile, ttnn.bfloat8_b),
        _cb(geo.CB_GATE_SILU, all_cores, pages["slice"], bf16_tile, ttnn.bfloat16),
        _cb(geo.CB_H_LOCAL, all_cores, pages["gu_block"], bfp8_tile, ttnn.bfloat8_b),
        _cb(geo.CB_OUT_INTERM, all_cores, pages["out_block"], bf16_tile, ttnn.bfloat16),
    ]

    # ---- compile-time args --------------------------------------------------------------------
    shared = {
        "EMB_T": blk.emb_t,
        "HID_T": blk.hid_t,
        "KR_PAD": blk.kr_pad,
        "HN_PAD": blk.hn_pad,
        "EC_MAX": blk.ec_max,
        "M_BLOCK": geo.M_BLOCK,
        "HGROUPS": hgroups,
        "KGROUPS": kgroups,
        "NUM_CORES": num_cores,
        "INPUT_FORMAT": input_format,
        "MAILBOX_MAGIC": MAILBOX_MAGIC,
        "M_EFF_MIN": blk.m_eff_min,
        "W_RESIDENT": int(geo.W_RESIDENT),
        "WD_RESIDENT": int(geo.WD_RESIDENT),
        "GU_CHUNKS": blk.gu_chunks,
        "XPRIO": int(geo.XPRIO),
        "WD_SPLIT": blk.wd_split,
        "WG_SHARD_W": wg_shard_w,
        "WD_SHARD_W": wd_shard_w,
        "SCATTER_NOC_SPLIT": int(geo.SCATTER_NOC_SPLIT),
        "GATHER_PAGES": blk.gather_pages,
        "W_TILE_BYTES": w_tile,
        "BFP8_TILE": bfp8_tile,
        "SEM_GO": geo.SEM_GO,
        "SEM_DATA": geo.SEM_DATA,
        "SEM_HSLICE": geo.SEM_HSLICE,
        "SEM_XSTAGED": geo.SEM_XSTAGED,
        "SEM_H_RDY_BASE": geo.SEM_H_RDY_BASE,
        "SEM_H_FREE": geo.SEM_H_FREE,
        "SEM_WDSPLIT": geo.SEM_WDSPLIT,
        "DEPTH_H": geo.DEPTH_H,
        "HACK_AHEAD": blk.hack_ahead,
        "WD_AHEAD": blk.wd_ahead,
        "OUT_SUBBLOCK_H_GU": geo.OUT_SUBBLOCK_H_GU,
        "OUT_SUBBLOCK_H_DN": blk.out_subblock_h_dn,
        "HN_BLOCK": blk.hn_block,
        "ELTWISE_BLK": geo.ELTWISE_BLK,
        "DEST_LIMIT": geo.DEST_AUTO_LIMIT_TILES,
        "M_T_MAX": input_m_tiles,
        "LOCAL_EXPERT_ID": local_expert_id,
        "X_PAGE": int(input_tensor.buffer_page_size()),
        "X_SLICE": x_stick_slice,
        "COUNTS_PAGE": max(int(counts.buffer_aligned_page_size()), dram_align),
        "IDX_PAGE": max(int(global_expert_idx_table.buffer_aligned_page_size()), dram_align),
    }
    shared.update({n: getattr(geo, n) for n in dir(geo) if n.startswith("CB_")})

    reader_ct = _ct_args("reader", shared)
    reader_ct.extend(x_mcast.compile_time_args())
    reader_ct.extend(h_mcast.compile_time_args())
    for t in (input_tensor, w_gate, w_down, counts, global_expert_idx_table):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    writer_ct = _ct_args("writer", shared)
    for t in (w_up, output_tensor, w_down):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    compute_ct = _ct_args("compute", shared)

    # ---- runtime args -------------------------------------------------------------------------
    mailbox_addr = mailbox.buffer_address()
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for y in range(kgroups):
        for x in range(hgroups):
            core = ttnn.CoreCoord(x, y)
            i = y * hgroups + x
            kr, kstart = blk.kr_sizes[y], blk.kr_starts[y]
            hn, hstart = blk.hn_sizes[x], x * blk.hn_pad
            ec, jstart = blk.ec_sizes[i], blk.ec_starts[i]

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
                x,
                y,
            ]
            # The whole COLUMN in virtual coordinates: the scatter's peer list (invite fan-out +
            # gather destinations). Row r is at index r on every core in the column, which is what
            # makes "worker r owns tiles [r*a, (r+1)*a)" agree grid-wide.
            for r in range(kgroups):
                args.extend(_virt(device, x, r))
            args.extend(x_mcast.runtime_args(core))
            args.extend(h_mcast.runtime_args(core))
            reader_rt[x][y] = args

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
                x,
                y,
                x % kgroups,
            ]
            for r in range(kgroups):
                wargs.extend(_virt(device, x, r))
            writer_rt[x][y] = wargs

            compute_rt[x][y] = [mailbox_addr, kr, hn, ec, x, y]

    # ---- defines: the ablation hook only -------------------------------------------------------
    dm_defines = [("ABLATE_" + a.upper(), "1") for a in _ablations(geo.ABLATE, geo.DM_ABLATIONS)]
    compute_defines = []
    if "skip_compute" in geo.ABLATE.split("+"):
        compute_defines.append(("SKIP_COMPUTE", "1"))
    # `SKIP_COMPUTE` elides only the inner matmul LLK; every eltwise_chain in the TU keeps running.
    # `skip_eltwise` closes that hole — it keeps every CB cycle and DEST sync and drops only math.
    if "skip_eltwise" in geo.ABLATE.split("+"):
        compute_defines.append(("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1"))

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
    # Every one is monotone and zero-init; none is reset within a dispatch. See RACE_AUDIT.md.
    semaphores = list(x_mcast.owned_semaphores()) + list(h_mcast.owned_semaphores())
    for sem in (geo.SEM_GO, geo.SEM_DATA, geo.SEM_HSLICE, geo.SEM_XSTAGED, geo.SEM_WDSPLIT):
        semaphores.append(ttnn.SemaphoreDescriptor(id=sem, core_ranges=all_cores, initial_value=0))
    for s in range(geo.DEPTH_H):
        semaphores.append(ttnn.SemaphoreDescriptor(id=geo.SEM_H_RDY_BASE + s, core_ranges=all_cores, initial_value=0))
    semaphores.append(ttnn.SemaphoreDescriptor(id=geo.SEM_H_FREE, core_ranges=all_cores, initial_value=0))
    if geo.SEM_COUNT > geo.NUM_DEVICE_SEMAPHORES:
        raise RuntimeError(
            f"moe_fused_swiglu: needs {geo.SEM_COUNT} semaphores, device has {geo.NUM_DEVICE_SEMAPHORES} "
            f"(DEPTH_H {geo.DEPTH_H} is what scales this)"
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
