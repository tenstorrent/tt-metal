# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""moe_fused_swiglu — ProgramDescriptor.

Realises the Blocking Model of ``op_design.md`` §1 on the full HGROUPS x KGROUPS worker grid:

  * **Hn** (hidden, gate/up output axis) is split across the grid COLUMNS  -> ``HN_PAD``
  * **Kg** (emb, gate/up contraction) is split across the grid ROWS        -> ``KR_PAD`` / ``kr(y)``
  * **Ne** (emb, ``down`` output axis) is split across ALL cores           -> ``ec(i)``
  * **Kh** (hidden, ``down`` contraction) stays sequential per core        -> ``HGROUPS`` K-blocks
  * **M**  (tokens) is the sequential outer loop                          -> ``M_BLOCK``

The dependent axis (Kg) is combined by a binary reduce tree down each column; the two
reuse-shared operands are broadcast (``x`` along the row, ``h`` across the whole grid).

EVERY block factor, buffer depth and core assignment below is a named parameter with ONE
definition. Every CB page count, loop trip count and grid formula is derived from those
parameters — none is a whole-op dimension (``EMB_T``, ``HID_T``, ``capacity``) and none is a
magic literal.
"""

import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
HIDDEN = 2048

# ---------------------------------------------------------------------------
# BLOCKING KNOBS — the single source of truth. Each is defined exactly once.
# ---------------------------------------------------------------------------
#: Token tile-rows per M-block. The sequential outer loop is ceil(M_t / M_BLOCK); the graded
#: counts 128 and 256 fit ONE block, so the weight stream is read exactly once for them.
#: Raising it (to fold count=512 into one block as well) is a knob turn — it costs
#: (M_BLOCK - 8) * KR_PAD tiles of resident-x L1 plus the same scaling on every per-block CB.
M_BLOCK = 8

#: Token tiles per output sub-block. DEST budget is out_subblock_h * out_subblock_w <=
#: DEST_AUTO_LIMIT (8 at half sync with fp32_dest_acc_en=False), and out_subblock_w carries the
#: whole HN_PAD / ec width here, so the height factor is 1.
OUT_SUBBLOCK_H = 1

#: K tiles per gate/up matmul K-block, expressed as a fraction of the per-row K extent.
#: 1 == the coarsest correct block (num_k_blocks == 1), which is what lets the gate and the up
#: matmul share ONE resident `x` block (op_design.md §6 "the cb_x_tiles-consumed-twice
#: contract"). Splitting it further would need the documented second-CB copy of x.
KB1_FRACTION = 1

#: Buffer depths (per streaming CB).
DEPTH_W = 2  # weight CBs: overlap the next K-block's DRAM read with compute
DEPTH_H = 3  # h all-gather: 3 so a late round's producer is not flow-controlled by itself
DEPTH_OUT = 2
DEPTH_XSTAGE = 1  # tilized x staging slots (a core injects <= ceil(M_BLOCK/HGROUPS) rows/block)
XSTICK_ROWS = 1  # tile-rows of row-major x sticks held in flight

#: Read-coalescing knob: max BANK-CONTIGUOUS weight/output tiles fetched per NoC transaction.
#: 1 reproduces the naive one-transaction-per-tile read (the ablation baseline).
#: Overridable for `/perf-measure` A/B via MOE_SWIGLU_WRUN.
WRUN = int(os.environ.get("MOE_SWIGLU_WRUN", 8))

#: `/perf-measure` ablation hook (payload stubbed, ALL synchronisation scaffolding intact).
#: MOE_SWIGLU_ABLATE=skip_compute defines SKIP_COMPUTE in the compute TU, which drops the inner
#: matmul LLK call while keeping every CB wait/push, reload and L1_ACC toggle — the documented
#: way to separate the dataflow ceiling from the compute ceiling. NOT a correctness mode.
ABLATE = os.environ.get("MOE_SWIGLU_ABLATE", "")

#: W_down prefetch depth, in phase-2 K-blocks kept in flight ahead of the round that consumes
#: them. Clamped to [1, HGROUPS].
#:
#: MEASURED (emb 7168, count 256, bf16_rm): 1 -> 227.8 us, 4 -> 228.7 us, 11 -> 240.1 us. Deeper
#: prefetch does NOT help, which is itself the finding: the phase-2 weight stream is not
#: DRAM-latency bound. Parked at 1 (byte-identical to the naive per-round read) and kept as a
#: live knob — it becomes worth turning only once the h all-gather stops being the critical path.
WD_AHEAD = int(os.environ.get("MOE_SWIGLU_WD_AHEAD", 1))

#: Reduce-tree fan-in cap (>= ceil(log2(KGROUPS)) for the binary tree below).
MAX_CHILDREN = 5

#: Mailbox handshake word — the reader publishes {count, M_t, m_blocks} plus this flag.
MAILBOX_MAGIC = 0xC0FFEE01
MAILBOX_WORDS = 16

# ---------------------------------------------------------------------------
# Circular buffers (semantic names; the numeric slot is only the buffer index)
# ---------------------------------------------------------------------------
CB_X_IN = 0  # row-major x stick slices (bf16) or bfp8 tiles
CB_X_TILES = 1  # resident bfp8 in0 block, filled by the row multicast
CB_X_STAGE = 2  # tilized x tile-row awaiting its multicast turn
CB_W_GATE = 3
CB_W_UP = 4
CB_W_DOWN = 5
CB_REDUCE_GATE_IN = 6  # incoming child partials (gate)
CB_REDUCE_UP_IN = 7  # incoming child partials (up)
CB_H = 8  # gathered h, one phase-2 K-block per round
CB_IDX_SCRATCH = 9
CB_COUNTS_SCRATCH = 10
CB_OUT_TILES = 16
CB_GATE_ACC = 24  # gate partial accumulator (matmul out + in-place reduce adds)
CB_UP_ACC = 25
CB_GATE_SEND = 26  # partials handed to the writer for the unicast to the tree parent
CB_UP_SEND = 27
CB_GATE_SILU = 28  # root only: SiLU(sum(gate))
CB_H_LOCAL = 29  # root only: this column's h slice, awaiting its all-gather round
CB_OUT_INTERM = 30  # phase-2 packer-L1 accumulation region

# ---------------------------------------------------------------------------
# Semaphores
# ---------------------------------------------------------------------------
SEM_X_BASE = 0  # x row multicast (data_ready, consumer_ready)
SEM_H_BASE = 2  # h all-gather   (data_ready, consumer_ready)
SEM_GO = 4  # reduce tree: parent -> child "your slot is free, send"
SEM_DATA = 5  # reduce tree: child -> parent "partials landed"


def _split(total, groups):
    """`base + (i < rem)` split — alignment-aware, no floor on a tile count."""
    base, rem = total // groups, total % groups
    sizes = [base + (1 if i < rem else 0) for i in range(groups)]
    starts, acc = [], 0
    for s in sizes:
        starts.append(acc)
        acc += s
    return sizes, starts


def _reduce_tree(kgroups, hgroups):
    """Binary reduce tree per grid column.

    Column ``x``'s root is row ``x % kgroups`` so the 13 roots (which additionally carry the
    SwiGLU and the h-multicast injection) spread over all rows. Relative index
    ``r = (y - root) % kgroups``; node ``r`` receives from ``r + 2^(l-1)`` at level ``l`` when
    ``r % 2^l == 0``, and sends to ``r - lowbit(r)``. Depth = ceil(log2(kgroups)).
    """
    info = {}
    for x in range(hgroups):
        root_y = x % kgroups
        for y in range(kgroups):
            r = (y - root_y) % kgroups
            children = []
            s = 1
            while s < kgroups:
                if r % (2 * s) == 0 and r + s < kgroups:
                    children.append((x, (root_y + r + s) % kgroups))
                s *= 2
            parent = None
            if r != 0:
                low = r & (-r)
                parent = (x, (root_y + r - low) % kgroups)
            info[(x, y)] = {"is_root": r == 0, "parent": parent, "children": children}
    return info


def _virt(device, x, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return int(c.x), int(c.y)


def _cb(index, core_ranges, num_pages, page_size, data_format):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def make_mailbox(device, num_cores):
    """Zeroed L1 scratch, one page per L1 bank, used as the token-count mailbox.

    The reader publishes the DEVICE-resident count here; compute (all three TRISCs) and the
    writer spin on the magic word. Zeroed host-side so a stale magic from a previous dispatch
    cannot be mistaken for a fresh publish.
    """
    import torch  # local: ttnn must not carry a global torch import

    return ttnn.from_torch(
        torch.zeros((num_cores, MAILBOX_WORDS), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


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
):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()

    # ---- grid / core assignment (every core count derives from the device grid) ----
    HGROUPS = int(grid.x)  # hidden groups == grid columns
    KGROUPS = int(grid.y)  # emb-contraction groups == grid rows
    num_cores = HGROUPS * KGROUPS
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(HGROUPS - 1, KGROUPS - 1))])
    if KGROUPS < 2:
        raise RuntimeError(f"moe_fused_swiglu needs a grid at least 2 rows tall (got {HGROUPS}x{KGROUPS})")

    emb = int(input_tensor.shape[-1])
    capacity = int(input_tensor.shape[-2])
    hidden = int(w_gate.shape[-1])
    EMB_T = emb // TILE
    HID_T = hidden // TILE
    M_T_MAX = input_m_tiles

    # ---- block factors, derived from the knobs ----
    kr_sizes, kr_starts = _split(EMB_T, KGROUPS)
    KR_PAD = max(kr_sizes)  # uniform in0 K stride; per-row kr shrinks the FMA loop
    KB1 = max(1, (KR_PAD * KB1_FRACTION))
    num_k_blocks_gu = (KR_PAD + KB1 - 1) // KB1
    if num_k_blocks_gu != 1:
        raise RuntimeError(
            "moe_fused_swiglu: gate/up K-blocking > 1 needs the second-CB copy of the resident x "
            "block (op_design.md §6); set KB1_FRACTION = 1."
        )

    HN_PAD = (HID_T + HGROUPS - 1) // HGROUPS  # uniform hidden width per column group
    hn_sizes = [max(0, min(HN_PAD, HID_T - x * HN_PAD)) for x in range(HGROUPS)]
    if min(hn_sizes) == 0:
        raise RuntimeError(f"moe_fused_swiglu: hidden {hidden} cannot fill {HGROUPS} column groups")

    wd_ahead = max(1, min(WD_AHEAD, HGROUPS))
    ec_sizes, ec_starts = _split(EMB_T, num_cores)
    # EC_MAX is the phase-2 N *stride*: every phase-2 CB reserves/pushes in EC_MAX-wide units so
    # its page count is a multiple of the increment (a CB whose total is not a multiple of its
    # reserve granularity walks its FIFO pointer off the end). Cores with ec < EC_MAX leave the
    # tail columns unread — out_subblock_w stays `ec`, so no extra FMA work is done.
    EC_MAX = max(ec_sizes)

    # DEST budget: out_subblock_h * out_subblock_w <= DEST_AUTO_LIMIT. out_subblock_w carries the
    # full HN_PAD (gate/up) and ec (down) widths, so both must fit.
    dest_limit = 8
    if OUT_SUBBLOCK_H * HN_PAD > dest_limit or OUT_SUBBLOCK_H * EC_MAX > dest_limit:
        raise RuntimeError(
            f"moe_fused_swiglu: out sub-block {OUT_SUBBLOCK_H}x(max {HN_PAD},{EC_MAX}) exceeds the "
            f"DEST budget of {dest_limit} tiles"
        )
    if M_BLOCK % OUT_SUBBLOCK_H != 0:
        raise RuntimeError(f"moe_fused_swiglu: M_BLOCK {M_BLOCK} must be a multiple of {OUT_SUBBLOCK_H}")

    # ---- DRAM bank-run coalescing precondition (op_design.md §1.5) ----
    dram_align = ttnn.get_dram_alignment()
    bfp4_tile = ttnn.tile_size(ttnn.bfloat4_b)
    bfp8_tile = ttnn.tile_size(ttnn.bfloat8_b)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    try:
        num_banks = int(ttnn._ttnn.device.GetMemoryView(device, ttnn.BufferType.DRAM).num_banks)
    except Exception:  # pragma: no cover - defensive
        num_banks = 0
    remap = int(
        num_banks > 1
        and WRUN > 1
        and HID_T % num_banks == 0
        and EMB_T % num_banks == 0
        and bfp4_tile % dram_align == 0
        and bfp8_tile % dram_align == 0
    )
    if not remap:
        num_banks = max(num_banks, 1)

    # ---- reduce tree ----
    tree = _reduce_tree(KGROUPS, HGROUPS)

    # ---- collectives: emit the mcast wire (Mcast1D / Mcast2D own the coord + rect math) ----
    # Reader runs on NOC_0 (ReaderDataMovementConfig -> preferred_noc_for_dram_read).
    # DataReadySignal: Flag, NOT the Counter op_design.md §4.2/§4.4 asks for. Counter was tried
    # and HANGS on both families: mcast_pipe.inl:153 hands `inc_multicast` the LOOPBACK
    # (INCLUDE-source) fan-out while the atomic multicast is EXCLUDE-source, so its atomic
    # barrier waits forever for an ack from a destination that was never addressed. Both mcasts
    # here loopback (src cb != dst cb), so both trip it. Flag is correct but forces the sender of
    # round r+1 to wait for every receiver to reset round r's flag — see the perf notes.
    # `handshake` is the receiver->sender "my cb slot is free" ack, i.e. the CB flow control, and
    # stays on (MOE_SWIGLU_ABLATE=no_handshake drops it for MEASUREMENT only — not correct).
    handshake = ABLATE != "no_handshake"
    counter = ttnn._ttnn.mcast_host.McastDataReady.Flag
    x_mcast = ttnn.Mcast1D(
        device,
        all_cores,
        ttnn.Mcast1DShape.PerRow,
        0,
        ttnn.McastConfig(
            noc=ttnn.NOC.NOC_0,
            handshake=handshake,
            data_ready=counter,
            rotating_sender=True,
            base_sem_id=SEM_X_BASE,
        ),
    )
    h_mcast = ttnn.Mcast2D(
        device,
        all_cores,
        ttnn.CoreCoord(0, 0),
        ttnn.McastConfig(
            noc=ttnn.NOC.NOC_0,
            handshake=handshake,
            data_ready=counter,
            rotating_sender=True,
            base_sem_id=SEM_H_BASE,
        ),
    )
    assert x_mcast.num_senders() == HGROUPS, x_mcast.num_senders()
    assert h_mcast.num_senders() == num_cores, h_mcast.num_senders()

    # ---- page sizes ----
    x_is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    input_format = 0 if x_is_rm else 1
    x_stick_slice = KR_PAD * TILE * input_tensor.element_size() if x_is_rm else bfp8_tile
    x_page = int(input_tensor.buffer_page_size())
    counts_page = int(counts.buffer_aligned_page_size())
    idx_page = int(global_expert_idx_table.buffer_aligned_page_size())

    # ---- CB page counts: functions of the knobs only ----
    n_x_tiles = M_BLOCK * KR_PAD  # ONE slot -> identical mcast landing address on every core
    n_gu_block = M_BLOCK * HN_PAD
    n_w_gu = KR_PAD * HN_PAD  # one gate/up K-block (num_k_blocks == 1)
    # cb_w_down spans ALL HGROUPS phase-2 K-blocks: the per-M-block pushes then sum to exactly
    # the CB size (the cb_api wrap rule) whatever wd_ahead is, so wd_ahead stays a free knob.
    n_w_down = HGROUPS * HN_PAD * EC_MAX
    n_out_block = M_BLOCK * EC_MAX

    cbs = [
        _cb(
            CB_X_IN,
            all_cores,
            XSTICK_ROWS * TILE if x_is_rm else KR_PAD,
            x_stick_slice,
            ttnn.bfloat16 if x_is_rm else ttnn.bfloat8_b,
        ),
        _cb(CB_X_TILES, all_cores, n_x_tiles, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_X_STAGE, all_cores, DEPTH_XSTAGE * KR_PAD, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_W_GATE, all_cores, DEPTH_W * n_w_gu, bfp4_tile, ttnn.bfloat4_b),
        _cb(CB_W_UP, all_cores, DEPTH_W * n_w_gu, bfp4_tile, ttnn.bfloat4_b),
        _cb(CB_W_DOWN, all_cores, n_w_down, bfp4_tile, ttnn.bfloat4_b),
        _cb(CB_REDUCE_GATE_IN, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_REDUCE_UP_IN, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_H, all_cores, DEPTH_H * n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_IDX_SCRATCH, all_cores, 1, max(idx_page, dram_align), ttnn.uint32),
        _cb(CB_COUNTS_SCRATCH, all_cores, 1, max(counts_page, dram_align), ttnn.uint32),
        _cb(CB_OUT_TILES, all_cores, DEPTH_OUT * n_out_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_GATE_ACC, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_UP_ACC, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_GATE_SEND, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_UP_SEND, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_GATE_SILU, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_H_LOCAL, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_OUT_INTERM, all_cores, n_out_block, bf16_tile, ttnn.bfloat16),
    ]

    # -----------------------------------------------------------------------
    # Reader
    # -----------------------------------------------------------------------
    reader_ct = [
        input_format,
        M_T_MAX,
        local_expert_id,
        EMB_T,
        HID_T,
        KR_PAD,
        HN_PAD,
        EC_MAX,
        M_BLOCK,
        HGROUPS,
        KGROUPS,
        num_banks,
        WRUN,
        SEM_GO,
        SEM_DATA,
        x_page,
        x_stick_slice,
        max(counts_page, dram_align),
        max(idx_page, dram_align),
        bfp4_tile,
        bfp8_tile,
        MAX_CHILDREN,
        remap,
        MAILBOX_MAGIC,
        wd_ahead,
        CB_X_IN,
        CB_X_TILES,
        CB_X_STAGE,
        CB_W_GATE,
        CB_W_DOWN,
        CB_REDUCE_GATE_IN,
        CB_REDUCE_UP_IN,
        CB_H,
        CB_H_LOCAL,
        CB_IDX_SCRATCH,
        CB_COUNTS_SCRATCH,
    ]
    reader_ct.extend(x_mcast.compile_time_args())
    reader_ct.extend(h_mcast.compile_time_args())
    for t in (input_tensor, w_gate, w_down, counts, global_expert_idx_table):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    writer_ct = [
        EMB_T,
        HID_T,
        KR_PAD,
        HN_PAD,
        EC_MAX,
        M_BLOCK,
        HGROUPS,
        KGROUPS,
        num_banks,
        WRUN,
        SEM_GO,
        SEM_DATA,
        bfp4_tile,
        bfp8_tile,
        remap,
        MAILBOX_MAGIC,
        CB_W_UP,
        CB_OUT_TILES,
        CB_GATE_SEND,
        CB_UP_SEND,
        CB_REDUCE_GATE_IN,
        CB_REDUCE_UP_IN,
    ]
    for t in (w_up, output_tensor):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    compute_ct = [
        M_BLOCK,
        KR_PAD,
        HN_PAD,
        EC_MAX,
        HGROUPS,
        HID_T,
        input_format,
        OUT_SUBBLOCK_H,
        MAILBOX_MAGIC,
        CB_X_IN,
        CB_X_TILES,
        CB_X_STAGE,
        CB_W_GATE,
        CB_W_UP,
        CB_W_DOWN,
        CB_GATE_ACC,
        CB_UP_ACC,
        CB_GATE_SEND,
        CB_UP_SEND,
        CB_GATE_SILU,
        CB_REDUCE_GATE_IN,
        CB_REDUCE_UP_IN,
        CB_H_LOCAL,
        CB_H,
        CB_OUT_INTERM,
        CB_OUT_TILES,
    ]

    mailbox_addr = mailbox.buffer_address()
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    for y in range(KGROUPS):
        for x in range(HGROUPS):
            core = ttnn.CoreCoord(x, y)
            i = y * HGROUPS + x
            node = tree[(x, y)]
            kr, kstart = kr_sizes[y], kr_starts[y]
            hn, hstart = hn_sizes[x], x * HN_PAD
            ec, jstart = ec_sizes[i], ec_starts[i]
            # tile-rows this core injects into the row multicast
            n_inject = len([t for t in range(M_BLOCK) if t % HGROUPS == x])

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
                1 if node["is_root"] else 0,
                len(node["children"]),
                x,
            ]
            for c in range(MAX_CHILDREN):
                if c < len(node["children"]):
                    cx, cy = _virt(device, *node["children"][c])
                else:
                    cx, cy = 0, 0
                args.extend([cx, cy])
            args.extend(x_mcast.runtime_args(core))
            args.extend(h_mcast.runtime_args(core))
            reader_rt[x][y] = args

            px, py = _virt(device, *node["parent"]) if node["parent"] is not None else (0, 0)
            writer_rt[x][y] = [
                mailbox_addr,
                w_up.buffer_address(),
                output_tensor.buffer_address(),
                kr,
                kstart,
                hstart,
                hn,
                ec,
                jstart,
                1 if node["is_root"] else 0,
                px,
                py,
            ]

            compute_rt[x][y] = [
                mailbox_addr,
                kr,
                hn,
                ec,
                1 if node["is_root"] else 0,
                len(node["children"]),
                n_inject,
            ]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "moe_fused_swiglu_reader.cpp"),
            core_ranges=all_cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "moe_fused_swiglu_writer.cpp"),
            core_ranges=all_cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "moe_fused_swiglu_compute.cpp"),
            core_ranges=all_cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_kernel_config,
            defines=[("SKIP_COMPUTE", "1")] if ABLATE == "skip_compute" else [],
        ),
    ]

    semaphores = list(x_mcast.owned_semaphores()) + list(h_mcast.owned_semaphores())
    semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=all_cores, initial_value=0))
    semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_DATA, core_ranges=all_cores, initial_value=0))

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
