# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for the 2D dual-multicast matmul.

Maps the output tile-grid (Mt x Nt) onto a GR x GC Tensix grid (rows own
M-blocks, cols own N-blocks). The first column (X==0) reads its activation
row-block and multicasts it across the row; the first row (Y==0) reads its
weight column-block and multicasts it down the column. K is streamed in
K-blocks of `in0_block_w`, so per-core L1 stays bounded for arbitrary K.
M/N blocks are bounded by an L1 budget; when a core's region exceeds one
block the grid iterates per-core blocks in lock-step.
"""

from pathlib import Path
import math

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"

# --- CB indices (semantic names in the kernels) ---
CB_IN0_ACT = 0  # activation row-block (in0)
CB_IN1_WEIGHT = 1  # weight column-block (in1)
CB_OUT = 16  # finished output block
CB_INTERM = 24  # K spill/reload (internal to matmul_block)

# Usable L1 budget for the 4 matmul CBs (bytes). Conservative; blocks shrink to fit.
L1_MM_BUDGET = 1_000_000


def _ceil_div(a, b):
    return (a + b - 1) // b


def _find_max_divisor(value, max_div):
    """Largest divisor of `value` that is <= max_div (>= 1)."""
    for d in range(min(max_div, value), 0, -1):
        if value % d == 0:
            return d
    return 1


def _choose_subblock(bM, bN, dest_limit):
    """Largest (h, w) with h | bM, w | bN, h*w <= dest_limit (by tile count)."""
    best_h, best_w, best_area = 1, 1, 1
    for h in range(1, bM + 1):
        if bM % h:
            continue
        for w in range(1, bN + 1):
            if bN % w:
                continue
            area = h * w
            if area <= dest_limit and area >= best_area:
                best_h, best_w, best_area = h, w, area
    return best_h, best_w


def _dest_limit(fp32_acc, full_sync):
    if full_sync:
        return 8 if fp32_acc else 16
    return 4 if fp32_acc else 8


def _effective_compute_config(user_cfg, fp32_acc, eff_low_precision):
    """Build the ComputeConfigDescriptor actually attached to the compute kernel.

    Copies the caller's config verbatim EXCEPT for one hardware-correctness
    workaround: on Wormhole B0, HiFi4 + fp32_dest_acc_en with bf16/bf8b inputs
    silently corrupts the matmul K-accumulator (issue #38306; warned in
    matmul_block_helpers.hpp). bf16/bf8b carry <=7 mantissa bits, which fit
    entirely in HiFi2, so HiFi4 buys them no precision — clamp HiFi4 -> HiFi2
    for low-precision inputs to dodge the corruption with zero precision cost.
    fp32 inputs keep HiFi4 (their only correct fidelity).
    """
    fidelity = getattr(user_cfg, "math_fidelity", ttnn.MathFidelity.HiFi4)
    approx = bool(getattr(user_cfg, "math_approx_mode", False))
    full_sync = bool(getattr(user_cfg, "dst_full_sync_en", False))
    precise = bool(getattr(user_cfg, "bfp8_pack_precise", False))
    if eff_low_precision and fidelity == ttnn.MathFidelity.HiFi4:
        fidelity = ttnn.MathFidelity.HiFi2
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity,
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc,
        dst_full_sync_en=full_sync,
        bfp8_pack_precise=precise,
    )


def _distribute(Mt, Nt, Kt, gx, gy, dest_limit, tileA_bytes, tileB_bytes, tileC_bytes, interm_bytes):
    """Choose K-blocking, output block size, grid extent, and per-core loop counts.

    The L1 footprint is dtype-aware: in0/in1/out each carry their own tile-byte
    size (bf16 = 2 KiB, fp32 = 4 KiB, bf8b ~1 KiB) and the interm accumulator its
    own (fp32 = 4 KiB under fp32_dest_acc_en, else the output dtype). Phase 0 used
    a single fp32 tile size for every CB — fine only when all inputs were fp32.
    """
    in0_block_w = _find_max_divisor(Kt, 4)
    num_k_blocks = Kt // in0_block_w

    # Start by covering the grid with one block per core, then shrink to fit L1.
    bM = max(1, _ceil_div(Mt, gy))
    bN = max(1, _ceil_div(Nt, gx))

    def footprint(bm, bn):
        in0 = bm * in0_block_w * 2 * tileA_bytes
        in1 = in0_block_w * bn * 2 * tileB_bytes
        out = bm * bn * 2 * tileC_bytes
        interm = bm * bn * interm_bytes
        return in0 + in1 + out + interm

    while footprint(bM, bN) > L1_MM_BUDGET and (bM > 1 or bN > 1):
        if bM >= bN and bM > 1:
            bM = max(1, bM // 2)
        elif bN > 1:
            bN = max(1, bN // 2)
        else:
            break

    sb_h, sb_w = _choose_subblock(bM, bN, dest_limit)
    in0_num_subblocks = bM // sb_h
    in1_num_subblocks = bN // sb_w

    num_m_blocks = _ceil_div(Mt, bM)
    num_n_blocks = _ceil_div(Nt, bN)
    GR = min(gy, num_m_blocks)
    GC = min(gx, num_n_blocks)
    per_core_M_blocks = _ceil_div(num_m_blocks, GR)
    per_core_N_blocks = _ceil_div(num_n_blocks, GC)

    return {
        "in0_block_w": in0_block_w,
        "num_k_blocks": num_k_blocks,
        "block_M_tiles": bM,
        "block_N_tiles": bN,
        "out_subblock_h": sb_h,
        "out_subblock_w": sb_w,
        "in0_num_subblocks": in0_num_subblocks,
        "in1_num_subblocks": in1_num_subblocks,
        "GR": GR,
        "GC": GC,
        "per_core_M_blocks": per_core_M_blocks,
        "per_core_N_blocks": per_core_N_blocks,
    }


def create_program_descriptor(input_tensor, weight, output_tensor, compute_kernel_config):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y

    A_shape = list(input_tensor.shape)
    B_shape = list(weight.shape)
    M, K = A_shape[-2], A_shape[-1]
    N = B_shape[-1]
    # ceil_div tile counts handle non-tile-aligned M/K/N (Refinement 2): the
    # partial last tile along each dim is a real tile the kernels process in
    # full. ttnn zero-fills its out-of-logical-shape padding at from_torch time
    # (all dtypes incl. bf8b), so the K dot-product over the pad is 0*0=0 and
    # the M/N output pad (also 0) is sliced off by the output's logical shape.
    # No in-kernel masking required — see the reader/writer head comments.
    Mt = _ceil_div(M, 32)
    Kt = _ceil_div(K, 32)
    Nt = _ceil_div(N, 32)
    batch = 1
    for d in A_shape[:-2]:
        batch *= d

    # Refinement 3 — true batched matmul. The reader's in1 (weight) tile-id is
    # b * weight_batch_stride + gk * Nt + gn. For a SHARED 2D weight the stride is
    # 0 (every batch re-reads the same (K, N) block — the Phase-0 behavior). For a
    # BATCHED weight (..., K, N) whose leading dims match the activation's, the
    # stride is Kt * Nt so batch b reads weight matrix b. The flattened batch index
    # `b` in the kernel loop maps identically over A's and B's leading dims (both
    # row-major over the same leading shape), so b indexes the matching weight.
    # The activation read, writer, and dual-multicast topology are unchanged: the
    # in1 sender (grid row Y=0) reads weight block b for its N-column and multicasts
    # it down the column to all cores working on batch b in lock-step.
    B_lead = B_shape[:-2]
    weight_batched = len(B_lead) > 0 and not all(d == 1 for d in B_lead)
    weight_batch_stride = (Kt * Nt) if weight_batched else 0

    tileA_bytes = input_tensor.buffer_page_size()
    tileB_bytes = weight.buffer_page_size()
    tileC_bytes = output_tensor.buffer_page_size()

    fp32_acc = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True))
    full_sync = bool(getattr(compute_kernel_config, "dst_full_sync_en", False))
    dest_limit = _dest_limit(fp32_acc, full_sync)

    # ----- intermediate (K-spill/reload) CB format -----
    # matmul_block's last-K-block "pack to out" data-format reconfig is gated on
    # (packer_l1_acc || fp32_dest_acc_en) (matmul_block_helpers.inl:394).
    #   * fp32_dest_acc_en=True  -> interm = Float32: the fp32 accumulator is
    #     preserved across K-blocks; the gated reconfig swaps the packer to the
    #     output format for the final pack, so out may be any dtype.
    #   * fp32_dest_acc_en=False, default (software spill, packer_l1_acc=False):
    #     interm MUST equal the output format — no reconfig fires before the final
    #     pack, so the packer stays at interm's format. A mismatch corrupts output.
    #
    # Refinements 1 (Lever B) + 1b: for a LOW-PRECISION output with acc=False the
    # default software spill re-quantizes the running K-sum to the OUTPUT format on
    # EVERY K-block reload (the partial is reloaded into the 16-bit DEST and re-packed
    # at out's format each block), which dominates the deep-K error. Opt these corners
    # into HARDWARE packer-L1 accumulation: with packer_l1_acc=True the last-block
    # pack-to-out data-format reconfig (gated on packer_l1_acc || fp32_dest_acc_en,
    # matmul_block_helpers.inl:394) fires, so cb_interm can carry a format FINER than
    # the output. The running K-sum then accumulates in L1 in that finer format across
    # all K-blocks (never reloading to the 16-bit DEST until the final block), and the
    # final pack reconfigs down to the output format. The 16-bit DEST is still honored
    # per-K-block — packer_l1_acc is an orthogonal hardware knob (the user controls
    # only DEST width via fp32_dest_acc_en; L1-accumulation strategy is the op's).
    #
    #   * bf8b out (Refinement 1, Lever B): interm = Float16_b (bf16). K-sum stays bf16
    #     instead of re-quantizing to bf8b each spill (deep-K relRMS 0.39 -> 0.022).
    #   * bf16 out (Refinement 1b): interm = Float32. The cross-K-block running sum
    #     accumulates in fp32 instead of re-quantizing to bf16 each spill, bounding the
    #     in-DEST 16-bit accumulation run to ONE K-block (in0_block_w*32 K-elements)
    #     rather than the full K. Drops the deep-K (K=8192) relRMS 0.128 -> 0.009 while
    #     leaving fp32_dest_acc_en=False (16-bit DEST per block) intact.
    #
    # fp32 acc (acc=True) keeps its native fp32 DEST accumulator + Float32 interm.
    # #28800 (l1_acc breaks fp32_dest_acc_en) does not apply: these branches are
    # acc=False only, where packer-L1 and the 16-bit DEST do not conflict.
    use_packer_l1_acc = (not fp32_acc) and (output_tensor.dtype in (ttnn.bfloat8_b, ttnn.bfloat16))
    if fp32_acc:
        interm_format = ttnn.float32
    elif use_packer_l1_acc:
        # interm one level finer than output: bf8b out -> bf16 interm; bf16 out -> fp32.
        interm_format = ttnn.bfloat16 if output_tensor.dtype == ttnn.bfloat8_b else ttnn.float32
    else:
        interm_format = output_tensor.dtype
    interm_bytes = ttnn.tile_size(interm_format)

    d = _distribute(Mt, Nt, Kt, gx, gy, dest_limit, tileA_bytes, tileB_bytes, tileC_bytes, interm_bytes)
    in0_block_w = d["in0_block_w"]
    num_k_blocks = d["num_k_blocks"]
    block_M_tiles = d["block_M_tiles"]
    block_N_tiles = d["block_N_tiles"]
    out_subblock_h = d["out_subblock_h"]
    out_subblock_w = d["out_subblock_w"]
    in0_num_subblocks = d["in0_num_subblocks"]
    in1_num_subblocks = d["in1_num_subblocks"]
    GR = d["GR"]
    GC = d["GC"]
    per_core_M_blocks = d["per_core_M_blocks"]
    per_core_N_blocks = d["per_core_N_blocks"]

    # ===== core grid =====
    grid_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GC - 1, GR - 1))
    core_grid = ttnn.CoreRangeSet([grid_range])

    # ===== circular buffers (identical on all grid cores) =====
    def make_cb(index, page_size, num_pages, data_format):
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index,
                    data_format=data_format,
                    page_size=page_size,
                )
            ],
        )

    in0_block_tiles = block_M_tiles * in0_block_w
    in1_block_tiles = in0_block_w * block_N_tiles
    out_block_tiles = block_M_tiles * block_N_tiles

    cb_in0 = make_cb(CB_IN0_ACT, tileA_bytes, in0_block_tiles * 2, input_tensor.dtype)
    cb_in1 = make_cb(CB_IN1_WEIGHT, tileB_bytes, in1_block_tiles * 2, weight.dtype)
    cb_out = make_cb(CB_OUT, tileC_bytes, out_block_tiles * 2, output_tensor.dtype)
    cb_interm = make_cb(CB_INTERM, interm_bytes, out_block_tiles, interm_format)

    # ===== mcast helpers (host counterpart of kernel_lib/mcast_pipe) =====
    # in0 = activation: per-row mcast, sender in column 0 (broadcasts across its row).
    # in1 = weight:     per-column mcast, sender in row 0 (broadcasts down its column).
    # The helper OWNS semaphore creation, logical->virtual + rect corners + NOC-order (min/max over
    # actual virtual coords — the BH-correct swap the hand-rolled virt() lacked), and the degenerate
    # single-line case. mc.active() reproduces the old in0_mcast=GC>1 / in1_mcast=GR>1 gate;
    # mc.owned_semaphores() replaces the 4 hand-written descriptors; mc.compile_time_args() emits the
    # 5-word block [active, data_ready_id, consumer_ready_id, num_active, flags] (flags carries
    # pre_handshake + the data-ready signal). in1's base is chained off in0's next_base_sem_id() so the
    # ids never overlap (in0 takes 0,1; in1 takes 2,3) without hardcoding the count here.
    mc_in0 = ttnn.Mcast1D(device, core_grid, ttnn.Mcast1DShape.PerRow, 0, ttnn.McastConfig(base_sem_id=0))
    mc_in1 = ttnn.Mcast1D(
        device, core_grid, ttnn.Mcast1DShape.PerColumn, 0, ttnn.McastConfig(base_sem_id=mc_in0.next_base_sem_id())
    )
    semaphores = mc_in0.owned_semaphores() + mc_in1.owned_semaphores()

    # ===== reader kernel =====
    # CT layout (joint wire with matmul_reader.cpp's McastArgs<CT,RT> decoder):
    #   0..9  scalars | 10..14 in0 mcast (5 words) | 15..19 in1 mcast | 20..22 tile bytes + weight stride
    #   | 23.. TensorAccessorArgs(A) then (B)
    reader_ct = [
        Mt,
        Nt,
        Kt,
        batch,
        block_M_tiles,
        block_N_tiles,
        in0_block_w,
        num_k_blocks,
        per_core_M_blocks,
        per_core_N_blocks,
    ]
    reader_ct += list(mc_in0.compile_time_args())  # 10..14: [active, data_ready, consumer_ready, num_active, flags]
    reader_ct += list(mc_in1.compile_time_args())  # 15..19
    reader_ct += [
        tileA_bytes,
        tileB_bytes,
        weight_batch_stride,  # Refinement 3: in1 per-batch tile offset (0 if shared)
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()

    A_addr = input_tensor.buffer_address()
    B_addr = weight.buffer_address()
    C_addr = output_tensor.buffer_address()

    for Y in range(GR):
        for X in range(GC):
            # The whole virt()/sender-receiver branch collapses into two runtime_args(core) calls:
            # each returns the 4-word block the kernel's McastArgs<CT,RT> decodes off RT (sender ->
            # dest rect, receiver -> [sender_x, sender_y, 0, 0], degenerate -> [0,0,0,0]).
            core = ttnn.CoreCoord(X, Y)
            reader_rt[X][Y] = [A_addr, B_addr, Y, X] + list(mc_in0.runtime_args(core)) + list(mc_in1.runtime_args(core))
            writer_rt[X][Y] = [C_addr, Y, X]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "matmul_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ===== compute kernel =====
    total_blocks_per_core = batch * per_core_M_blocks * per_core_N_blocks
    compute_ct = [
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_h,
        out_subblock_w,
        in0_block_w,
        num_k_blocks,
        total_blocks_per_core,
        1 if use_packer_l1_acc else 0,  # Lever B: hardware packer L1 K-accumulation
    ]
    # Effective config = caller's, with the #38306 HiFi4->HiFi2 clamp for
    # bf16/bf8b inputs (effective dtype = coarser of activation/weight).
    eff_low_precision = (input_tensor.dtype != ttnn.float32) or (weight.dtype != ttnn.float32)
    effective_compute_config = _effective_compute_config(compute_kernel_config, fp32_acc, eff_low_precision)

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "matmul_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct,
        runtime_args=[],
        config=effective_compute_config,
    )

    # ===== writer kernel =====
    writer_ct = [
        Mt,
        Nt,
        batch,
        block_M_tiles,
        block_N_tiles,
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_h,
        out_subblock_w,
        per_core_M_blocks,
        per_core_N_blocks,
        tileC_bytes,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "matmul_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=semaphores,
        cbs=[cb_in0, cb_in1, cb_out, cb_interm],
    )
