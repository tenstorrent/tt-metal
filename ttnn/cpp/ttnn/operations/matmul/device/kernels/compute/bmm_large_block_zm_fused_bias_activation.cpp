// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "internal/mod_div_lib.h"

#ifdef FUSE_BIAS
#include "api/compute/bcast.h"
#endif

#include "api/compute/eltwise_binary.h"
#ifdef SFPU_ACTIVATION
#include "bmm_fused_activation.hpp"
#endif

// ============================================================================
// FUSED_ARGMAX — opt-in greedy-argmax epilogue for the DRAM-sharded decode
// matmul (Blackhole only; factory sets the define only when the caller
// provides the partials tensor and the eligibility gates pass: no bias, no
// activation, TILE bf16 output, per_core_M == 1, 32x32 tiles).
//
// The pack RISC (TRISC2) carries a Zve32f vector unit that is idle in this
// kernel. After each output subblock's pack sequence is QUEUED, TRISC2 scans
// the PREVIOUS subblock's freshly packed bf16 tiles straight out of the
// output CB and keeps per-row running argmax+maxval state; per-worker
// (global_index, value_bits) partials are staged into a small CB that the
// in1 sender/writer drains to the preallocated partials tensor (8 DRAM-bank
// workers => 8 partials, combined by the caller: 8 compares).
//
// Deferred-consume order (the silicon-measured mechanic, THREEWAY_
// OPPORTUNITIES.md MEASURED M.3): scanning the subblock that was just packed
// would chain the NEXT subblock's queue behind pack_done + scan and stall the
// math thread on DST halves (+24% measured at the LM-head cadence). Deferring
// the scan by one FULL SUBBLOCK means the mailbox wait for subblock s-1 is
// answered while subblock s's whole sequence sits queued in the Tensix FIFO,
// so the scan runs entirely inside the queued-work shadow (defer-by-2 at
// tile granularity measured +0.21% wall; a subblock defer is the same
// mechanic at 8-tile granularity). The mailbox is the c_out received stream
// register: the queued cb_push_back STOREREG updates it only after the pack
// HW physically wrote the tiles — true data-ready, and legal by construction
// because this kernel single-buffers the whole output block (pages are not
// recycled until the writer drains them at the end).
//
// Semantics — exactly ttnn.argmax's bfloat16_greater + smallest-index rule:
// bfloat16_greater is a pure sign-magnitude bit-pattern total order; the xor
// trick (t = x ^ 0x8000) makes unsigned lane max agree with it whenever any
// sign-0 pattern exists, and an all-negative row takes an exact unsigned-MIN
// fix-up (both-negative order is reversed). Cross-tile combines happen in the
// fully monotone domain m = x ^ ((x >> 15 arith) | 0x8000), seeded with
// m(0xFF80) — the incumbent's -inf init. Strictly-greater updates plus a
// first-match in-tile re-scan preserve the smallest-index tie-break; across
// workers, ascending-bank host combine with strictly-greater keeps the
// lowest GLOBAL index. The scan covers only the LOGICAL output width, so
// padded-vocab tail columns never participate (no -inf mask add needed).
//
// REGISTER BUDGET (hard-earned, argmax_rvv_tile_compute.cpp): e16m4 keeps the
// vector working set at 8 of 32 vregs; helpers stay noinline; any change must
// be checked for vlenb-scaled stack frames (TRISC2 has ~256B of stack; e16m8
// dual-stream spills multi-KB and hangs silicon). VLEN=128: 16 lanes of e16
// need m2 (e16m1 caps vl at 8).
// ============================================================================
#if defined(FUSED_ARGMAX) && defined(TRISC_PACK)
#include <riscv_vector.h>
#include "api/compute/common.h"           // get_local_cb_interface / stream-register CB pointers
#include "internal/tt-1xx/risc_common.h"  // invalidate_l1_cache(), reg_read

namespace fused_argmax {

constexpr uint16_t kSignBit = 0x8000u;
// Monotone image of 0xFF80 (-inf) — the incumbent scan's initial max value.
constexpr uint16_t kInitMono = 0x007Fu;

// Running per-row state. Static => local data memory, not the (tiny) stack.
static uint16_t s_mono[32];
static uint16_t s_raw[32];
static uint32_t s_idx[32];

struct FaState {
    uint32_t c_out_base = 0;     // byte address of output CB page 0 (pre-push snapshot)
    uint32_t page_bytes = 0;     // output CB page stride (bf16 32x32 tile = 2048B)
    uint32_t recv_reg = 0;       // MMIO address of the output CB received stream register
    uint16_t recv_base = 0;      // its value before the first push
    uint32_t n_tile_offset = 0;  // this worker's global column offset, in tiles
    uint32_t valid_w = 0;        // valid (logical) output tiles on this worker
    uint32_t valid_rows = 0;     // valid output rows (decode batch), <= 32
    uint32_t n_groups = 0;       // row-pair groups = ceil(valid_rows / 2)
    uint32_t sb_tiles = 0;       // tiles per subblock push
    uint32_t total_tiles = 0;    // per-core output tiles (incl. subblock padding)
    uint32_t sb_pushed = 0;      // subblock pack sequences queued so far
    uint32_t sb_consumed = 0;    // subblocks scanned so far
};
static FaState fa;

inline void init(uint32_t out_cb, uint32_t sb_tiles, uint32_t total_tiles) {
    LocalCBInterface& intf = get_local_cb_interface(out_cb);
    fa.c_out_base = intf.fifo_wr_ptr << 4;  // pre-push: wr ptr == base
    fa.page_bytes = intf.fifo_page_size << 4;
    fa.recv_reg = (uint32_t)(uintptr_t)get_cb_tiles_received_ptr((int)out_cb);
    fa.recv_base = (uint16_t)reg_read(fa.recv_reg);
    fa.n_tile_offset = get_arg_val<uint32_t>(1);
    fa.valid_w = get_arg_val<uint32_t>(2);
    fa.valid_rows = get_arg_val<uint32_t>(3);
    fa.n_groups = (fa.valid_rows + 1) / 2;
    fa.sb_tiles = sb_tiles;
    fa.total_tiles = total_tiles;
    fa.sb_pushed = 0;
    fa.sb_consumed = 0;
    for (uint32_t r = 0; r < 32; r++) {
        s_mono[r] = kInitMono;
        s_raw[r] = 0xFF80u;  // -inf: the incumbent's init value
        s_idx[r] = 0;
    }
}

// Scan row-pair group `g` across a run of `ntiles` CONSECUTIVE packed bf16
// 32x32 tiles (faces 0..3, 512B each; left faces hold cols 0..15, right
// faces +512B cols 16..31) starting at L1 address base0, and fold it into
// the running per-row state. Batching pass A across the whole subblock
// amortizes the (expensive) per-row lane reductions over ntiles tiles —
// exactly the argmax-RVV op's chunk structure, with a contiguous-page walk.
// first_tile_col is the GLOBAL column index of the first tile in the run.
__attribute__((noinline)) static void scan_group_run(
    uint32_t base0, uint32_t ntiles, uint32_t g, uint32_t rows_in_group, uint32_t first_tile_col) {
    const uint32_t left_off = ((g < 8) ? 0u : 1024u) + (g & 7u) * 64u;
    const size_t vl = __riscv_vsetvl_e16m4(32);

    // Pass A: lane-wise max of x ^ 0x8000 across the run's tiles. The two
    // 16-elem face rows of a row-pair are CONTIGUOUS in a face, so one e16m4
    // (vl=32) load covers rows {2g, 2g+1} x one face.
    vuint16m4_t acc_l = __riscv_vmv_v_x_u16m4(0, vl);
    vuint16m4_t acc_r = __riscv_vmv_v_x_u16m4(0, vl);
    for (uint32_t t = 0; t < ntiles; t++) {
        const uint32_t base = base0 + t * fa.page_bytes;
        vuint16m4_t va = __riscv_vle16_v_u16m4((const uint16_t*)(base + left_off), vl);
        vuint16m4_t vb = __riscv_vle16_v_u16m4((const uint16_t*)(base + left_off + 512u), vl);
        va = __riscv_vxor_vx_u16m4(va, kSignBit, vl);
        vb = __riscv_vxor_vx_u16m4(vb, kSignBit, vl);
        acc_l = __riscv_vmaxu_vv_u16m4(acc_l, va, vl);
        acc_r = __riscv_vmaxu_vv_u16m4(acc_r, vb, vl);
    }

    for (uint32_t rr = 0; rr < rows_in_group; rr++) {
        const uint32_t row = 2 * g + rr;
        const uint32_t row_off = left_off + rr * 32u;  // 16 bf16 = 32B per face row

        vuint16m4_t sl = rr ? __riscv_vslidedown_vx_u16m4(acc_l, 16, vl) : acc_l;
        vuint16m4_t sr = rr ? __riscv_vslidedown_vx_u16m4(acc_r, 16, vl) : acc_r;
        const vuint16m1_t z = __riscv_vmv_s_x_u16m1(0, 1);
        uint16_t t_row = __riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m4_u16m1(sl, z, 16));
        const uint16_t t_r = __riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m4_u16m1(sr, z, 16));
        if (t_r > t_row) {
            t_row = t_r;
        }

        uint16_t mono;
        uint16_t raw;
        if (t_row >= kSignBit) {
            // Some sign-0 pattern exists: xor-domain max IS the winner.
            mono = t_row;
            raw = (uint16_t)(t_row ^ kSignBit);
        } else {
            // All-negative run-row: both-negative order is reversed — take
            // the exact unsigned MIN over the (still resident) run.
            // NOTE: 16 lanes of e16 need m2 — VLEN=128 caps e16m1 at vl=8.
            const size_t vl1 = __riscv_vsetvl_e16m2(16);
            vuint16m2_t mn = __riscv_vmv_v_x_u16m2(0xFFFFu, vl1);
            for (uint32_t t = 0; t < ntiles; t++) {
                const uint32_t base = base0 + t * fa.page_bytes;
                vuint16m2_t a = __riscv_vle16_v_u16m2((const uint16_t*)(base + row_off), vl1);
                vuint16m2_t b = __riscv_vle16_v_u16m2((const uint16_t*)(base + row_off + 512u), vl1);
                a = __riscv_vxor_vx_u16m2(a, kSignBit, vl1);
                b = __riscv_vxor_vx_u16m2(b, kSignBit, vl1);
                mn = __riscv_vminu_vv_u16m2(mn, a, vl1);
                mn = __riscv_vminu_vv_u16m2(mn, b, vl1);
            }
            const uint16_t t_min =
                __riscv_vmv_x_s_u16m1_u16(__riscv_vredminu_vs_u16m2_u16m1(mn, __riscv_vmv_s_x_u16m1(0xFFFFu, 1), vl1));
            mono = (uint16_t)(t_min ^ 0x7FFFu);
            raw = (uint16_t)(t_min ^ kSignBit);
        }

        // Strictly-greater keeps the earliest occurrence across runs;
        // ascending first-match re-scan keeps it within the run.
        if (mono > s_mono[row]) {
            const size_t vl1 = __riscv_vsetvl_e16m2(16);
            uint32_t idx = 0;
            for (uint32_t t = 0; t < ntiles; t++) {
                const uint32_t base = base0 + t * fa.page_bytes;
                const vuint16m2_t ca = __riscv_vle16_v_u16m2((const uint16_t*)(base + row_off), vl1);
                const long f_l = __riscv_vfirst_m_b8(__riscv_vmseq_vx_u16m2_b8(ca, raw, vl1), vl1);
                if (f_l >= 0) {
                    idx = (first_tile_col + t) * 32u + (uint32_t)f_l;
                    break;
                }
                const vuint16m2_t cb = __riscv_vle16_v_u16m2((const uint16_t*)(base + row_off + 512u), vl1);
                const long f_r = __riscv_vfirst_m_b8(__riscv_vmseq_vx_u16m2_b8(cb, raw, vl1), vl1);
                if (f_r >= 0) {
                    idx = (first_tile_col + t) * 32u + 16u + (uint32_t)f_r;
                    break;
                }
            }
            s_mono[row] = mono;
            s_raw[row] = raw;
            s_idx[row] = idx;
        }
    }
}

// Consume one pushed subblock: mailbox-wait until the output CB received
// stream register (updated by the queued cb_push_back STOREREG only after
// the pack HW physically wrote the tiles) covers it, then scan its (valid)
// tiles as one contiguous run per row-pair group.
static void consume_subblock(uint32_t sbi) {
    const uint32_t t0 = sbi * fa.sb_tiles;
    uint32_t t1 = t0 + fa.sb_tiles;
    if (t1 > fa.total_tiles) {
        t1 = fa.total_tiles;
    }
    while ((uint16_t)((uint16_t)reg_read(fa.recv_reg) - fa.recv_base) < (uint16_t)t1) {
    }
    fa.sb_consumed = sbi + 1;
    // Clamp to the logical width: subblock-pad DST lanes and bank-pad tiles
    // are packed but never scanned.
    const uint32_t tv = (t1 < fa.valid_w) ? t1 : fa.valid_w;
    if (tv <= t0) {
        return;
    }
    invalidate_l1_cache();  // BH: L1 data reads after MMIO count poll
    const uint32_t base0 = fa.c_out_base + t0 * fa.page_bytes;
    for (uint32_t g = 0; g < fa.n_groups; g++) {
        const uint32_t rows_in_group = (fa.valid_rows - 2 * g < 2u) ? (fa.valid_rows - 2 * g) : 2u;
        scan_group_run(base0, tv - t0, g, rows_in_group, fa.n_tile_offset + t0);
    }
}

// Called right after each output subblock's pack sequence is queued:
// defer-by-one-subblock consume order (see header note).
//
// MODE SPLIT (silicon-measured): for decode-b1 shapes (<= 2 row-pair groups)
// the per-subblock scan is light enough to ride the pack shadow (+0.9%
// per-op device span). For full-tile row counts the per-run extraction cost
// times 16 groups exceeds the LAST inner K-block's shadow (all output packs
// concentrate there in this kernel's K-outer loop order; measured +34%/op
// when scanned per subblock). Those shapes skip the in-loop scans entirely
// and take ONE full-width run per group at finalize — the argmax-RVV op's
// chunk shape, amortizing extraction over the whole per-core width, and
// overlapping the writer's output write-back instead.
inline void on_subblock_pushed() {
    fa.sb_pushed++;
    if (fa.n_groups <= 2 && fa.sb_pushed >= 2) {
        consume_subblock(fa.sb_pushed - 2);
    }
}

// Tail: scan whatever is still outstanding, then publish the partials page
// (32 x (index, value bits) u32 pairs) into the staging CB — we are its sole
// producer; the in1 sender/writer drains it with plain dataflow calls.
static void finalize(uint32_t cb_partials) {
    const uint32_t n_subblocks = (fa.total_tiles + fa.sb_tiles - 1) / fa.sb_tiles;
    if (fa.n_groups <= 2) {
        while (fa.sb_consumed < n_subblocks) {
            consume_subblock(fa.sb_consumed);
        }
    } else if (fa.valid_w > 0) {
        // Full-tile row counts: one full-width run per group (see the mode
        // note at on_subblock_pushed). Wait once for every tile, then scan.
        while ((uint16_t)((uint16_t)reg_read(fa.recv_reg) - fa.recv_base) < (uint16_t)fa.total_tiles) {
        }
        invalidate_l1_cache();  // BH: L1 data reads after MMIO count poll
        for (uint32_t g = 0; g < fa.n_groups; g++) {
            const uint32_t rows_in_group = (fa.valid_rows - 2 * g < 2u) ? (fa.valid_rows - 2 * g) : 2u;
            scan_group_run(fa.c_out_base, fa.valid_w, g, rows_in_group, fa.n_tile_offset);
        }
    }
    LocalCBInterface& intf = get_local_cb_interface(cb_partials);
    volatile uint32_t* page = (volatile uint32_t*)(intf.fifo_wr_ptr << 4);
    for (uint32_t r = 0; r < 32; r++) {
        page[2 * r] = s_idx[r];
        page[2 * r + 1] = (uint32_t)s_raw[r];
    }
    asm volatile("fence" ::: "memory");
    (void)page[63];                                      // read-back orders the L1 stores before the MMIO count store
    get_cb_tiles_received_ptr((int)cb_partials)[0] = 1;  // sole producer; counter is 0 at launch
}

}  // namespace fused_argmax
#endif  // FUSED_ARGMAX && TRISC_PACK

// Please update
// tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp
// when making any changes to this file.
// Have to keep a copy because cannot import ttnn into tests/tt_metal.
// With FUSE_BIAS: row_broadcast_bias (row-broadcast vs elementwise add_tiles) is compile-time arg 18 here;
// the perf copy uses index 14 (different compile-time arg layout).

/**
 * @brief Transposes a block of tiles from one circular buffer to another.
 *
 * This function reads a block of tiles from the input circular buffer (cb), performs a width-height
 * (WH) transpose on each tile, and writes the transposed tiles to the output circular buffer.
 * The operation is performed in blocks of `block_size` tiles for efficiency, with a separate loop
 * at the end to handle any leftover tiles when the total tile count is not divisible by
 * `block_size`. The default block size is 4, since there are guaranteed to be 4 tiles in the dst
 *               regs irrespective of dst sync mode or data format.
 *
 * @tparam in0_block_num_tiles The number of tiles in the block to be transposed.
 * @tparam block_size The number of tiles in each block to be transposed.
 * @param in0_transpose_dfb_id Circular buffer ID to read the original tiles from.
 * @param in0_dfb_id Circular buffer ID to which the transposed tiles are written.
 */
template <uint32_t in0_block_num_tiles, uint32_t block_size = 4>
FORCE_INLINE void transpose_tile_block(uint32_t in0_transpose_dfb_id, uint32_t in0_dfb_id) {
    DataflowBuffer in0_transpose_dfb(in0_transpose_dfb_id);
    DataflowBuffer in0_dfb(in0_dfb_id);
    constexpr uint32_t num_blocks = in0_block_num_tiles / block_size;
    constexpr uint32_t last_block_size = in0_block_num_tiles % block_size;
    // Lets do 2 passes: One loop until last and one last for the left overs
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        in0_transpose_dfb.wait_front(block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            transpose_tile(in0_transpose_dfb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in0_transpose_dfb.pop_front(block_size);

        in0_dfb.reserve_back(block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            pack_tile(tile_idx, in0_dfb_id);
        }
        tile_regs_release();
        in0_dfb.push_back(block_size);
    }

    if constexpr (last_block_size > 0) {
        in0_transpose_dfb.wait_front(last_block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            transpose_tile(in0_transpose_dfb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in0_transpose_dfb.pop_front(last_block_size);

        in0_dfb.reserve_back(last_block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            pack_tile(tile_idx, in0_dfb_id);
        }
        tile_regs_release();
        in0_dfb.push_back(last_block_size);
    }
}

FORCE_INLINE void reload_from_cb_to_dst(
    uint32_t in0_dfb_id,
    uint32_t in1_dfb_id,
    uint32_t mm_partials_dfb_id,
    uint32_t mm_partials_reload_dfb_id,
    bool in1_transpose_tile,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    DataflowBuffer mm_partials_dfb(mm_partials_dfb_id);
    // mm_partials_reload_dfb_id is the CB view the reload copies through. It equals mm_partials_dfb_id
    // unless the partials CB is also read as an FPU operand elsewhere (the fused bias add reads it via
    // SrcA), in which case UnpackToDestFp32 cannot be set on it directly; instead a second buffer index
    // aliases the same SRAM with UnpackToDestFp32 set, and the reload copies through that alias while the
    // FPU consumer keeps the original view. The alias has its own read pointer, so align it with the
    // partials CB's current read position before copying.
    // Reconfigure input
    copy_tile_to_dst_init_short_with_dt(in1_dfb_id, mm_partials_reload_dfb_id);
    mm_partials_dfb.wait_front(out_subblock_num_tiles);

    if (mm_partials_reload_dfb_id != mm_partials_dfb_id) {
        // Only the unpacker owns cb_interface / the read pointer; keep this off the MATH/PACK threads.
        UNPACK(
            (get_local_cb_interface(mm_partials_reload_dfb_id).fifo_rd_ptr =
                 get_local_cb_interface(mm_partials_dfb_id).fifo_rd_ptr));
    }

    uint32_t start_dst_index = 0;
    uint32_t start_tile_index = 0;
    copy_block(mm_partials_reload_dfb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);

    mm_partials_dfb.pop_front(out_subblock_num_tiles);
    // Reconfigure srcA back
    reconfig_data_format_srca(mm_partials_reload_dfb_id, in1_dfb_id);
    matmul_block_init(in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
}

template <uint32_t out_subblock_w, uint32_t out_block_w>
inline void reblock_and_untilize(
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t interm_dfb_id,
    uint32_t out_dfb_id) {
    DataflowBuffer interm_dfb(interm_dfb_id);
    DataflowBuffer out_dfb(out_dfb_id);
    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_out_subblocks_in_col);
    interm_dfb.wait_front(num_tiles_in_row_of_subblocks);

    uint32_t within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        uint32_t block_offset = 0;

        out_dfb.reserve_back(out_block_w);
        for (uint32_t n = 0; n < num_out_subblocks_in_col; n++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                uint32_t tile_index = block_offset + within_block_index + w;
                copy_tile(interm_dfb_id, tile_index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<out_subblock_w, out_block_w>(out_dfb_id, 1, n);
            tile_regs_release();
            block_offset += out_subblock_num_tiles;
        }
        out_dfb.push_back(out_block_w);

        within_block_index += out_subblock_w;
    }
    interm_dfb.pop_front(num_tiles_in_row_of_subblocks);
}

void kernel_main() {
// RUNTIME ARGS
#ifdef MATMUL_DRAM_SHARDED
    const bool is_worker_core = get_arg_val<uint32_t>(0) == 1;
    // if not worker core, skip
    if (not is_worker_core) {
        return;
    }
#endif

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(4);  // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(5);                               // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(6);  // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);     // outer inner dim (in inner dim blocks)
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);         // outer inner dim (in inner dim blocks)
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);         // outer inner dim (in inner dim blocks)
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);  // out_subblock_h * out_subblock_w;
    constexpr uint32_t batch = get_compile_time_arg_val(13);                   // batch dim
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(14);     // number of tiles in out_block
    constexpr bool untilize_out = get_compile_time_arg_val(15);                // untilize output
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_compile_time_arg_val(16);
    constexpr bool in0_transpose_tile = (bool)get_compile_time_arg_val(17);

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

    constexpr uint32_t in0_dfb_id = in0_transpose_tile ? get_named_compile_time_arg_val("cb_in0_transposed")
                                                       : get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_dfb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t out_dfb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t mm_partials_dfb_id = get_named_compile_time_arg_val("cb_intermed0");
    // CB view the cross-block reload copies through: the UnpackToDestFp32-marked alias of the partials
    // CB when it is also read as an FPU operand (fused bias), otherwise the partials CB itself.
#ifdef MM_PARTIALS_RELOAD_ALIAS_CB
    // The partials CB is also read as an FPU operand (fused bias) and so cannot carry UnpackToDestFp32;
    // the reload instead copies through this alias view of the same SRAM, which does carry the flag.
    constexpr uint32_t mm_partials_reload_dfb_id = MM_PARTIALS_RELOAD_ALIAS_CB;
#else
    constexpr uint32_t mm_partials_reload_dfb_id = mm_partials_dfb_id;
#endif
    constexpr uint32_t untilize_mode_out_dfb_id = untilize_out ? mm_partials_dfb_id : out_dfb_id;
    // When in0 needs to be transposed, the original data is read from cb_in0 (in0_transpose_dfb_id),
    // transposed, and the result is written to cb_in0_transposed (in0_dfb_id), which is then used
    // as input for the matmul call.
    constexpr uint32_t in0_transpose_dfb_id = get_named_compile_time_arg_val("cb_in0");

    DataflowBuffer in0_dfb(in0_dfb_id);
    DataflowBuffer in1_dfb(in1_dfb_id);
    DataflowBuffer mm_partials_dfb(mm_partials_dfb_id);
    DataflowBuffer untilize_mode_out_dfb(untilize_mode_out_dfb_id);

#ifdef FUSE_BIAS
    constexpr uint32_t bias_dfb_id = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t bias_ntiles = get_named_compile_time_arg_val("bias_ntiles");
    constexpr uint32_t mm_out_dfb_id = mm_partials_dfb_id;
    // true: row-0 broadcast ([N] / [...,1,N]); false: elementwise add_tiles (bias has multiple M rows).
    constexpr bool row_broadcast_bias = (bool)get_compile_time_arg_val(18);
    DataflowBuffer bias_dfb(bias_dfb_id);
#else
    constexpr uint32_t mm_out_dfb_id = untilize_mode_out_dfb_id;
#endif
    DataflowBuffer mm_out_dfb(mm_out_dfb_id);

    // Number of valid in1 columns in the last in1 subblock. For the DRAM-sharded variant the
    // planner may pad per_core_N_compute beyond per_core_N_in1_sender so that out_subblock_w can be
    // larger; the reader only pushes per_core_N_in1_sender tiles per block into cb_in1. To avoid
    // reading those non-existent (padded) cb_in1 tiles, the compute kernel narrows the matmul_block
    // call on the last in1 subblock to last_subblock_w_valid lanes. When no padding occurs this
    // equals out_subblock_w and the original full-width path is preserved.
#ifdef MATMUL_DRAM_SHARDED
    constexpr uint32_t last_subblock_w_valid = get_named_compile_time_arg_val("last_subblock_w_valid");
#else
    constexpr uint32_t last_subblock_w_valid = out_subblock_w;
#endif
    constexpr bool last_subblock_padded = last_subblock_w_valid < out_subblock_w;

#ifdef SFPU_ACTIVATION
    constexpr KernelActivation activation_type =
        static_cast<KernelActivation>(get_named_compile_time_arg_val("activation_type"));
    constexpr uint32_t activation_param0 = get_named_compile_time_arg_val("activation_param0");
    constexpr uint32_t activation_param1 = get_named_compile_time_arg_val("activation_param1");
    constexpr uint32_t activation_param2 = get_named_compile_time_arg_val("activation_param2");

    ActivationInitHelper<activation_type, activation_param0, activation_param1>::init();
#endif

#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool spill = num_blocks_inner_dim > 1;

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_dfb_id, in1_dfb_id, mm_partials_dfb_id);
    matmul_block_init(in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);

#if defined(FUSED_ARGMAX) && defined(TRISC_PACK)
    // Snapshot the output CB geometry BEFORE any push (wr ptr still == base) and
    // load this worker's scan extent from the runtime args.
    fused_argmax::init(out_dfb_id, out_subblock_num_tiles, out_block_num_tiles);
#endif
    for (uint32_t b = 0; b < batch; b++) {
        if constexpr (get_batch_from_reader) {
            // Check whether this batch is valid
            bool is_batch_valid = false;
            UNPACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            MATH(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            PACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            if (!is_batch_valid) {
                continue;
            }
        }

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                bool enable_reload = false;

#ifdef PACK_RELU
                // for each batch we start with relu disabled so that intermediate results are not relu'd
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                }
#endif

                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((pack_reconfig_data_format(mm_partials_dfb_id)));
                }

                for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
                    bool last_out = block == (num_blocks_inner_dim - 1);
// Configure packer once for pack out without Bias
#if not defined FUSE_BIAS and defined PACK_RELU
                    if (last_out) {
                        // if last block we pack the final result with relu enabled
                        PACK((llk_pack_relu_config(ReluConfig::zero())));
                    }
#endif

                    if constexpr (in0_transpose_tile) {
                        reconfig_data_format_srca(in1_dfb_id, in0_transpose_dfb_id);
                        transpose_init(in0_transpose_dfb_id);
                        PACK((pack_reconfig_data_format(in0_dfb_id)));
#ifdef PACKER_L1_ACC
                        PACK((llk_pack_reconfig_l1_acc(0)));
#endif
                        transpose_tile_block<in0_block_num_tiles>(in0_transpose_dfb_id, in0_dfb_id);
                        reconfig_data_format_srca(in0_transpose_dfb_id, in1_dfb_id);
                        matmul_block_init(
                            in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                        PACK((pack_reconfig_data_format(mm_partials_dfb_id)));
                    }

                    in0_dfb.wait_front(in0_block_num_tiles);
                    in1_dfb.wait_front(in1_block_num_tiles);

                    int in0_index_subblock_offset = 0;
                    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                        int in1_index_subblock_offset = 0;
                        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                            // When last_subblock_padded is true the last in1 subblock has
                            // (out_subblock_w - last_subblock_w_valid) padded lanes whose cb_in1 tiles were
                            // never pushed by the reader. Narrow matmul_block so the unpacker only touches
                            // tiles that exist; the padded dst lanes are left at whatever the previous
                            // operation wrote there and the output writer (BRISC) drops those columns.
                            const bool is_last_in1_subblock_padded =
                                last_subblock_padded && (in1_subblock == in1_num_subblocks - 1);
                            const uint32_t effective_subblock_w =
                                is_last_in1_subblock_padded ? last_subblock_w_valid : out_subblock_w;

                            tile_regs_acquire();
                            if (enable_reload) {
                                reload_from_cb_to_dst(
                                    in0_dfb_id,
                                    in1_dfb_id,
                                    mm_partials_dfb_id,
                                    mm_partials_reload_dfb_id,
                                    in1_transpose_tile,
                                    out_subblock_num_tiles,
                                    out_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                            }

#ifndef SKIP_COMPUTE
                            // Compute output sub-block
                            uint32_t dst_index =
                                0;  // start at 0, each call to matmul_block internally increments dst_index
                            uint32_t in0_index = in0_index_subblock_offset;  // offset into in0 block
                            uint32_t in1_index = in1_index_subblock_offset;  // offset into in1 block
                            // inner dim that we accumulate is the inner dim of in0/in1, which is in0_block_w
                            for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                                // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                                // accumulation is done by iterating matmul_block across inner dim
                                // in0_block_w is passed as innder dim (kt) to matmul_block, internally used to stride
                                // in0
                                matmul_block(
                                    in0_dfb_id,
                                    in1_dfb_id,
                                    in0_index,
                                    in1_index,
                                    dst_index,
                                    in1_transpose_tile,
                                    effective_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                                in0_index++;               // stride right by 1
                                in1_index += in1_block_w;  // to stride down by 1 need to stride by in_per_core_w
                                                           // (should be called in1_block_w)
                            }

#endif  // SKIP_COMPUTE

                            if (last_out) {
                                tile_regs_commit();
                                mm_out_dfb.reserve_back(out_subblock_num_tiles);

#if defined SFPU_ACTIVATION and not defined FUSE_BIAS
                                apply_activation_from_pack<
                                    activation_type,
                                    activation_param0,
                                    activation_param1,
                                    activation_param2>(out_subblock_num_tiles);
#else
                                tile_regs_wait();
#endif

#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                                PACK((pack_reconfig_data_format(mm_out_dfb_id)));
#endif

#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                                if (block == 0) {  // no accumulation for first iteration
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
#else
                                PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif
                                uint32_t start_dst_index = 0;
                                pack_block(start_dst_index, mm_out_dfb_id, out_subblock_num_tiles);

                                tile_regs_release();
                                mm_out_dfb.push_back(out_subblock_num_tiles);

#if defined(FUSED_ARGMAX) && defined(TRISC_PACK)
                                // RVV argmax scan of the PREVIOUS subblock, in the shadow of
                                // the sequence just queued (defer-by-one-subblock).
                                fused_argmax::on_subblock_pushed();
#endif

                            } else {
                                tile_regs_commit();
                                mm_partials_dfb.reserve_back(out_subblock_num_tiles);
                                tile_regs_wait();

#ifdef PACKER_L1_ACC
                                if (block == 0) {  // no accumulation for first iteration
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else if (block == 1) {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                } else if (in0_transpose_tile) {
                                    // For each block, l1_acc would have been enabled during the
                                    // transpose stage. So let us put it back here.
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
#endif

                                uint32_t start_dst_index = 0;
                                pack_block(start_dst_index, mm_partials_dfb_id, out_subblock_num_tiles);

                                tile_regs_release();
                                mm_partials_dfb.push_back(out_subblock_num_tiles);
                            }

                            in1_index_subblock_offset += out_subblock_w;
                        }
                        in0_index_subblock_offset += in0_subblock_num_tiles;
                    }

#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                    if (block < num_blocks_inner_dim - 1) {
                        // Wait/pop in subblock-sized steps so the step size
                        // matches the bias section's wait_front(out_subblock_num_tiles),
                        // satisfying the CB API requirement that all wait_front
                        // increments on a given CB are identical.
                        for (uint32_t s = 0; s < out_block_num_tiles; s += out_subblock_num_tiles) {
                            mm_partials_dfb.wait_front(out_subblock_num_tiles);
                            mm_partials_dfb.pop_front(out_subblock_num_tiles);
                        }
                    }
                    // never reload when with bias, bias uses intermediate buffer
                    enable_reload = false;
#else
                    // Last iteration does spill and reload to output buffer
                    if (block < num_blocks_inner_dim - 2) {
                        for (uint32_t s = 0; s < out_block_num_tiles; s += out_subblock_num_tiles) {
                            mm_partials_dfb.wait_front(out_subblock_num_tiles);
                            mm_partials_dfb.pop_front(out_subblock_num_tiles);
                        }
                    }
                    if (block == num_blocks_inner_dim - 2) {
                        enable_reload = true;
                    }  // reload when last iteration
#endif
#else
                    if constexpr (spill) {
                        enable_reload = true;
                    }
#endif

                    in0_dfb.pop_front(in0_block_num_tiles);
                    in1_dfb.pop_front(in1_block_num_tiles);
                }

#ifdef FUSE_BIAS
#ifdef PACK_RELU
                // if last block we pack the final result with relu enabled
                PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                PACK((pack_reconfig_data_format(out_dfb_id)));
#endif
#ifdef PACKER_L1_ACC
                PACK((llk_pack_reconfig_l1_acc(0)));
#endif
                reconfig_data_format(in1_dfb_id, mm_partials_dfb_id, in0_dfb_id, bias_dfb_id);
                if constexpr (row_broadcast_bias) {
                    add_bcast_rows_init(mm_partials_dfb_id, bias_dfb_id);
                } else {
                    add_init(mm_partials_dfb_id, bias_dfb_id);
                }
                // Reader only pushes bias once when num_blocks_w_dim == 1;
                // the tiles stay in the CB for reuse across bh/batch iterations.
                if ((b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                    bias_dfb.wait_front(bias_ntiles);
                }
                for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                    int in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                        // See matmul stage: the last in1 subblock has padded lanes whose bias tile was
                        // never pushed by the reader. Redirect those out-of-range bias_tile_idx reads to
                        // tile 0 of cb_bias to keep them in-bounds; the resulting padded output columns
                        // are dropped by the writer.
                        const bool is_last_in1_subblock_padded =
                            last_subblock_padded && (in1_subblock == in1_num_subblocks - 1);
                        // Redundant wait since we know data was just pushed
                        mm_partials_dfb.wait_front(out_subblock_num_tiles);
                        tile_regs_acquire();
                        for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
#ifdef BIAS_FULL_BLOCK
                            // The bias CB holds a full [M, N] tile block. m_tile is this output tile's
                            // row within that block; bias_tile_idx is the position of the matching bias
                            // tile in the CB (row m_tile, column in1_index_subblock_offset). Only
                            // matmul_multicore_reuse_optimized loads the full block; other callers load a
                            // single bias row and use the N-only index below.
                            const uint32_t m_tile = in0_subblock * out_subblock_h + j;
                            uint32_t bias_tile_idx = m_tile * in1_block_w + in1_index_subblock_offset;
#else
                            uint32_t bias_tile_idx = in1_index_subblock_offset;
#endif
                            for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                                const uint32_t safe_bias_tile_idx =
                                    (is_last_in1_subblock_padded && k >= last_subblock_w_valid)
                                        ? 0u              // Padded output columns with tile 0 of cb_bias added are
                                        : bias_tile_idx;  // dropped by the writer.

                                if constexpr (row_broadcast_bias) {
                                    add_tiles_bcast_rows(mm_partials_dfb_id, bias_dfb_id, i, safe_bias_tile_idx, i);
                                } else {
                                    add_tiles(mm_partials_dfb_id, bias_dfb_id, i, safe_bias_tile_idx, i);
                                }
                                bias_tile_idx++;
                            }
                        }
                        tile_regs_commit();

                        mm_partials_dfb.pop_front(out_subblock_num_tiles);

                        // Pack out to output buffer
                        untilize_mode_out_dfb.reserve_back(out_subblock_num_tiles);

#ifdef SFPU_ACTIVATION
                        PACK(TTI_SEMWAIT(
                            p_stall::STALL_TDMA | p_stall::STALL_CFG,
                            semaphore::t6_sem(semaphore::MATH_PACK),
                            p_stall::STALL_ON_ZERO));

                        // Flip destination register offset for PACKER access
                        PACK(TT_SETC16(
                            DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            ActivationApplyHelper<activation_type, activation_param0, activation_param1>::apply(i);
                        }

                        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
#else
                        tile_regs_wait();
#endif
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, untilize_mode_out_dfb_id);
                        }
                        tile_regs_release();
                        untilize_mode_out_dfb.push_back(out_subblock_num_tiles);

                        in1_index_subblock_offset += out_subblock_w;
                    }
                }
                if constexpr (num_blocks_w_dim > 1) {
                    bias_dfb.pop_front(bias_ntiles);
                }
#endif  // FUSE_BIAS
                if constexpr (untilize_out) {
#ifdef PACK_RELU
                    PACK((llk_pack_relu_config(ReluConfig::none())));
#endif  // PACK_RELU
#ifndef FUSE_BIAS
                    reconfig_data_format_srca(in1_dfb_id, mm_partials_dfb_id);
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                    PACK((pack_reconfig_data_format(out_dfb_id)));
#endif
#ifdef PACKER_L1_ACC
                    PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif  // FUSE_BIAS
                    pack_untilize_dest_init<out_subblock_w, out_block_w>(out_dfb_id);
                    copy_tile_to_dst_init_short(mm_partials_dfb_id);
                    for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                        reblock_and_untilize<out_subblock_w, out_block_w>(
                            in1_num_subblocks, out_subblock_num_tiles, out_subblock_h, mm_partials_dfb_id, out_dfb_id);
                    }
                    pack_untilize_uninit(mm_partials_dfb_id);
                }
                if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
#ifdef FUSE_BIAS
                    // reconfigure unpacker df for src A and src B
                    reconfig_data_format(mm_partials_dfb_id, in1_dfb_id, bias_dfb_id, in0_dfb_id);
#else
                    // reconfigure unpacker df for src A
                    reconfig_data_format_srca(mm_partials_dfb_id, in1_dfb_id);
#endif
                    // reconfigure init for matmul
                    matmul_block_init(
                        in0_dfb_id, in1_dfb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        }
    }
#ifdef FUSE_BIAS
    // For num_blocks_w_dim == 1 the reader pushes bias once and the kernel holds it resident,
    // reusing it across all batch/bh/block iterations without popping. Pop it once here, after the
    // last use, so the CB is balanced. (For num_blocks_w_dim > 1 the per-block pop above already
    // balances each re-pushed bias block.)
    if constexpr (num_blocks_w_dim == 1) {
        bias_dfb.pop_front(bias_ntiles);
    }
#endif

#if defined(FUSED_ARGMAX) && defined(TRISC_PACK)
    // Scan the last outstanding subblock(s) and publish this worker's partials page.
    fused_argmax::finalize(get_named_compile_time_arg_val("cb_argmax_partials"));
#endif
}
