// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_qkv_rotary_heads — compute.
//
// matmul_accumulate_block + matmul_finish_rotate_block: ONE compute_kernel_lib::matmul_block call per M block covers
// all K blocks (block_m_tiles x lane_n_tiles output, K = num_k_blocks x k_block_tiles). Packer L1 accumulation in fp32
// cb_partials over K blocks 0 .. n-2; the last K block reloads each subblock's partial into DEST, accumulates the
// last K tiles (the last of which is the bias tile: x's ones column x W's bias row), and packs once to bf16
// cb_heads_out (LastBlockTarget::Out). out_subblock_h = 1, so cb_heads_out is tile-row-major.
//
// q/k cores (rotary = 1): subblock = one head (1 x 2: DEST 0-1). The PostComputeFn hook copies the row's C and S
// tiles (fp32 cb_rotary_table) into DEST 2-3 and rotates the head in fp32 DEST on the SFPU:
//     DEST0 = t0 * C - t1 * S,  DEST1 = t1 * C + t0 * S
// before the helper's single bf16 pack. v cores (rotary = 0): plain matmul + bias, widest DEST-fitting subblock.
//
// Raw-LLK justification (custom SFPU rotate_head_pair): no kernel_lib element computes a 4-in / 2-out map inside a
// matmul's DEST window. eltwise_chain owns its own acquire/commit window and cannot run inside the matmul hook, and
// the two-operand SFPU binaries (SubBinary / MulBinary) would need a fifth fp32 DEST slot to keep t0 alive for out1
// (half-sync fp32 DEST has 4). The C/S copy is the documented copy_init / copy_tile pair.
// Rotary placement (knob rotary_on_pack): 1 = PACKER thread (the design): the math thread only copies C/S into DEST
// 2-3 in PostComputeFn; the rotation runs as a custom packer functor in matmul_block's Activation slot
// (is_custom_pack_activation, sfpu_activation_helpers.hpp) on the committed DEST half while the math thread runs the
// next head's FMAs. 0 = MATH thread inside PostComputeFn (op_design.md lamp L2 fallback).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {

// fp32 DEST: one tile is 32 sfpi dst_reg rows (each 32 datums). Tiles 0..3 of the current half are t0, t1, C, S;
// C and S align element-wise with t0 / t1 (same position rows, same frequency columns).
constexpr uint32_t kDestRegsPerTile = 32;

// Static addressing: every SFPLOAD / SFPSTORE uses a compile-time offset (tile k = k * kDestRegsPerTile) from the
// DEST RWC base, and the walk advances the base with dst_reg++ (one static INCRWC). A loop-variant index
// (dst_reg[i]) makes the compiler build each load/store word at run time and push it through the instruction-buffer
// MMIO (add + sw per access), so the issuing TRISC — not the SFPU — bounds the rotate. The caller resets the RWC
// afterwards (clear_dst_reg_addr / _llk_math_eltwise_sfpu_done_).
inline void rotate_head_pair() {
#pragma GCC unroll 8
    for (uint32_t i = 0; i < kDestRegsPerTile; ++i) {
        const sfpi::vFloat t0 = sfpi::dst_reg[0];
        const sfpi::vFloat t1 = sfpi::dst_reg[kDestRegsPerTile];
        const sfpi::vFloat c = sfpi::dst_reg[2 * kDestRegsPerTile];
        const sfpi::vFloat s = sfpi::dst_reg[3 * kDestRegsPerTile];
        sfpi::dst_reg[0] = t0 * c - t1 * s;
        sfpi::dst_reg[kDestRegsPerTile] = t1 * c + t0 * s;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH || TRISC_PACK

namespace ckl = compute_kernel_lib;

// matmul_finish_rotate_block hook (math thread, last K block). Called once per subblock in (in0 subblock, in1
// subblock) order: the M row of the subblock is calls / in1_num_subblocks. Puts the row's C and S into DEST 2-3;
// with ROTATE_HERE it also rotates before commit (math-thread fallback).
// CS_ON_RELOAD (knob cs_on_reload, default 1): the C/S copy runs in the helper's post-reload slot, right after the
// subblock's fp32 partials reload, on the datacopy config the reload already set up (tables are fp32 like the
// partials) — no SrcA reconfig and no copy_init of its own. 0: copied after the FMAs, with its own SrcA reconfig
// (in1 bf16 -> fp32) and copy_init.
template <
    uint32_t cb_table,
    uint32_t cb_weights,
    uint32_t in1_num_subblocks,
    uint32_t head_tiles,
    bool ROTATE_HERE,
    bool CS_ON_RELOAD>
struct RotaryPostCompute {
    static constexpr bool post_reload_hook = CS_ON_RELOAD;
    static constexpr bool kAblateCs = get_named_compile_time_arg_val("ablate_rotary") == 1;
    uint32_t calls = 0;

    ALWI void copy_cs() const {
        const uint32_t row = calls / in1_num_subblocks;
        copy_tile(cb_table, 2 * row, head_tiles);          // C -> DEST 2
        copy_tile(cb_table, 2 * row + 1, head_tiles + 1);  // S -> DEST 3
    }
    // Helper post-reload slot (CS_ON_RELOAD): SrcA / unpacker hold the fp32 partials' datacopy config.
    ALWI void post_reload(uint32_t /*out_subblock_num_tiles*/) const {
        if constexpr (!kAblateCs) {
            copy_cs();
        }
    }
    ALWI void operator()(uint32_t /*out_subblock_num_tiles*/) {
        if constexpr (kAblateCs) {
            return;
        }
        if constexpr (!CS_ON_RELOAD) {
            // The helper left SrcA in in1's (bf16) format after the reload; the tables are fp32 like the partials.
            reconfig_data_format_srca(cb_weights, cb_table);
            copy_init(cb_table);
            copy_cs();
        }
        ++calls;
        if constexpr (ROTATE_HERE) {
            MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused>([]() {})));
            MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::rotate_head_pair, 0, VectorMode::RC_custom)));
        }
    }
};

// Packer-thread rotary: a custom functor for matmul_block's Activation slot. Replaces tile_regs_wait(): waits for
// the math thread's commit, points the packer thread's SFPU at the committed DEST half, rotates DEST 0-1 with C/S in
// DEST 2-3, and stalls the packer until the SFPU is done (mirrors apply_activation_from_pack).
struct RotaryFromPack {
    static constexpr bool custom_pack_apply = true;
    static constexpr KernelActivation activation = KernelActivation::NONE;
    static ALWI void init() {
        PACK((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused>([]() {})));
    }
    static ALWI void apply_from_pack(uint32_t /*out_subblock_num_tiles*/) {
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
        if constexpr (get_named_compile_time_arg_val("ablate_rotary") == 0) {  // 2: keep the copy, drop the SFPU
            // SFPU base = tile 0 of the packer's (committed) DEST half. Unlike _llk_math_eltwise_sfpu_start_ this
            // does NOT STALLWAIT(STALL_SFPU, MATH): the MATH_PACK semaphore above is posted by the math thread only
            // once MATH and SFPU are idle (set_math_semaphores), so the committed half is complete; waiting for
            // the FPU to go idle again would serialize the rotation behind the next head's FMAs on the other half.
            PACK((ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0)));
            PACK((ckernel::sfpu::rotate_head_pair()));
            PACK((ckernel::math::clear_dst_reg_addr()));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    }
};

// Waits for the M block's C/S tiles right before the last K block (they arrive a whole K loop earlier).
template <uint32_t cb_table, uint32_t table_tiles>
struct WaitTablesBeforeLastBlock {
    ALWI void operator()(uint32_t /*block*/, uint32_t /*num_k_blocks*/, bool is_last) const {
        if (is_last) {
            cb_wait_front(cb_table, table_tiles);
        }
    }
};

// The last K block of `shape` alone, reloading the partials an earlier call left in the interm buffer.
constexpr ckl::MatmulBlockShape last_k_block_reloading(ckl::MatmulBlockShape shape) {
    shape.num_k_blocks = 1;
    shape.reload_on_first_k_block = true;
    return shape;
}

void kernel_main() {
    constexpr uint32_t cb_x_block = get_named_compile_time_arg_val("cb_x_block");
    constexpr uint32_t cb_weights = get_named_compile_time_arg_val("cb_weights");
    constexpr uint32_t cb_rotary_table = get_named_compile_time_arg_val("cb_rotary_table");
    constexpr uint32_t cb_partials = get_named_compile_time_arg_val("cb_partials");
    constexpr uint32_t cb_heads_out = get_named_compile_time_arg_val("cb_heads_out");
    constexpr uint32_t block_m_tiles = get_named_compile_time_arg_val("block_m_tiles");
    constexpr uint32_t k_block_tiles = get_named_compile_time_arg_val("k_block_tiles");
    constexpr uint32_t num_k_blocks = get_named_compile_time_arg_val("num_k_blocks");
    constexpr uint32_t lane_n_tiles = get_named_compile_time_arg_val("lane_n_tiles");
    constexpr uint32_t out_subblock_w = get_named_compile_time_arg_val("out_subblock_w");
    constexpr bool rotary = get_named_compile_time_arg_val("rotary") == 1;
    constexpr bool rotary_on_pack = get_named_compile_time_arg_val("rotary_on_pack") == 1;
    constexpr uint32_t out_subblock_h = 1;  // keeps cb_heads_out tile-row-major for the writer
    constexpr uint32_t in1_num_subblocks = lane_n_tiles / out_subblock_w;
    constexpr uint32_t head_tiles = get_named_compile_time_arg_val("head_tiles");          // DEST slots of t0, t1
    constexpr uint32_t table_tiles = get_named_compile_time_arg_val("table_block_tiles");  // C, S of each row
    static_assert(lane_n_tiles % out_subblock_w == 0);
    static_assert(table_tiles == 2 * block_m_tiles, "cb_rotary_table holds one C and one S tile per row");
    static_assert(
        !rotary || out_subblock_w == head_tiles || get_named_compile_time_arg_val("ablate_rotary") == 1,
        "a q/k subblock is exactly one head");
    using RotaryHook = RotaryPostCompute<
        cb_rotary_table,
        cb_weights,
        in1_num_subblocks,
        head_tiles,
        !rotary_on_pack,
        get_named_compile_time_arg_val("cs_on_reload") == 1>;
    // The C/S copy rides the last K block's partials reload, which exists only when there is more than one K block.
    static_assert(!rotary || num_k_blocks > 1 || get_named_compile_time_arg_val("cs_on_reload") == 0);
    using RotaryPack = std::conditional_t<rotary_on_pack, RotaryFromPack, ckl::NoneActivation>;
    using TableWait = WaitTablesBeforeLastBlock<cb_rotary_table, table_tiles>;
    // TileRowMajor (knob out_layout_row_major): partial spills / reloads and output pushes go per tile row
    // (lane_n_tiles tiles) instead of per subblock, and no whole-block output reserve at K block 0. With
    // out_subblock_h = 1 both layouts put the same tiles at the same cb_heads_out offsets.
    constexpr auto kOutLayout = get_named_compile_time_arg_val("out_layout_row_major") == 1
                                    ? ckl::OutputCBLayout::TileRowMajor
                                    : ckl::OutputCBLayout::SubblockMajor;

    const uint32_t num_m_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_block, cb_weights, cb_heads_out);
    if constexpr (rotary && rotary_on_pack) {
        RotaryFromPack::init();  // packer-thread SFPU init: boot-time, the caller's job (as ActivationInitHelper)
    }

    CircularBuffer x_buf(cb_x_block);
    CircularBuffer w_buf(cb_weights);
    CircularBuffer partials_buf(cb_partials);
    CircularBuffer out_buf(cb_heads_out);
    constexpr auto shape = ckl::MatmulBlockShape::of(
        block_m_tiles, in1_num_subblocks, out_subblock_h, out_subblock_w, k_block_tiles, num_k_blocks);
    // q/k K loop in two phases (perf lamp L1) when the accumulation subblock differs from the head subblock:
    // accum_shape = K blocks 0 .. n-2 at 1 x accum_subblock_w; finish_shape = the last K block at 1 x head, reloading
    // the partials the first phase left in cb_partials. TileRowMajor partials are row-grouped at absolute column
    // offsets, so the two phases' subblock widths need not match. Otherwise one call covers all K blocks.
    constexpr uint32_t accum_subblock_w = get_named_compile_time_arg_val("accum_subblock_w");
    constexpr bool split_k_loop = rotary && accum_subblock_w != out_subblock_w && num_k_blocks > 1;
    static_assert(lane_n_tiles % accum_subblock_w == 0);
    static_assert(
        !split_k_loop || kOutLayout == ckl::OutputCBLayout::TileRowMajor,
        "per-phase subblock widths need row-grouped (TileRowMajor) partials");
    constexpr auto accum_shape = ckl::MatmulBlockShape::of(
        block_m_tiles,
        lane_n_tiles / accum_subblock_w,
        out_subblock_h,
        accum_subblock_w,
        k_block_tiles,
        num_k_blocks - 1);
    constexpr auto finish_shape = split_k_loop ? last_k_block_reloading(shape) : shape;
    // Output push granularity of the q/k finish call (knob out_drain_tiles, the writer's drain step). The finish call
    // is a single K block, so SubblockMajor reserves / pushes each head right after its pack (no whole-block reserve);
    // with out_subblock_h = 1 it lays the tiles out exactly as TileRowMajor does, and its per-subblock reload of the
    // row-grouped partials reads them in the same order. Per-head pushes let the writer drain a row's first heads
    // while its last heads compute. TileRowMajor (one push per row) when the writer drains whole rows.
    constexpr bool finish_push_per_head =
        split_k_loop && get_named_compile_time_arg_val("out_drain_tiles") < lane_n_tiles;
    static_assert(!finish_push_per_head || get_named_compile_time_arg_val("out_drain_tiles") % out_subblock_w == 0);
    constexpr auto kFinishLayout = finish_push_per_head ? ckl::OutputCBLayout::SubblockMajor : kOutLayout;

    for (uint32_t m_block = 0; m_block < num_m_blocks; ++m_block) {
        if constexpr (get_named_compile_time_arg_val("ablate_compute_all") == 1) {
            // ABLATION ONLY: the CB protocol of one M block with no unpack / math / pack.
            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                cb_wait_front(cb_x_block, block_m_tiles * k_block_tiles);
                cb_wait_front(cb_weights, k_block_tiles * lane_n_tiles);
                cb_pop_front(cb_x_block, block_m_tiles * k_block_tiles);
                cb_pop_front(cb_weights, k_block_tiles * lane_n_tiles);
            }
            if constexpr (rotary) {
                cb_wait_front(cb_rotary_table, table_tiles);
                cb_pop_front(cb_rotary_table, table_tiles);
            }
            cb_reserve_back(cb_heads_out, block_m_tiles * lane_n_tiles);
            cb_push_back(cb_heads_out, block_m_tiles * lane_n_tiles);
            continue;
        }
        if constexpr (split_k_loop) {
            // K blocks 0 .. n-2 at the wide accumulation subblock: packer L1 accumulation into cb_partials, the
            // last of them left pushed (row-grouped) for the finish call below to reload.
            ckl::matmul_block<false, true, ckl::LastBlockTarget::Interm, kOutLayout>(
                x_buf, w_buf, out_buf, partials_buf, accum_shape);
        }
        if constexpr (rotary) {
            ckl::matmul_block<
                false,
                true,
                ckl::LastBlockTarget::Out,
                kFinishLayout,
                ckl::matmul_config::InitMode::Short,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                RotaryHook,
                TableWait,
                ckl::NoPostKBlock,
                0,
                ckl::NoKBlockInnerDimFn,
                ckl::NoIn0Source,
                ckl::NoIn1BaseOffset,
                false,
                RotaryPack>(x_buf, w_buf, out_buf, partials_buf, finish_shape, RotaryHook{}, TableWait{});
            cb_pop_front(cb_rotary_table, table_tiles);
        } else {
            ckl::matmul_block<false, true, ckl::LastBlockTarget::Out, kOutLayout>(
                x_buf, w_buf, out_buf, partials_buf, shape);
        }
    }
}
