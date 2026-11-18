// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// genify_matmul compute — block operations `matmul_accumulate_block` + `epilogue_block` of regime R1.
//
// Epilogue none: one compute_kernel_lib::matmul_block call per block of this core's region. The
// helper walks the num_k_blocks K-blocks (waiting / popping cb_act_tiles and cb_weight_tiles per
// K-block), accumulates the partial sums of the whole block in cb_partials with packer L1
// accumulation (TileRowMajor spill, exact one-block capacity), reloads them on the last K-block and
// packs the result row group by row group into cb_output_tiles. With a single K-block there is no
// accumulator: cb_output_tiles is passed as the helper's same-format interm placeholder.
// The ragged last K-block is handled by LastKBlockInnerDim (only last_block_k_tiles FMA steps).
//
// Epilogue != none (op_design.md Block Operation Realization 2', 3a-3c): matmul_block packs its last
// K-block into cb_partials (LastBlockTarget::Interm, the whole block stays fronted there), then ONE
// epilogue block op reads it and packs cb_output_tiles row by row:
//   bias / bias_relu  add_bias_bcast_rows<RowBroadcast, TileRowMajor> (+ PACK_RELU for bias_relu)
//   swiglu            eltwise_chain over (block_m, block_n/2): gate (Strided base 0) -> D0,
//                     up (Strided base block_n/2) -> D1, SwiGluMul (one fused SFPU pass, Perf 1,
//                     genify_matmul_swiglu_sfpu.hpp); the reader permuted B so all gate tiles
//                     of the N-block precede all up tiles. R4 keeps B in natural (gate, up) pair
//                     order instead and walks the block flat with row stride 2 (swiglu_natural_pairs).
//   addcmul           eltwise_chain over (block_m, block_n): FPU Mul with the row-broadcast gate, then
//                     SFPU AddBinary of the streamed residual (no DEST reuse ahead of an FPU op).
// The bias / gate row is waited once before the first batch's epilogue and popped after the last.
//
// SwiGluMul (Perf 2): rewritten as a hand-scheduled SFPLOADMACRO instruction stream -- isolated focus epilogue
// 61.8 -> 46.6 us (bf16 grade), fp32 grade 75.7 -> 54.9 us, same precision; RAW-LLK, justified at the head of
// genify_matmul_swiglu_sfpu.hpp; do NOT "fix" it back. The result now lands in the up slot (D1, out_slot).
// SwiGluMul (Perf 1): the chain library has no silu(a)*b element, so the op declares one through the chain's
// documented BinaryOp CRTP extension point (genify_matmul_swiglu_sfpu.hpp): out = silu(gate) * up in ONE raw-SFPI
// pass (branch-free sigmoid, Newton step folded into the product, rows software-pipelined in pairs). It replaced
// a local Silu element (silu_tile) + MulBinary: isolated focus epilogue 85.3 -> 61.8 us (bf16 grade), fp32
// output 116.9 -> 75.6 us (fp32 grade), with lower error on every measured metric. It is a chain element, not a
// bypass of the chain (the chain still owns init, DEST sync, CB sync and data-format reconfig).
//
// K deliveries (Perf 2 head ramp, genify_matmul_kschedule.hpp): both walks follow the reader's delivery schedule
// (LastKBlockInnerDim -> KS::kv); the raw walk reads A with the delivery's row stride under GM_COMPACT_A.
// K-walk under an epilogue (Perf 1): kw::matmul_accumulate_interm (genify_matmul_kwalk.hpp) replaces
// compute_kernel_lib::matmul_block<..., LastBlockTarget::Interm, ...>. RAW-LLK, justified by measurement
// (PERF FOCUS 1 whole op 466.3 -> 441.5 us, output bit-identical): every matmul_block flavour writes the
// DEST offset (SETC16) before each output tile, ~1 FPU cycle per 16-cycle LoFi tile-K; the raw walk advances
// DEST with an address modifier on the last MVMUL instead, and reserves / pushes cb_partials once per block
// (no per-K-block drain on the unpack thread). Do NOT "fix" it back to the helper. The helper keeps the
// cases the raw walk cannot express: sb_h > 1 subblocks, operand residency, and the epilogue-none
// LastBlockTarget::Out reload path.
//
// Regime R3 `block_sharded` (Refinement 5): K-block `own_kb` (this core's grid column) is its own resident
// A shard, pushed by the reader into cb_act_shard (backed on the shard, zero-copy); the helper's
// In0SourceFn selects that CB for the block (same data format as cb_act_tiles: both are A). Under the
// rotating row multicast (`own_round_slot`) the reader also pushes the own round's lockstep slot of
// cb_act_tiles; a PostKBlockFn retires it unread. Every other regime passes own_kb = NO_OWN_KB.
//
// Operand residency (Refinement 8, R1 / R1a one-K-block walks). `weight_resident`: the core's single B
// block (its whole column range) is pushed once and reused by every M-block of the walk -- matmul_block's
// in1 policy WaitAndRetainOnLastBlock (wait, never pop with one K-block) -- and popped here after the
// last block. `act_resident` (the twin): each M-block's A is pushed once and reused by its nb_n N-blocks
// (in0 WaitAndRetainOnLastBlock), popped here after the last one. The header hands those pops to the
// caller under this policy.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"  // AddBinary (addcmul)
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/genify_matmul_swiglu_sfpu.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/genify_matmul_kwalk.hpp"
#include "ttnn/ttnn/operations/genify_matmul/kernels/genify_matmul_kschedule.hpp"

// Per-stage device zones (Perf 1, PERMANENT -- free with the profiler off; never remove, extend instead).
// One execution per block (per batch slice under an epilogue). Each is recorded on all three TRISCs and
// is OCCUPANCY: cmp_matmul includes the helper's per-K-block cb_wait_front on A / B (unpack thread) and
// the reserve on cb_partials / cb_output_tiles (pack thread); cmp_epilogue includes the writer's
// back-pressure on cb_output_tiles. Ablate (SKIP_COMPUTE / CKL_ELTWISE_CHAIN_SKIP_COMPUTE) before ranking.
//   cmp_matmul     the block's whole K-walk (matmul_block)
//   cmp_epilogue   the fused epilogue over the accumulated block (bias / swiglu / addcmul)

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_act_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_weight_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(2);
constexpr uint32_t cb_partials = get_compile_time_arg_val(3);
constexpr uint32_t block_batches = get_compile_time_arg_val(4);
constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(5);
constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(6);
constexpr uint32_t sb_h = get_compile_time_arg_val(7);
constexpr uint32_t sb_w = get_compile_time_arg_val(8);
constexpr uint32_t block_k_tiles = get_compile_time_arg_val(9);
constexpr uint32_t num_k_blocks = get_compile_time_arg_val(10);
constexpr uint32_t last_block_k_tiles = get_compile_time_arg_val(11);
constexpr uint32_t block_n_tiles = get_compile_time_arg_val(12);
constexpr uint32_t epilogue = get_compile_time_arg_val(13);  // 0 none, 1 bias, 2 bias_relu, 3 swiglu, 4 addcmul
constexpr uint32_t cb_row = get_compile_time_arg_val(14);    // bias (1, 2) or gate (4) N-block row
constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(15);
constexpr uint32_t block_m_tiles = get_compile_time_arg_val(16);
constexpr uint32_t epilogue_block_tiles = get_compile_time_arg_val(17);  // chain DEST lanes per acquire
// 1: silu runs the full-precision (fp32) sigmoid; 0: the bf16-precision sigmoid (host picks 0 only
// for a non-fp32 output, whose own rounding dominates).
constexpr bool silu_fp32_precision = get_compile_time_arg_val(18) != 0;
// swiglu B-column order inside the block: 0 = gate half then up half (the R1 / R1a reader permutes per
// tile); 1 = natural (gate, up) tile pairs (R4 reads whole bank-shard rows in one coalesced read).
constexpr bool swiglu_natural_pairs = get_compile_time_arg_val(19) != 0;
constexpr uint32_t cb_act_shard = get_compile_time_arg_val(20);      // R3: A's resident shard (zero-copy)
constexpr bool own_round_slot = get_compile_time_arg_val(21) != 0;   // R3a: retire the own round's slot
constexpr bool weight_resident = get_compile_time_arg_val(22) != 0;  // B fronted once, reused per block
constexpr bool act_resident = get_compile_time_arg_val(23) != 0;     // A fronted once per M-block
static_assert(!(weight_resident || act_resident) || (num_k_blocks == 1 && block_batches == 1));
static_assert(!(weight_resident && act_resident));
constexpr uint32_t NO_OWN_KB = 0xFFFFFFFFu;

constexpr bool has_epilogue = epilogue != 0;
constexpr bool is_bias = epilogue == 1 || epilogue == 2;
constexpr bool is_bias_relu = epilogue == 2;
constexpr bool is_swiglu = epilogue == 3;
constexpr bool is_addcmul = epilogue == 4;
constexpr bool has_row = is_bias || is_addcmul;
constexpr uint32_t half_n = block_n_tiles / 2;
constexpr uint32_t block_tiles = block_m_tiles * block_n_tiles;
// swiglu walk over the block (Strided tile = base + r * row_stride + c). Permuted halves: grid
// (block_m, half_n), gate at row-stride block_n from 0, up from half_n. Natural pairs: the gate of
// output tile t = m * half_n + p is block tile m * block_n + 2p = 2t, its up 2t + 1, so a flat
// (block_m * half_n, 1) walk with row stride 2 reads both and packs t in row-major output order.
constexpr uint32_t swiglu_rows = swiglu_natural_pairs ? block_m_tiles * half_n : block_m_tiles;
constexpr uint32_t swiglu_cols = swiglu_natural_pairs ? 1 : half_n;
constexpr uint32_t swiglu_stride = swiglu_natural_pairs ? 2 : block_n_tiles;
constexpr uint32_t swiglu_up_base = swiglu_natural_pairs ? 1 : half_n;

// FMA step count per K-block: the balanced split leaves the last K-block narrower; the reader
// pushes the nominal block_k pages, so the garbage tail must never be summed.
// The K deliveries (Perf 2 head ramp, genify_matmul_kschedule.hpp): the identity [block_k.., last] without
// GM_K_RAMP; with it, the ramp head then whole block_k deliveries then the remainder.
using KS = gm_ks::Sched<block_k_tiles, num_k_blocks, last_block_k_tiles>;
static_assert(!gm_ks::ramp || num_k_blocks > 1, "a one-K-block walk keeps its single DEST accumulation");
struct LastKBlockInnerDim {
    ALWI uint32_t operator()(uint32_t block, uint32_t /*block_k*/) const { return KS::kv(block); }
};

// R3: K-block own_kb reads this core's resident A shard instead of cb_act_tiles.
struct OwnShardSource {
    uint32_t own_kb;
    ALWI uint32_t operator()(uint32_t block, uint32_t in0_cb_id) const {
        return block == own_kb ? cb_act_shard : in0_cb_id;
    }
};

// R3a: the own round's cb_act_tiles slot only kept the row's write pointers in lockstep; retire it.
struct RetireOwnRoundSlot {
    uint32_t own_kb;
    ALWI void operator()(uint32_t block, uint32_t /*num_k_blocks*/, bool /*is_last*/) const {
        if constexpr (own_round_slot) {
            if (block == own_kb) {
                CircularBuffer act_buf(cb_act_tiles);
                act_buf.wait_front(block_m_tiles * block_k_tiles);
                act_buf.pop_front(block_m_tiles * block_k_tiles);
            }
        }
    }
};

// Epilogue none: no accumulator CB when there is a single K-block; the helper only needs a
// same-format placeholder. Any epilogue always lands the last K-block in cb_partials.
constexpr uint32_t cb_interm = (has_epilogue || num_k_blocks > 1) ? cb_partials : cb_output_tiles;

// The raw K-walk (kw::matmul_accumulate_interm) is the epilogue path's walk wherever it can be expressed: it is
// written for 1 x sb_w subblocks over a whole block, with both operands waited / popped per K-block. Carve-outs
// (helper): sb_h > 1, and operand residency (WaitAndRetainOnLastBlock).
constexpr bool kwalk_expressible = sb_h == 1 && !act_resident && !weight_resident &&
                                   block_m_tiles == in0_num_subblocks && block_n_tiles == in1_num_subblocks * sb_w;
// Refinement 11 (host NONE_RAW_WALK -> GM_NONE_RAW_WALK): an epilogue-none multi-K-block walk takes the raw
// K-walk too -- it accumulates the whole block in cb_partials exactly as under an epilogue, and an identity
// epilogue (one CopyTile -> PackTile chain, row by row) moves it to cb_output_tiles. That copy replaces the
// helper's last-K-block reload of the partials, so the walk saves the helper's per-tile DEST-offset write.
#ifndef GM_NONE_RAW_WALK
#define GM_NONE_RAW_WALK 0
#endif
constexpr bool none_raw_walk = GM_NONE_RAW_WALK != 0 && !has_epilogue && num_k_blocks > 1 && kwalk_expressible;
// A compaction (head ramp) needs the raw walk: compute_kernel_lib::matmul_block strides A rows by block_k.
static_assert(!gm_ks::compact_a || (has_epilogue && kwalk_expressible), "GM_COMPACT_A needs the raw K-walk");

template <ckl::LastBlockTarget target>
ALWI void run_matmul(
    CircularBuffer& act_buf,
    CircularBuffer& weight_buf,
    CircularBuffer& out_buf,
    CircularBuffer& interm_buf,
    const ckl::MatmulBlockShape& shape,
    uint32_t own_kb) {
    ckl::matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/true,
        target,
        ckl::OutputCBLayout::TileRowMajor,
        ckl::matmul_config::InitMode::Short,
        act_resident ? ckl::InputPolicy::WaitAndRetainOnLastBlock : ckl::InputPolicy::WaitAndPopPerKBlock,
        weight_resident ? ckl::InputPolicy::WaitAndRetainOnLastBlock : ckl::InputPolicy::WaitAndPopPerKBlock,
        ckl::NoPostCompute,
        ckl::NoPreKBlock,
        RetireOwnRoundSlot,
        /*untilize_block_ct_dim=*/0,
        LastKBlockInnerDim,
        OwnShardSource>(
        act_buf,
        weight_buf,
        out_buf,
        interm_buf,
        shape,
        ckl::NoPostCompute{},
        ckl::NoPreKBlock{},
        /*in1_per_core_w=*/block_n_tiles,
        /*out_row_width=*/block_n_tiles,
        RetireOwnRoundSlot{own_kb},
        LastKBlockInnerDim{},
        OwnShardSource{own_kb});
}

// epilogue_block: cb_partials (one fronted block) -> cb_output_tiles (row by row).
ALWI void run_epilogue(
    CircularBuffer& partials_buf,
    CircularBuffer& row_buf,
    CircularBuffer& out_buf,
    DataflowBuffer& partials_dfb,
    DataflowBuffer& row_dfb,
    DataflowBuffer& residual_dfb,
    DataflowBuffer& out_dfb) {
    using namespace ckl;
    if constexpr (is_bias) {
        // The helper waits / pops cb_partials per row group; the bias row lifecycle is the caller's.
        if constexpr (is_bias_relu) {
            pack_relu_config(ReluConfig::zero());
        }
        add_bias_bcast_rows<BiasBroadcast::RowBroadcast, OutputCBLayout::TileRowMajor>(
            partials_buf,
            row_buf,
            out_buf,
            BiasAddShape::of(in0_num_subblocks, in1_num_subblocks, sb_h, sb_w, block_n_tiles));
        if constexpr (is_bias_relu) {
            pack_relu_config(ReluConfig::none());
        }
    } else if constexpr (is_swiglu) {
        // Strided addressing requires caller-managed (None, None) lifecycles on cb_partials.
        // SwiGluMul grade: fp32 (degree-4 2^f, two Newton steps) for an fp32 output on 32-bit DEST, else bf16.
        constexpr bool swiglu_fp32_grade = silu_fp32_precision && DST_ACCUM_MODE;
        partials_buf.wait_front(block_tiles);
        eltwise_chain(
            IterationShape::grid(swiglu_rows, swiglu_cols).block_size(epilogue_block_tiles),
            CopyTile<
                input(
                    cb_partials,
                    WaitPolicy::None,
                    PopPolicy::None,
                    InputTileMapping::Block,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Strided),
                Dst::D0>{partials_dfb, StridedTileRange{0, swiglu_stride}},
            CopyTile<
                input(
                    cb_partials,
                    WaitPolicy::None,
                    PopPolicy::None,
                    InputTileMapping::Block,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Strided),
                Dst::D1>{partials_dfb, StridedTileRange{swiglu_up_base, swiglu_stride}},
            SwiGluMul<swiglu_fp32_grade>{},  // result lands in the up slot (D1)
            PackTile<
                output(cb_output_tiles, ReservePolicy::PerOuter, PushPolicy::PerOuter),
                SwiGluMul<swiglu_fp32_grade>::out_slot>{out_dfb});
        partials_buf.pop_front(block_tiles);
    } else if constexpr (is_addcmul) {
        partials_buf.wait_front(block_tiles);
        eltwise_chain(
            IterationShape::grid(block_m_tiles, block_n_tiles),
            BinaryFpu<
                BinaryFpuOp::Mul,
                input(cb_partials, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block),
                input(cb_row, BroadcastDim::Row, WaitPolicy::None, PopPolicy::None, InputTileMapping::Row),
                Dst::D0>{partials_dfb, row_dfb},
            CopyTile<input(cb_residual_tiles), Dst::D1>{residual_dfb},
            AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
            PackTile<output(cb_output_tiles, ReservePolicy::PerOuter, PushPolicy::PerOuter)>{out_dfb});
        partials_buf.pop_front(block_tiles);
    }
}

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t own_kb = get_arg_val<uint32_t>(1);  // R3: this core's K shard; NO_OWN_KB elsewhere
    // A residency: blocks per M-block (N-blocks, n fastest); A is popped after each group. R1 only.
    [[maybe_unused]] const uint32_t nb_n = act_resident ? get_arg_val<uint32_t>(2) : 1;

    // matmul maps in0 -> SrcB, in1 -> SrcA: Reverse is mandatory.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_tiles, cb_weight_tiles, cb_output_tiles);

    CircularBuffer act_buf(cb_act_tiles), weight_buf(cb_weight_tiles), out_buf(cb_output_tiles), interm_buf(cb_interm);

    if constexpr (none_raw_walk) {
        DataflowBuffer partials_dfb(cb_partials), out_dfb(cb_output_tiles);
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            {
                MaybeDeviceZoneScope("cmp_matmul");
                kw::matmul_accumulate_interm<
                    sb_w,
                    in0_num_subblocks,
                    in1_num_subblocks,
                    block_k_tiles,
                    num_k_blocks,
                    last_block_k_tiles>(
                    cb_act_tiles, cb_weight_tiles, cb_partials, OwnShardSource{own_kb}, RetireOwnRoundSlot{own_kb});
            }
            pack_reconfig_l1_acc(0);  // the copy packs fresh tiles
            {
                MaybeDeviceZoneScope("cmp_epilogue");
                using namespace ckl;
                interm_buf.wait_front(block_tiles);
                eltwise_chain(
                    IterationShape::grid(block_m_tiles, block_n_tiles),
                    CopyTile<input(cb_partials, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block), Dst::D0>{
                        partials_dfb},
                    PackTile<output(cb_output_tiles, ReservePolicy::PerOuter, PushPolicy::PerOuter)>{out_dfb});
                interm_buf.pop_front(block_tiles);
            }
        }
    } else if constexpr (!has_epilogue) {
        const auto shape = ckl::MatmulBlockShape::of(
            in0_num_subblocks, in1_num_subblocks, sb_h, sb_w, block_k_tiles, KS::n, block_batches);
#ifdef GM_NSPLIT_LAST_N
        // t2_progressive_nsplit: the core's LAST N-block of each M-block is ragged (GM_NSPLIT_LAST_N < block_n columns
        // of the nominal region). Its last in1 subblock runs the FMA on only the valid columns (the helper's
        // last_in1_subblock_w_valid: matmul ct_dim narrowed, DEST / pack / every CB count stay full width), so the
        // unwritten column(s) of the output row groups are garbage the writer never writes (invalid columns). The
        // helper cannot drop whole subblocks (its L1-acc partials drain counts assume the full in1 width), so when the
        // ragged part does not fit the block's last subblock, the ragged block runs a 1 x block_n subblock instead
        // (the shape is a runtime argument; the output / partials row-group pushes change granularity, not totals).
        // The narrowing is only safe on a ONE-ROW subblock: a 2 x 2 subblock narrowed to ct_dim 1 (FLUX qkv 256 x 6144
        // x 9216, block 8 x 2) hung inside the matmul LLK (unpack waiting for math's context, math mid-matmul_block).
        static_assert(GM_NSPLIT_LAST_N <= block_n_tiles);
        constexpr bool ragged_same_sb = sb_h == 1 && (in1_num_subblocks - 1) * sb_w < GM_NSPLIT_LAST_N;
        static_assert(ragged_same_sb || block_n_tiles <= ckl::DEST_AUTO_LIMIT);
        auto shape_last = ragged_same_sb
                              ? shape
                              : ckl::MatmulBlockShape::of(
                                    in0_num_subblocks * sb_h, 1, 1, block_n_tiles, block_k_tiles, KS::n, block_batches);
        shape_last.last_in1_subblock_w_valid =
            ragged_same_sb ? GM_NSPLIT_LAST_N - (in1_num_subblocks - 1) * sb_w : GM_NSPLIT_LAST_N;
        const uint32_t nb_n_walk = get_arg_val<uint32_t>(2);
#endif
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const bool act_last_use = !act_resident || (block_idx + 1) % nb_n == 0;
            {
                MaybeDeviceZoneScope("cmp_matmul");
#ifdef GM_NSPLIT_LAST_N
                const bool ragged = (block_idx % nb_n_walk) == nb_n_walk - 1;
                run_matmul<ckl::LastBlockTarget::Out>(
                    act_buf, weight_buf, out_buf, interm_buf, ragged ? shape_last : shape, own_kb);
#else
                run_matmul<ckl::LastBlockTarget::Out>(act_buf, weight_buf, out_buf, interm_buf, shape, own_kb);
#endif
            }
            if constexpr (act_resident) {
                if (act_last_use) {
                    act_buf.pop_front(block_m_tiles * block_k_tiles);  // the M-block's A, after its last N-block
                }
            }
        }
    } else {
        // One batch slice per matmul call: cb_partials holds exactly one block, and the epilogue
        // must drain it before the next slice is accumulated.
        const auto shape = ckl::MatmulBlockShape::of(
            in0_num_subblocks, in1_num_subblocks, sb_h, sb_w, block_k_tiles, KS::n, /*batch=*/1);
        CircularBuffer row_buf(has_row ? cb_row : cb_partials);
        DataflowBuffer partials_dfb(cb_partials), row_dfb(has_row ? cb_row : cb_partials),
            residual_dfb(is_addcmul ? cb_residual_tiles : cb_partials), out_dfb(cb_output_tiles);
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            for (uint32_t bi = 0; bi < block_batches; ++bi) {
                {
                    MaybeDeviceZoneScope("cmp_matmul");
                    if constexpr (!kwalk_expressible) {
                        run_matmul<ckl::LastBlockTarget::Interm>(
                            act_buf, weight_buf, out_buf, interm_buf, shape, own_kb);
                    } else {
                        kw::matmul_accumulate_interm<
                            sb_w,
                            in0_num_subblocks,
                            in1_num_subblocks,
                            block_k_tiles,
                            num_k_blocks,
                            last_block_k_tiles>(
                            cb_act_tiles,
                            cb_weight_tiles,
                            cb_partials,
                            OwnShardSource{own_kb},
                            RetireOwnRoundSlot{own_kb});
                    }
                }
                // The helper leaves packer L1 accumulation on after a multi-K-block walk; the
                // epilogue packs fresh tiles (bias helper prerequisite: caller disables L1_ACC).
                pack_reconfig_l1_acc(0);
                if constexpr (has_row) {
                    if (bi == 0) {
                        row_buf.wait_front(block_n_tiles);
                    }
                }
                {
                    MaybeDeviceZoneScope("cmp_epilogue");
                    run_epilogue(interm_buf, row_buf, out_buf, partials_dfb, row_dfb, residual_dfb, out_dfb);
                }
            }
            if constexpr (has_row) {
                row_buf.pop_front(block_n_tiles);
            }
            if constexpr (act_resident) {
                if ((block_idx + 1) % nb_n == 0) {
                    act_buf.pop_front(block_m_tiles * block_k_tiles);  // the M-block's A, after its last N-block
                }
            }
        }
    }
    if constexpr (weight_resident) {
        if (num_blocks != 0) {
            weight_buf.pop_front(block_k_tiles * block_n_tiles);  // the retained B block, after its last use
        }
    }
}
