// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm_head_rope — compute.
//
// Block schedule (op_design.md), per block = 1 row-tile x valid_heads (<= block_heads) heads x head_tiles:
//   sum_squares_block  eltwise_chain BinaryFpu<Mul, x, x, D0, PerRow>     grid(hb, head_tiles) -> cb_sumsq (hb)
//   rsqrt_mean_block   packed-column rsqrt (raw LLK, see below)                               -> cb_rstd (hb, col 0)
//   normalize_block    eltwise_chain BinaryFpu<Mul, x, rstd Col-bcast> [+ DestReuseBinary<Mul, w_full, DEST_TO_SRCB>]
//                                                                         -> cb_normed (rope) | cb_output_tiles
//   rope_block         [HAS_ROPE] one DEST pass: (y @ T) * sin + y * cos (raw LLK, see below) -> cb_output_tiles
// Operand order (precision): every FPU multiply puts the wide (fp32-DEST / DEST-width-page) operand on srcB and the
// bf16 operand (x, w, cos, sin, T, column-select tiles) on srcA. Wide values on srcA lose low mantissa bits under the
// fidelity phases and biased the weight+rope output by ~-0.36% (rel RMS 0.0044 -> 0.0019 after the swap). A DEST
// tile moved into srcB by a DEST-reuse op takes srcB's current format, so srcB is configured from the fp32 CB first.
// normalize chain is DEST-blocked (block_size(DEST_AUTO_LIMIT), PerBlockSize pack).
// Prologue [HAS_WEIGHT]: expand_weight_rows — unary_bcast<Row> of the row-0 weight tiles -> cb_weight (resident).
// Row-tile loop is driven by cb_ctrl_compute: the reader publishes each row-tile it claimed (dynamic split, see
// reader) or the end marker; read_tile_value broadcasts the word to all three TRISCs.
//
// Raw-LLK deviation (rsqrt_mean_block) — Perf 1, measured: stage 15.5 -> 3.2 us math per core at 2304x3072 w+rope,
// op output bit-identical on 8 shapes. Bypasses compute_kernel_lib::reduce<SUM, REDUCE_ROW> + per-tile post-op:
// the helper produces one rstd tile per head, so the eps-add + rsqrt SFPU pass runs on hb full tiles although only
// column 0 is ever read. Instead matmul_tiles(sumsq_h, S_h) (S_h = 1/HEAD_DIM down column h, bf16) accumulates every
// head's row-mean into column h of ONE DEST tile, one add_unary + rsqrt pass (VectorMode::C: faces 0/2 = columns
// 0..15, hence block_heads <= 16) covers the whole block, and matmul_tiles(packed, E_h) (E_h = 1.0 at [h][0]) moves
// column h back to column 0 of rstd tile h. The move rounds rstd to srcB (tf32) width — exactly the width normalize's
// Col-broadcast srcB already reads it at. S_h / E_h are generated in L1 by the writer (idle at startup).
// Helper gap: reduce has no mode that writes row-reductions into a chosen output column of a shared tile.
//
// Raw-LLK deviation (rope_block) — Perf 1, measured: rotate+rope 21.8 -> 15.0 us math per core at 2304x3072 w+rope
// (whole op -3.9%), output unchanged with fp32 DEST. Per group of n <= DEST_AUTO_LIMIT tiles: matmul_tiles(y, T) ->
// D[j]; mul_reuse_dest_tiles<DEST_TO_SRCB>(sin) -> D[j] = sin * rot; mul_tiles(cos, y) -> D[j] += cos * y (ELWMUL
// accumulates into DEST on WH/BH). Bypassed: eltwise_chain (BinaryFpu / DestReuseBinary / AddBinary) + a separate
// rotate pass. Helper gap: the chain has no matmul element, so rot = y @ T cannot seed a DEST slot inside the chain's
// acquire; it had to be packed to an fp32 CB and re-unpacked, forcing lane width 2 and an SFPU add. What would close
// it: a chain element MatmulTile<in0 Block, in1 fixed tile, Dst::D0> usable before DestReuseBinary<Mul, sin Row,
// DEST_TO_SRCB>, and a BinaryFpu<Mul> allowed to accumulate into an occupied slot (ELWMUL's native D += a*b; today
// only DestAccumulation::PerRow exposes it). kernel_lib has no matmul helper on this branch.
//
// Tail blocks (valid_heads < block_heads): the reader pushes / the writer pops the full block_tiles window on
// cb_input_tiles / cb_output_tiles; compute-internal block CBs (cb_sumsq, cb_rstd, cb_normed) are realigned to a
// block boundary after each block, so every multi-tile window starts at a block-aligned ring offset and never wraps
// mid-window.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

using namespace compute_kernel_lib;

namespace {

// Advance a compute-internal CB (compute is both producer and consumer) by `pad` empty pages so the next
// block's window starts block-aligned.
template <class DFB>
ALWI void realign_block_cb(DFB& cb, uint32_t pad) {
    if (pad != 0) {
        cb.reserve_back(pad);
        cb.push_back(pad);
        cb.wait_front(pad);
        cb.pop_front(pad);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rope_cos = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rope_sin = get_compile_time_arg_val(2);
    constexpr uint32_t cb_weight_row = get_compile_time_arg_val(3);
    constexpr uint32_t cb_trans_mat = get_compile_time_arg_val(4);
    constexpr uint32_t cb_colsel = get_compile_time_arg_val(5);  // S_h tiles (block_heads), writer-generated
    constexpr uint32_t cb_weight = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sumsq = get_compile_time_arg_val(7);
    constexpr uint32_t cb_rstd = get_compile_time_arg_val(8);
    constexpr uint32_t cb_normed = get_compile_time_arg_val(9);
    constexpr uint32_t cb_colext = get_compile_time_arg_val(10);  // E_h tiles (block_heads), writer-generated
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t block_heads = get_compile_time_arg_val(12);
    constexpr uint32_t head_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t cb_rstd_packed = get_compile_time_arg_val(14);  // 1 DEST-width page
    constexpr uint32_t cb_ctrl_compute = get_compile_time_arg_val(15);
    constexpr bool dynamic_rows = get_compile_time_arg_val(16) != 0;
    constexpr uint32_t block_tiles = block_heads * head_tiles;
    static_assert(block_heads <= 16, "packed-column rsqrt: heads of a block must fit columns 0..15 (faces 0/2)");

    const uint32_t num_heads = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_sumsq);

    DataflowBuffer x(cb_input_tiles);
    DataflowBuffer colsel(cb_colsel);
    DataflowBuffer colext(cb_colext);
    DataflowBuffer rstd_packed(cb_rstd_packed);
    DataflowBuffer sumsq(cb_sumsq);
    DataflowBuffer rstd(cb_rstd);
    DataflowBuffer out(cb_output_tiles);

#ifdef HAS_WEIGHT
    // ---------------------------------------------------------------- expand_weight_rows (once per kernel)
    DataflowBuffer weight_row(cb_weight_row);
    DataflowBuffer weight(cb_weight);
    const uint32_t weight_tiles = num_heads * head_tiles;
    auto expand_weight_rows = [&]() {
        MaybeDeviceZoneScope("compute_weight_expand");
        unary_bcast<
            BroadcastDim::Row,
            input(cb_weight_row, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
            output(cb_weight, ReservePolicy::Upfront, PushPolicy::AtEnd)>(
            IterationShape::tiles(weight_tiles), weight_row, weight);
        weight.wait_front(weight_tiles);
    };
#endif

#ifdef HAS_ROPE
    DataflowBuffer cos(cb_rope_cos);
    DataflowBuffer sin(cb_rope_sin);
    DataflowBuffer trans(cb_trans_mat);
    DataflowBuffer normed(cb_normed);
    trans.wait_front(1);
    constexpr uint32_t cb_normalize_dst = cb_normed;
    auto& normalize_dst = normed;
#else
    constexpr uint32_t cb_normalize_dst = cb_output_tiles;
    auto& normalize_dst = out;
#endif

    colsel.wait_front(block_heads);
    colext.wait_front(block_heads);

#ifdef HAS_WEIGHT
    expand_weight_rows();
#endif
    const uint32_t blocks_per_row = (num_heads + block_heads - 1) / block_heads;
    DataflowBuffer ctrl(cb_ctrl_compute);
    for (uint32_t r = 0;; ++r) {
        if constexpr (dynamic_rows) {
            ctrl.wait_front(1);
            const uint32_t ctrl_word = read_tile_value(cb_ctrl_compute, 0, 0);
            ctrl.pop_front(1);
            if (ctrl_word == 0xFFFFFFFFu) {
                break;
            }
        } else if (r != 0) {
            break;  // Rg >= Rt: exactly one (static) row-tile per core, no ctrl handshake
        }
#ifdef HAS_ROPE
        {
            MaybeDeviceZoneScope("compute_wait_rope");
            cos.wait_front(head_tiles);
            sin.wait_front(head_tiles);
        }
#endif
        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            const uint32_t first_head = b * block_heads;
            const uint32_t hb = (num_heads - first_head) < block_heads ? (num_heads - first_head) : block_heads;
            const uint32_t bt = hb * head_tiles;

            {
                MaybeDeviceZoneScope("compute_wait_x");
                x.wait_front(block_tiles);
            }

            // ------------------------------------------------ sum_squares_block: per head, sum_d x_d * x_d in D0
            {
                MaybeDeviceZoneScope("compute_sumsq");
                eltwise_chain(
                    IterationShape::grid(hb, head_tiles),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(cb_input_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block),
                        input(cb_input_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block),
                        Dst::D0,
                        DestAccumulation::PerRow>{x, x},
                    PackTile<output(
                        cb_sumsq,
                        ReservePolicy::PerOuter,
                        PushPolicy::PerOuter,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Direct,
                        DestAccumulation::PerRow)>{sumsq});
            }

            // ------------------------------------------------ rsqrt_mean_block: rstd = rsqrt(sum/HEAD_DIM + eps)
            {
                MaybeDeviceZoneScope("compute_rsqrt");
                // row-mean of head h -> column h of DEST 0; one eps-add + rsqrt pass on columns 0..15.
                sumsq.wait_front(hb);
                reconfig_data_format<SrcOrder::Reverse>(cb_sumsq, cb_colsel);
                pack_reconfig_data_format(cb_rstd_packed);
                matmul_init(cb_sumsq, cb_colsel);
                tile_regs_acquire();
                for (uint32_t h = 0; h < hb; ++h) {
                    matmul_tiles(cb_sumsq, cb_colsel, h, h, 0);
                }
                binop_with_scalar_tile_init();
                add_unary_tile(0, eps_bits, VectorMode::C);
                rsqrt_tile_init();
                rsqrt_tile<RsqrtMode::Default>(0, VectorMode::C);
                tile_regs_commit();
                rstd_packed.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, cb_rstd_packed);
                tile_regs_release();
                rstd_packed.push_back(1);
                sumsq.pop_front(hb);
                // column h -> column 0 of rstd tile h.
                rstd_packed.wait_front(1);
                reconfig_data_format<SrcOrder::Reverse>(cb_rstd_packed, cb_colext);
                pack_reconfig_data_format(cb_rstd);
                matmul_init(cb_rstd_packed, cb_colext);
                for (uint32_t i = 0; i < hb; i += DEST_AUTO_LIMIT) {
                    const uint32_t n = (hb - i) < DEST_AUTO_LIMIT ? (hb - i) : DEST_AUTO_LIMIT;
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < n; ++j) {
                        matmul_tiles(cb_rstd_packed, cb_colext, 0, i + j, j);
                    }
                    tile_regs_commit();
                    rstd.reserve_back(n);
                    tile_regs_wait();
                    for (uint32_t j = 0; j < n; ++j) {
                        pack_tile(j, cb_rstd);
                    }
                    tile_regs_release();
                    rstd.push_back(n);
                }
                rstd_packed.pop_front(1);
            }

            // ------------------------------------------------ normalize_block: y = x * rstd [* w]
            {
                MaybeDeviceZoneScope("compute_normalize");
#ifdef HAS_WEIGHT
                eltwise_chain(
                    IterationShape::grid(hb, head_tiles).block_size(DEST_AUTO_LIMIT),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(cb_input_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block),
                        input(
                            cb_rstd, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col)>{
                        x, rstd},
                    DestReuseBinary<
                        BinaryFpuOp::Mul,
                        input(
                            cb_weight,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Block,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        DestReuseType::DEST_TO_SRCB>{weight, first_head * head_tiles},
                    PackTile<output(cb_normalize_dst, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{
                        normalize_dst});
#else
                eltwise_chain(
                    IterationShape::grid(hb, head_tiles).block_size(DEST_AUTO_LIMIT),
                    BinaryFpu<
                        BinaryFpuOp::Mul,
                        input(cb_input_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block),
                        input(
                            cb_rstd, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col)>{
                        x, rstd},
                    PackTile<output(cb_normalize_dst, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{
                        normalize_dst});
#endif
            }
            x.pop_front(block_tiles);
            realign_block_cb(sumsq, block_heads - hb);
            realign_block_cb(rstd, block_heads - hb);

#ifdef HAS_ROPE
            // ------------------------------------------------ rope_block [HAS_ROPE]: out = (y @ T) * sin + y * cos
            // One DEST pass per group of n <= DEST_AUTO_LIMIT tiles (raw LLK, see kernel head):
            //   matmul y @ T -> D[j]; D[j] = sin * D[j] (DEST reuse -> srcB); D[j] += cos * y (ELWMUL accumulates);
            //   pack D -> cb_output_tiles. srcA always holds bf16 (T, sin, cos), srcB the wide operand (y / DEST).
            {
                MaybeDeviceZoneScope("compute_rope");
                normed.wait_front(bt);
                reconfig_data_format_srca(cb_trans_mat);
                reconfig_data_format_srcb(cb_normed);
                pack_reconfig_data_format(cb_output_tiles);
                for (uint32_t i = 0; i < bt; i += DEST_AUTO_LIMIT) {
                    const uint32_t n = (bt - i) < DEST_AUTO_LIMIT ? (bt - i) : DEST_AUTO_LIMIT;
                    tile_regs_acquire();
                    matmul_init(cb_normed, cb_trans_mat);
                    for (uint32_t j = 0; j < n; ++j) {
                        matmul_tiles(cb_normed, cb_trans_mat, i + j, 0, j);
                    }
                    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_rope_sin);
                    for (uint32_t j = 0; j < n; ++j) {
                        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                            cb_rope_sin, (i + j) % head_tiles, j);
                    }
                    mul_init(cb_rope_cos, cb_normed);
                    for (uint32_t j = 0; j < n; ++j) {
                        mul_tiles(cb_rope_cos, cb_normed, (i + j) % head_tiles, i + j, j);
                    }
                    tile_regs_commit();
                    out.reserve_back(n);
                    tile_regs_wait();
                    for (uint32_t j = 0; j < n; ++j) {
                        pack_tile(j, cb_output_tiles);
                    }
                    tile_regs_release();
                    out.push_back(n);
                }
                normed.pop_front(bt);
            }
            realign_block_cb(normed, block_tiles - bt);
#endif
            // Pad the output window to block_tiles (writer pops whole blocks).
            if (bt < block_tiles) {
                out.reserve_back(block_tiles - bt);
                out.push_back(block_tiles - bt);
            }
        }
#ifdef HAS_ROPE
        cos.pop_front(head_tiles);
        sin.pop_front(head_tiles);
#endif
    }

#ifdef HAS_ROPE
    trans.pop_front(1);
#endif
#ifdef HAS_WEIGHT
    weight.pop_front(weight_tiles);
#endif
    colsel.pop_front(block_heads);
    colext.pop_front(block_heads);
}
