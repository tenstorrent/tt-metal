// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm compute. Per block of R_b tile-rows, per row, per stats window (stats_window_tiles: a whole number
// of the reader's C-tile chunks when streamed, else the whole row Wt):
//   residual_add_block : z = x + res                         -> work_a   (HAS_RESIDUAL)
//                        shifted (one_pass + residual): d = (x - x[:, 0]) + res instead
//   row_stats_window   : as the window's tiles land, S1 += z (and S2 += z * z when one_pass) in DEST;
//                        between windows the raw sums ride cb_stat_partial (fp32 tagged UnpackToDestFp32,
//                        lossless reload). Last window of the row: row collapse, then
//                        mean = S1 / W                         -> row_mean
//                        inv_std = rsqrt(S2 / W - mean^2 + eps) -> inv_std  (one_pass only)
//   then either the whole-row passes:
//   center_block       : c = z - mean (col bcast)            -> work_b
//   row_stats_block    : (two-pass only) S = sum_w c * c, collapse, rsqrt(S / W + eps) -> inv_std
//   normalize_block    : c * inv_std (col bcast)             -> work_a
//   affine_scale_block : * gamma (row bcast)                 -> work_b
//   affine_shift_block : + beta (row bcast)                  -> output
//   or, on cores the host picks (one_pass only), the same four passes per stats window through chunk_a /
//   chunk_b, so the writer starts at the row's first window.
//
// work_a / work_b alias one allocation; every whole-row phase moves whole blocks through each id so page i of
// one is page i of the other. The source of z stays resident (not popped) until centering.
//
// row_stats_window / row_stats_block use raw LLK. reduce() reads a resident block only after waiting all
// of it, or pops what it streams; neither keeps z resident while accumulating it chunk by chunk as the
// reader pushes, and it cannot carry the sum of squares alongside the sum. eltwise_chain's PerRow DEST
// accumulation has no post-accumulation hook, so collapsing and finalizing inside the accumulating window
// needs the window written out by hand.
//
// one_pass (host-gated to fp32 DEST + HiFi4): products are exact under HiFi4 and accumulate in fp32 DEST.
// Without a residual z = x is bf16-exact in SrcA and SrcB. A residual z read from the fp32 work page reaches
// SrcA and SrcB at different reduced precisions, which biased S2 against S1 enough to drive the variance
// negative at |mean| = 100; the per-row shift keeps the operands O(std) and removes that bias.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {
// rsqrt(x * scale + addend) on the SFPU loads holding tile column 0 only: a load covers 4 rows x 8
// columns (dest address bit 1 picks even / odd columns) and dst_reg[k] is address 2k, so column 0 is
// dst_reg[0], [2], [4], [6]. Half the rsqrt work of a VectorMode::C pass; normalize's col broadcast
// reads only column 0. Same rsqrt body as rsqrt_tile.
template <bool APPROXIMATION_MODE, bool fp32_dest_acc_en, uint32_t scale_bits>
inline void calculate_add_rsqrt_col0(uint32_t addend_bits) {
    const sfpi::vFloat scale = Converter::as_float(scale_bits);
    const sfpi::vFloat addend = Converter::as_float(addend_bits);
#pragma GCC unroll 4
    for (int group = 0; group < 4; ++group) {
        sfpi::vFloat y =
            _calculate_sqrt_body_<APPROXIMATION_MODE, true, false>(sfpi::dst_reg[2 * group] * scale + addend);
        if constexpr (!fp32_dest_acc_en) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[2 * group] = y;
    }
}

// Column 0 of DEST tiles 0 (S1) and 1 (S2, one_pass) after the row collapse: tile 0 <- mean = S1 * scale,
// tile 1 <- rsqrt(max(S2 * scale - mean^2, 0) + addend). Tile stride is 32 dst_reg rows.
template <bool APPROXIMATION_MODE, bool fp32_dest_acc_en, bool one_pass, uint32_t scale_bits>
inline void calculate_row_mean_inv_std_col0(uint32_t addend_bits) {
    constexpr uint32_t tile_rows = 32;
    const sfpi::vFloat scale = Converter::as_float(scale_bits);
#pragma GCC unroll 4
    for (int group = 0; group < 4; ++group) {
        sfpi::vFloat mean = sfpi::dst_reg[2 * group] * scale;
        if constexpr (!fp32_dest_acc_en) {
            mean = sfpi::convert<sfpi::vFloat16b>(mean, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[2 * group] = mean;
        if constexpr (one_pass) {
            const sfpi::vFloat addend = Converter::as_float(addend_bits);
            sfpi::vFloat var = sfpi::dst_reg[tile_rows + 2 * group] * scale - mean * mean;
            v_if(var < 0.0f) { var = 0.0f; }
            v_endif;
            sfpi::dst_reg[tile_rows + 2 * group] = _calculate_sqrt_body_<APPROXIMATION_MODE, true, false>(var + addend);
        }
    }
}
}  // namespace ckernel::sfpu
#endif

// lone_band_half_tile: an R3 core's lone band fills only the top half (faces 0-1) of its compute tile-row.
// Its passes run on a 16x32 TensorShape (num_faces = 2) in unpack, math and pack, so faces 2-3 are never
// touched. Raw LLK (the *_impl cores the CB-id API wraps): the CB-id compute API and the eltwise_chain /
// convenience helpers take the tile shape from the CB's compile-time num_faces table (constexpr per CB id),
// so a per-row face count on the same CBs is inexpressible through them. Single-tile stats copies, the SFPU
// row collapse and the row_shift copy stay full-tile (their bottom rows are stale and discarded).
namespace lone {
constexpr ckernel::TensorShape kHalfTile{16, 16, 1, 2};

#ifdef TRISC_UNPACK
ALWI uint32_t tile_address(uint32_t cb, uint32_t tile) {
    const auto& cb_interface = get_local_cb_interface(cb);
    return cb_interface.fifo_rd_ptr - 1 + cb_interface.fifo_page_size * tile;
}
#endif

template <EltwiseBinaryType op, BroadcastType bcast>
ALWI void binary_init(bool acc_to_dest = false) {
    UNPACK((llk_unpack_AB_init_impl<bcast>(kHalfTile, ckernel::Transpose::None)));
    MATH((llk_math_eltwise_binary_init_impl<op, bcast, MATH_FIDELITY>(kHalfTile, acc_to_dest)));
}

template <EltwiseBinaryType op, BroadcastType bcast>
ALWI void binary(uint32_t cb_a, uint32_t cb_b, uint32_t tile_a, uint32_t tile_b, uint32_t dst) {
    UNPACK((
        llk_unpack_AB_impl<bcast>(tile_address(cb_a, tile_a), tile_address(cb_b, tile_b), 0, unpack_src_format[cb_b])));
    MATH((llk_math_eltwise_binary_impl<op, bcast, DST_ACCUM_MODE, MATH_FIDELITY>(kHalfTile, dst, true)));
}

// DEST -> SrcA, the CB tile -> SrcB (DestReuseBinary DEST_TO_SRCA).
template <EltwiseBinaryType op>
ALWI void reuse_init(uint32_t cb) {
    UNPACK((llk_unpack_A_init_impl<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
        0, 0, kHalfTile, unpack_src_format[cb], unpack_dst_format[cb])));
    MATH((llk_math_eltwise_binary_init_impl<
          op,
          BroadcastType::NONE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::DEST_TO_SRCA>(kHalfTile, 0)));
}

template <EltwiseBinaryType op>
ALWI void reuse(uint32_t cb, uint32_t tile, uint32_t dst) {
    UNPACK((llk_unpack_A_impl<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
        tile_address(cb, tile), unpack_src_format[cb], unpack_dst_format[cb])));
    MATH((llk_math_eltwise_binary_impl<
          op,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::DEST_TO_SRCA>(kHalfTile, dst, true)));
}

// Packer MOP for faces 0-1; restore() re-programs the CB's own (full) shape.
ALWI void pack_half(uint32_t cb) {
    PACK((llk_pack_init_impl(pack_src_format[get_output_id(cb)], 16, 32, 2, 1, false)));
}
ALWI void pack_restore(uint32_t cb) { PACK((llk_pack_init(cb))); }

// One half-tile pass over `count` tiles of A (waited and popped per group of `lanes`) against B, packed to out.
// B: streamed (same index as A, popped with it), or resident at b_base (+ the A column when b_by_col).
// own_pack: the pass sets the half-tile packer MOP and restores it; false when the caller holds it for a run of passes.
enum class BSource { Streamed, Fixed, ByCol };

template <
    EltwiseBinaryType op,
    BroadcastType bcast,
    BSource b_src,
    uint32_t cb_a,
    uint32_t cb_b,
    uint32_t cb_out,
    uint32_t lanes,
    bool own_pack = true>
ALWI void pass(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& out, uint32_t count, uint32_t b_base = 0) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    if constexpr (own_pack) {
        pack_half(cb_out);
    }
    binary_init<op, bcast>();
    for (uint32_t col = 0; col < count; col += lanes) {
        const uint32_t n = count - col < lanes ? count - col : lanes;
        a.wait_front(n);
        if constexpr (b_src == BSource::Streamed) {
            b.wait_front(n);
        }
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t b_tile =
                b_src == BSource::Streamed ? i : (b_src == BSource::ByCol ? b_base + col + i : b_base);
            binary<op, bcast>(cb_a, cb_b, i, b_tile, i);
        }
        tile_regs_commit();
        out.reserve_back(n);
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        out.push_back(n);
        a.pop_front(n);
        if constexpr (b_src == BSource::Streamed) {
            b.pop_front(n);
        }
    }
    if constexpr (own_pack) {
        pack_restore(cb_out);
    }
}

// Shifted residual add: d = (x - shift col-bcast) + res, per group of lanes.
template <uint32_t cb_x, uint32_t cb_shift, uint32_t cb_res, uint32_t cb_out, uint32_t lanes, bool own_pack = true>
ALWI void shifted_add(DataflowBuffer& x, DataflowBuffer& res, DataflowBuffer& out, uint32_t count) {
    pack_reconfig_data_format(cb_out);
    if constexpr (own_pack) {
        pack_half(cb_out);
    }
    for (uint32_t col = 0; col < count; col += lanes) {
        const uint32_t n = count - col < lanes ? count - col : lanes;
        x.wait_front(n);
        res.wait_front(n);
        reconfig_data_format(cb_x, cb_shift);
        binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::COL>();
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            binary<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(cb_x, cb_shift, i, 0, i);
        }
        reconfig_data_format_srcb(cb_shift, cb_res);
        reuse_init<EltwiseBinaryType::ELWADD>(cb_res);
        for (uint32_t i = 0; i < n; ++i) {
            reuse<EltwiseBinaryType::ELWADD>(cb_res, i, i);
        }
        tile_regs_commit();
        out.reserve_back(n);
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        out.push_back(n);
        x.pop_front(n);
        res.pop_front(n);
    }
    if constexpr (own_pack) {
        pack_restore(cb_out);
    }
}

// row_stats_window on half tiles; `row` indexes the block's resident z.
template <
    uint32_t cb_z,
    uint32_t cb_stat_partial,
    uint32_t cb_row_mean,
    uint32_t cb_inv_std,
    uint32_t tensor_w_tiles,
    bool one_pass,
    uint32_t inv_w_bits,
    uint32_t epsilon_bits>
ALWI void stats_window(
    uint32_t row,
    uint32_t col0,
    uint32_t window_tiles,
    DataflowBuffer& z,
    DataflowBuffer& partial,
    DataflowBuffer& row_mean,
    DataflowBuffer& inv_std) {
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    constexpr uint32_t num_sums = one_pass ? 2 : 1;
    const uint32_t base = row * tensor_w_tiles + col0;
    const bool first = col0 == 0;
    const bool last = col0 + window_tiles == tensor_w_tiles;

    z.wait_front(base + window_tiles);
    reconfig_data_format(cb_z, cb_z);
    tile_regs_acquire();
    uint32_t tile = 0;
    if (!first) {
        partial.wait_front(num_sums);
        reconfig_data_format_srca(cb_z, cb_stat_partial);
        copy_init(cb_stat_partial);
        for (uint32_t sum = 0; sum < num_sums; ++sum) {
            copy_tile(cb_stat_partial, sum, sum);
        }
        reconfig_data_format_srca(cb_stat_partial, cb_z);
        partial.pop_front(num_sums);
    } else if (window_tiles % 2 != 0) {
        copy_init(cb_z);
        copy_tile(cb_z, base, 0);
        tile = 1;
    }
    binary_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE>(true);
    for (; tile < window_tiles; tile += 2) {
        binary<EltwiseBinaryType::ELWADD, BroadcastType::NONE>(cb_z, cb_z, base + tile, base + tile + 1, 0);
    }
    if constexpr (one_pass) {
        binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE>(true);
        for (uint32_t col = 0; col < window_tiles; ++col) {
            binary<EltwiseBinaryType::ELWMUL, BroadcastType::NONE>(cb_z, cb_z, base + col, base + col, 1);
        }
    }
    if (last) {
        sfpu_reduce_init<PoolType::SUM, dst_fmt>();
        for (uint32_t sum = 0; sum < num_sums; ++sum) {
            sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(sum, 1, 1);
        }
        rsqrt_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_row_mean_inv_std_col0,
            (APPROX, DST_ACCUM_MODE, one_pass, inv_w_bits),
            0,
            VectorMode::C,
            epsilon_bits));
    }
    tile_regs_commit();
    if (last) {
        pack_reconfig_data_format(cb_row_mean);
        row_mean.reserve_back(1);
        if constexpr (one_pass) {
            inv_std.reserve_back(1);
        }
        tile_regs_wait();
        pack_tile(0, cb_row_mean);
        if constexpr (one_pass) {
            pack_tile(1, cb_inv_std);
        }
        tile_regs_release();
        row_mean.push_back(1);
        if constexpr (one_pass) {
            inv_std.push_back(1);
        }
    } else {
        pack_reconfig_data_format(cb_stat_partial);
        partial.reserve_back(num_sums);
        tile_regs_wait();
        for (uint32_t sum = 0; sum < num_sums; ++sum) {
            pack_tile(sum, cb_stat_partial);
        }
        tile_regs_release();
        partial.push_back(num_sums);
    }
}

// row_stats_block (two-pass) on half tiles, block row `row` (c resident at the cb_centered front).
template <
    uint32_t cb_centered,
    uint32_t cb_inv_std,
    uint32_t tensor_w_tiles,
    uint32_t inv_w_bits,
    uint32_t epsilon_bits>
ALWI void row_stats(uint32_t row, DataflowBuffer& centered, DataflowBuffer& inv_std) {
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    const uint32_t base = row * tensor_w_tiles;
    centered.wait_front(base + tensor_w_tiles);
    reconfig_data_format(cb_centered, cb_centered);
    pack_reconfig_data_format(cb_inv_std);
    tile_regs_acquire();
    binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE>(true);
    for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
        binary<EltwiseBinaryType::ELWMUL, BroadcastType::NONE>(cb_centered, cb_centered, base + col, base + col, 0);
    }
    sfpu_reduce_init<PoolType::SUM, dst_fmt>();
    sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
    rsqrt_tile_init();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_add_rsqrt_col0,
        (APPROX, DST_ACCUM_MODE, inv_w_bits),
        0,
        VectorMode::C,
        epsilon_bits));
    tile_regs_commit();
    inv_std.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, cb_inv_std);
    tile_regs_release();
    inv_std.push_back(1);
}
}  // namespace lone

namespace {

using compute_kernel_lib::DataFormatReconfig;

// c (rows x Wt, resident at the cb_centered front, not popped) -> inv_std (one tile per row, column 0
// valid). Per row one DEST window: S = sum_w c * c (mul, acc_to_dest), sfpu_reduce ROW collapse, then
// rsqrt(S * 1/W + eps) on column 0.
template <
    uint32_t cb_centered,
    uint32_t cb_inv_std,
    uint32_t tensor_w_tiles,
    uint32_t inv_w_bits,
    uint32_t epsilon_bits>
ALWI void row_stats_block(uint32_t row0, uint32_t rows, DataflowBuffer& centered, DataflowBuffer& inv_std) {
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    centered.wait_front((row0 + rows) * tensor_w_tiles);
    reconfig_data_format(cb_centered, cb_centered);
    pack_reconfig_data_format(cb_inv_std);
    for (uint32_t row = row0; row < row0 + rows; ++row) {
        const uint32_t base = row * tensor_w_tiles;
        tile_regs_acquire();
        mul_init(cb_centered, cb_centered, true);  // a freshly acquired DEST reads 0: the first mul seeds S
        for (uint32_t col = 0; col < tensor_w_tiles; ++col) {
            mul_tiles(cb_centered, cb_centered, base + col, base + col, 0);
        }
        sfpu_reduce_init<PoolType::SUM, dst_fmt>();
        sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
        rsqrt_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_add_rsqrt_col0,
            (APPROX, DST_ACCUM_MODE, inv_w_bits),
            0,
            VectorMode::C,
            epsilon_bits));
        tile_regs_commit();
        inv_std.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_inv_std);
        tile_regs_release();
        inv_std.push_back(1);
    }
}

// One stats window: tiles [col0, col0 + window_tiles) of `row` at the z front (waited cumulatively, not popped).
// DEST 0 accumulates S1 (pairwise acc add), DEST 1 S2 (acc mul) when one_pass. A row's first window starts
// from a fresh (zero) DEST; later ones reload the raw sums from cb_stat_partial. An odd window occurs only
// as the whole row (Wt): its first tile seeds DEST 0 by copy.
template <
    uint32_t cb_z,
    uint32_t cb_stat_partial,
    uint32_t cb_row_mean,
    uint32_t cb_inv_std,
    uint32_t tensor_w_tiles,
    bool one_pass,
    uint32_t inv_w_bits,
    uint32_t epsilon_bits>
ALWI void row_stats_window(
    uint32_t row,
    uint32_t col0,
    uint32_t window_tiles,
    DataflowBuffer& z,
    DataflowBuffer& partial,
    DataflowBuffer& row_mean,
    DataflowBuffer& inv_std) {
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    constexpr uint32_t num_sums = one_pass ? 2 : 1;
    ASSERT(window_tiles % 2 == 0 || window_tiles == tensor_w_tiles);
    const uint32_t base = row * tensor_w_tiles + col0;
    const bool first = col0 == 0;
    const bool last = col0 + window_tiles == tensor_w_tiles;

    z.wait_front(base + window_tiles);
    reconfig_data_format(cb_z, cb_z);
    tile_regs_acquire();
    uint32_t tile = 0;
    if (!first) {
        partial.wait_front(num_sums);
        reconfig_data_format_srca(cb_z, cb_stat_partial);
        copy_init(cb_stat_partial);
        for (uint32_t sum = 0; sum < num_sums; ++sum) {
            copy_tile(cb_stat_partial, sum, sum);
        }
        reconfig_data_format_srca(cb_stat_partial, cb_z);
        partial.pop_front(num_sums);
    } else if (window_tiles % 2 != 0) {
        copy_init(cb_z);
        copy_tile(cb_z, base, 0);
        tile = 1;
    }
    add_init(cb_z, cb_z, true);
    for (; tile < window_tiles; tile += 2) {
        add_tiles(cb_z, cb_z, base + tile, base + tile + 1, 0);
    }
    if constexpr (one_pass) {
        mul_init(cb_z, cb_z, true);
        for (uint32_t col = 0; col < window_tiles; ++col) {
            mul_tiles(cb_z, cb_z, base + col, base + col, 1);
        }
    }
    if (last) {
        sfpu_reduce_init<PoolType::SUM, dst_fmt>();
        for (uint32_t sum = 0; sum < num_sums; ++sum) {
            sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(sum, 1, 1);
        }
        rsqrt_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_row_mean_inv_std_col0,
            (APPROX, DST_ACCUM_MODE, one_pass, inv_w_bits),
            0,
            VectorMode::C,
            epsilon_bits));
    }
    tile_regs_commit();
    if (last) {
        pack_reconfig_data_format(cb_row_mean);
        row_mean.reserve_back(1);
        if constexpr (one_pass) {
            inv_std.reserve_back(1);
        }
        tile_regs_wait();
        pack_tile(0, cb_row_mean);
        if constexpr (one_pass) {
            pack_tile(1, cb_inv_std);
        }
        tile_regs_release();
        row_mean.push_back(1);
        if constexpr (one_pass) {
            inv_std.push_back(1);
        }
    } else {
        pack_reconfig_data_format(cb_stat_partial);
        partial.reserve_back(num_sums);
        tile_regs_wait();
        for (uint32_t sum = 0; sum < num_sums; ++sum) {
            pack_tile(sum, cb_stat_partial);
        }
        tile_regs_release();
        partial.push_back(num_sums);
    }
}

// Streaming-pass operands: one CB handshake per IterationShape::block_size group of DEST lanes.
constexpr compute_kernel_lib::InputSpec lane_input(
    uint32_t cb, DataFormatReconfig reconfig = DataFormatReconfig::Enabled) {
    using namespace compute_kernel_lib;
    return input(cb, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block, reconfig);
}

constexpr compute_kernel_lib::OutputSpec lane_output(
    uint32_t cb, DataFormatReconfig reconfig = DataFormatReconfig::Enabled) {
    using namespace compute_kernel_lib;
    return output(cb, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize, reconfig);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t cb_row_work_a = get_compile_time_arg_val(4);
    constexpr uint32_t cb_row_work_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_row_mean = get_compile_time_arg_val(6);
    constexpr uint32_t cb_inv_std = get_compile_time_arg_val(7);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t block_row_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(10);
    constexpr bool has_residual = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(12);
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(13);
    // Tiles per DEST acquire on the streaming passes; the chain clamps it to DEST capacity.
    constexpr uint32_t dest_lanes = get_compile_time_arg_val(14);
    // Tiles per row-stats window: a multiple of the reader's chunk (stats start while the row lands) or Wt.
    constexpr uint32_t stats_window_tiles = get_compile_time_arg_val(15);
    constexpr bool one_pass = get_compile_time_arg_val(16) != 0;
    constexpr uint32_t cb_stat_partial = get_compile_time_arg_val(17);
    // stream_normalize (one_pass only): the per-window normalize + affine path is compiled in (two G-page
    // chunk CBs allocated).
    constexpr bool stream_normalize = get_compile_time_arg_val(18) != 0;
    constexpr uint32_t cb_chunk_a = get_compile_time_arg_val(19);
    constexpr uint32_t cb_chunk_b = get_compile_time_arg_val(20);
    static_assert(!stream_normalize || one_pass, "per-window normalize needs inv_std before centering");
    static_assert(tensor_w_tiles % stats_window_tiles == 0, "a stats window never straddles rows");
    // Leading tile-rows per core whose stats stream per window; later rows take one window per row (no
    // partial round trips).
    constexpr uint32_t streamed_rows = get_compile_time_arg_val(21);
    // one_pass with a residual stores d = (x - x[:, 0]) + res instead of z: shifted by each row's first element,
    // so the stats operands stay O(std) and SrcA / SrcB reading the fp32 work page at different precisions
    // cannot bias E[d^2] - E[d]^2 (unshifted, |mean| = 100 drove the variance negative). Centering subtracts
    // mean(d), so the output is unchanged.
    constexpr uint32_t cb_row_shift = get_compile_time_arg_val(22);
    constexpr bool shifted = one_pass && has_residual;
    constexpr bool half_lone = get_compile_time_arg_val(23) != 0;

    const uint32_t core_row_tiles = get_arg_val<uint32_t>(0);
    // Core row of the core's lone band (R3; first or last row of its run), or 0xFFFFFFFF.
    const uint32_t lone_row = get_arg_val<uint32_t>(2);
    // Per core: the host picks which cores take the per-window normalize (the work pair is allocated whenever
    // any core runs the whole-row passes).
    const bool per_window_normalize = stream_normalize && get_arg_val<uint32_t>(1) != 0;

    using namespace compute_kernel_lib;

    // Reconfig is dropped only where the format cannot change: work_a / work_b alias one allocation
    // with one page format, and gamma / beta share the I/O format (l1_ledger.md).
    constexpr auto same_work_fmt = DataFormatReconfig::Disabled;

    // Source of the mean and the centering: z = x + res when present, else x itself.
    constexpr uint32_t cb_z = has_residual ? cb_row_work_a : cb_input_tiles;

    compute_kernel_hw_startup(cb_input_tiles, cb_gamma_tiles, cb_output_tiles);

    DataflowBuffer input_tiles(cb_input_tiles);
    [[maybe_unused]] DataflowBuffer residual_tiles(cb_residual_tiles);
    DataflowBuffer gamma_tiles(cb_gamma_tiles);
    DataflowBuffer beta_tiles(cb_beta_tiles);
    DataflowBuffer work_a(cb_row_work_a);
    DataflowBuffer work_b(cb_row_work_b);
    DataflowBuffer row_mean(cb_row_mean);
    DataflowBuffer inv_std(cb_inv_std);
    DataflowBuffer stat_partial(cb_stat_partial);
    [[maybe_unused]] DataflowBuffer chunk_a(cb_chunk_a);
    [[maybe_unused]] DataflowBuffer chunk_b(cb_chunk_b);
    [[maybe_unused]] DataflowBuffer row_shift(cb_row_shift);
    DataflowBuffer output_tiles(cb_output_tiles);
    DataflowBuffer& z_tiles = has_residual ? work_a : input_tiles;

    constexpr uint32_t lone_lanes = dest_lanes < 4 ? dest_lanes : 4;
    for (uint32_t block_row = 0; block_row < core_row_tiles; block_row += block_row_tiles) {
        const uint32_t rows =
            (core_row_tiles - block_row) < block_row_tiles ? (core_row_tiles - block_row) : block_row_tiles;
        // The block's lone row (if any) runs on half tiles; every whole-row phase takes the full rows before it,
        // the lone row, then the full rows after it, so each phase still moves the whole block through its CBs.
        const bool has_lone = half_lone && lone_row >= block_row && lone_row < block_row + rows;
        const uint32_t lone_idx = has_lone ? lone_row - block_row : rows;
        const uint32_t rows_after = has_lone ? rows - lone_idx - 1 : 0;
        // Streaming passes take `dest_lanes` tiles per DEST acquire (one CB handshake per group).
        const auto lane_shape = [](uint32_t n) {
            return IterationShape::grid(n, tensor_w_tiles).block_size(dest_lanes);
        };
        // fn_full(first row, row count) over the full rows, fn_half() for the lone row, in block order.
        const auto by_rows = [&](auto fn_full, auto fn_half) {
            if (lone_idx > 0) {
                fn_full(0u, lone_idx);
            }
            if (has_lone) {
                fn_half();
                if (rows_after > 0) {
                    fn_full(lone_idx + 1, rows_after);
                }
            }
        };

        // residual_add_block + row_stats_window, interleaved per window so the stats run as tiles land
        {
            MaybeDeviceZoneScope("compute_add_and_stats");
            const uint32_t window = block_row < streamed_rows ? stats_window_tiles : tensor_w_tiles;
            const auto window_shape = IterationShape::grid(1, window).block_size(dest_lanes);
            for (uint32_t row = 0; row < rows; ++row) {
                if constexpr (shifted) {
                    // The row's first x tile (front of the input CB): column 0 is each row's shift.
                    copy<input(cb_input_tiles, WaitPolicy::Upfront, PopPolicy::None), output(cb_row_shift)>(
                        IterationShape::one_tile(), input_tiles, row_shift);
                }
                if (has_lone && row == lone_idx) {
                    MaybeDeviceZoneScope("compute_lone_add_and_stats");
                    // One half-tile packer MOP for the row; the stats tiles pack half too (only rows 0-15 are read).
                    lone::pack_half(cb_row_work_a);
                    for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += window) {
                        if constexpr (shifted) {
                            row_shift.wait_front(1);
                            lone::shifted_add<
                                cb_input_tiles,
                                cb_row_shift,
                                cb_residual_tiles,
                                cb_row_work_a,
                                lone_lanes,
                                false>(input_tiles, residual_tiles, work_a, window);
                        } else if constexpr (has_residual) {
                            lone::pass<
                                EltwiseBinaryType::ELWADD,
                                BroadcastType::NONE,
                                lone::BSource::Streamed,
                                cb_input_tiles,
                                cb_residual_tiles,
                                cb_row_work_a,
                                lone_lanes,
                                false>(input_tiles, residual_tiles, work_a, window);
                        }
                        lone::stats_window<
                            cb_z,
                            cb_stat_partial,
                            cb_row_mean,
                            cb_inv_std,
                            tensor_w_tiles,
                            one_pass,
                            inv_w_bits,
                            epsilon_bits>(row, col0, window, z_tiles, stat_partial, row_mean, inv_std);
                    }
                    lone::pack_restore(cb_row_work_a);
                    if constexpr (shifted) {
                        row_shift.pop_front(1);
                    }
                    continue;
                }
                for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += window) {
                    if constexpr (shifted) {
                        eltwise_chain(
                            window_shape,
                            BinaryFpu<
                                BinaryFpuOp::Sub,
                                lane_input(cb_input_tiles),
                                input(
                                    cb_row_shift,
                                    BroadcastDim::Col,
                                    WaitPolicy::Upfront,
                                    PopPolicy::None,
                                    InputTileMapping::Col)>{input_tiles, row_shift},
                            DestReuseBinary<
                                BinaryFpuOp::Add,
                                lane_input(cb_residual_tiles),
                                DestReuseType::DEST_TO_SRCA>{residual_tiles},
                            PackTile<lane_output(cb_row_work_a)>{work_a});
                    } else if constexpr (has_residual) {
                        add<lane_input(cb_input_tiles), lane_input(cb_residual_tiles), lane_output(cb_row_work_a)>(
                            window_shape, input_tiles, residual_tiles, work_a);
                    }
                    row_stats_window<
                        cb_z,
                        cb_stat_partial,
                        cb_row_mean,
                        cb_inv_std,
                        tensor_w_tiles,
                        one_pass,
                        inv_w_bits,
                        epsilon_bits>(row, col0, window, z_tiles, stat_partial, row_mean, inv_std);
                }
                if constexpr (shifted) {
                    row_shift.pop_front(1);
                }
            }
        }

        if (per_window_normalize) {
            if constexpr (stream_normalize) {
                // normalize -> affine per window, so the writer starts at the row's first window. Each pass
                // moves one window through a G-page chunk CB; the work alias pair cannot carry it (its ids
                // must move whole blocks).
                MaybeDeviceZoneScope("compute_stream_normalize");
                const auto window_shape = IterationShape::grid(1, stats_window_tiles).block_size(dest_lanes);
                constexpr auto col_stat = [](uint32_t cb) {
                    return input(
                        cb,
                        BroadcastDim::Col,
                        WaitPolicy::Upfront,
                        PopPolicy::None,
                        InputTileMapping::Col,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset);
                };
                constexpr auto affine_row = [](uint32_t cb) {
                    return input(
                        cb,
                        BroadcastDim::Row,
                        WaitPolicy::Upfront,
                        PopPolicy::None,
                        InputTileMapping::Row,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset);
                };
                gamma_tiles.wait_front(tensor_w_tiles);
                beta_tiles.wait_front(tensor_w_tiles);
                row_mean.wait_front(rows);
                inv_std.wait_front(rows);
                for (uint32_t row = 0; row < rows; ++row) {
                    if (has_lone && row == lone_idx) {
                        MaybeDeviceZoneScope("compute_lone_stream_normalize");
                        lone::pack_half(cb_output_tiles);
                        for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += stats_window_tiles) {
                            lone::pass<
                                EltwiseBinaryType::ELWSUB,
                                BroadcastType::COL,
                                lone::BSource::Fixed,
                                cb_z,
                                cb_row_mean,
                                cb_chunk_a,
                                lone_lanes,
                                false>(z_tiles, row_mean, chunk_a, stats_window_tiles, row);
                            lone::pass<
                                EltwiseBinaryType::ELWMUL,
                                BroadcastType::COL,
                                lone::BSource::Fixed,
                                cb_chunk_a,
                                cb_inv_std,
                                cb_chunk_b,
                                lone_lanes,
                                false>(chunk_a, inv_std, chunk_b, stats_window_tiles, row);
                            lone::pass<
                                EltwiseBinaryType::ELWMUL,
                                BroadcastType::ROW,
                                lone::BSource::ByCol,
                                cb_chunk_b,
                                cb_gamma_tiles,
                                cb_chunk_a,
                                lone_lanes,
                                false>(chunk_b, gamma_tiles, chunk_a, stats_window_tiles, col0);
                            lone::pass<
                                EltwiseBinaryType::ELWADD,
                                BroadcastType::ROW,
                                lone::BSource::ByCol,
                                cb_chunk_a,
                                cb_beta_tiles,
                                cb_output_tiles,
                                lone_lanes,
                                false>(chunk_a, beta_tiles, output_tiles, stats_window_tiles, col0);
                        }
                        lone::pack_restore(cb_output_tiles);
                        continue;
                    }
                    for (uint32_t col0 = 0; col0 < tensor_w_tiles; col0 += stats_window_tiles) {
                        eltwise_chain(
                            window_shape,
                            BinaryFpu<BinaryFpuOp::Sub, lane_input(cb_z), col_stat(cb_row_mean)>{
                                z_tiles, row_mean, 0, row},
                            PackTile<lane_output(cb_chunk_a)>{chunk_a});
                        eltwise_chain(
                            window_shape,
                            BinaryFpu<BinaryFpuOp::Mul, lane_input(cb_chunk_a), col_stat(cb_inv_std)>{
                                chunk_a, inv_std, 0, row},
                            PackTile<lane_output(cb_chunk_b)>{chunk_b});
                        eltwise_chain(
                            window_shape,
                            BinaryFpu<
                                BinaryFpuOp::Mul,
                                lane_input(cb_chunk_b, same_work_fmt),
                                affine_row(cb_gamma_tiles)>{chunk_b, gamma_tiles, 0, col0},
                            PackTile<lane_output(cb_chunk_a, same_work_fmt)>{chunk_a});
                        eltwise_chain(
                            window_shape,
                            BinaryFpu<
                                BinaryFpuOp::Add,
                                lane_input(cb_chunk_a, same_work_fmt),
                                affine_row(cb_beta_tiles)>{chunk_a, beta_tiles, 0, col0},
                            PackTile<lane_output(cb_output_tiles)>{output_tiles});
                    }
                }
                row_mean.pop_front(rows);
                inv_std.pop_front(rows);
            }
        } else {
            // center_block
            {
                MaybeDeviceZoneScope("compute_center");
                by_rows(
                    [&](uint32_t, uint32_t n) {
                        sub<lane_input(cb_z),
                            input(
                                cb_row_mean,
                                BroadcastDim::Col,
                                WaitPolicy::Upfront,
                                PopPolicy::AtEnd,
                                InputTileMapping::Col),
                            lane_output(cb_row_work_b)>(lane_shape(n), z_tiles, row_mean, work_b);
                    },
                    [&] {
                        MaybeDeviceZoneScope("compute_lone_center");
                        row_mean.wait_front(1);
                        lone::pass<
                            EltwiseBinaryType::ELWSUB,
                            BroadcastType::COL,
                            lone::BSource::Fixed,
                            cb_z,
                            cb_row_mean,
                            cb_row_work_b,
                            lone_lanes>(z_tiles, row_mean, work_b, tensor_w_tiles);
                        row_mean.pop_front(1);
                    });
            }

            // row_stats_block (two-pass; c stays resident for normalize_block)
            if constexpr (!one_pass) {
                MaybeDeviceZoneScope("compute_row_stats");
                by_rows(
                    [&](uint32_t row0, uint32_t n) {
                        row_stats_block<cb_row_work_b, cb_inv_std, tensor_w_tiles, inv_w_bits, epsilon_bits>(
                            row0, n, work_b, inv_std);
                    },
                    [&] {
                        MaybeDeviceZoneScope("compute_lone_row_stats");
                        lone::row_stats<cb_row_work_b, cb_inv_std, tensor_w_tiles, inv_w_bits, epsilon_bits>(
                            lone_idx, work_b, inv_std);
                    });
            }

            // normalize_block
            {
                MaybeDeviceZoneScope("compute_normalize");
                by_rows(
                    [&](uint32_t, uint32_t n) {
                        mul<lane_input(cb_row_work_b),
                            input(
                                cb_inv_std,
                                BroadcastDim::Col,
                                WaitPolicy::Upfront,
                                PopPolicy::AtEnd,
                                InputTileMapping::Col),
                            lane_output(cb_row_work_a)>(lane_shape(n), work_b, inv_std, work_a);
                    },
                    [&] {
                        MaybeDeviceZoneScope("compute_lone_normalize");
                        inv_std.wait_front(1);
                        lone::pass<
                            EltwiseBinaryType::ELWMUL,
                            BroadcastType::COL,
                            lone::BSource::Fixed,
                            cb_row_work_b,
                            cb_inv_std,
                            cb_row_work_a,
                            lone_lanes>(work_b, inv_std, work_a, tensor_w_tiles);
                        inv_std.pop_front(1);
                    });
            }
            {
                MaybeDeviceZoneScope("compute_wait_affine");
                gamma_tiles.wait_front(tensor_w_tiles);
                beta_tiles.wait_front(tensor_w_tiles);
            }

            // affine_scale_block
            {
                MaybeDeviceZoneScope("compute_affine_scale");
                by_rows(
                    [&](uint32_t, uint32_t n) {
                        mul<lane_input(cb_row_work_a, same_work_fmt),
                            input(
                                cb_gamma_tiles,
                                BroadcastDim::Row,
                                WaitPolicy::Upfront,
                                PopPolicy::None,
                                InputTileMapping::Row),
                            lane_output(cb_row_work_b, same_work_fmt)>(lane_shape(n), work_a, gamma_tiles, work_b);
                    },
                    [&] {
                        MaybeDeviceZoneScope("compute_lone_affine_scale");
                        lone::pass<
                            EltwiseBinaryType::ELWMUL,
                            BroadcastType::ROW,
                            lone::BSource::ByCol,
                            cb_row_work_a,
                            cb_gamma_tiles,
                            cb_row_work_b,
                            lone_lanes>(work_a, gamma_tiles, work_b, tensor_w_tiles, 0);
                    });
            }

            // affine_shift_block
            {
                MaybeDeviceZoneScope("compute_affine_shift");
                by_rows(
                    [&](uint32_t, uint32_t n) {
                        add<lane_input(cb_row_work_b, same_work_fmt),
                            input(
                                cb_beta_tiles,
                                BroadcastDim::Row,
                                WaitPolicy::Upfront,
                                PopPolicy::None,
                                InputTileMapping::Row,
                                same_work_fmt),
                            lane_output(cb_output_tiles)>(lane_shape(n), work_b, beta_tiles, output_tiles);
                    },
                    [&] {
                        MaybeDeviceZoneScope("compute_lone_affine_shift");
                        lone::pass<
                            EltwiseBinaryType::ELWADD,
                            BroadcastType::ROW,
                            lone::BSource::ByCol,
                            cb_row_work_b,
                            cb_beta_tiles,
                            cb_output_tiles,
                            lone_lanes>(work_b, beta_tiles, output_tiles, tensor_w_tiles, 0);
                    });
            }
        }
    }

    gamma_tiles.pop_front(tensor_w_tiles);
    beta_tiles.pop_front(tensor_w_tiles);
}
