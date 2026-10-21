// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the modulated deformable conv: the sampling geometry, the bilinear columns and the
// conv's matmul in one kernel, so the columns never leave L1.
//
// Kernel positions run outermost, the core's blocks of 32 output rows inside them; an input wider
// than one chunk of channels takes a pass per chunk at each position. During the first pass each
// block's geometry is computed on the SFPU two blocks ahead of its corners, so a reader gathers one
// block while compute works through another, from inputs staged by the reader that owns the block
// (even blocks the first, odd the second), and handed back to it, which keeps it for every pass.
// For each (pass, block) the reader's four corners are tilized on the unpacker, multiplied by their
// scalar columns and summed in fp32 dest, which gives the block's columns of that pass's chunk; a
// matmul with the pass's chunk x C_out slice of the weight then accumulates into the block's output
// tiles, in fp32 dest within the pass and in bfloat16 L1 across passes. After the last pass the
// bias is added, the ReLU applied, and the bfloat16 output handed to the block's reader.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/relu.h"

constexpr uint32_t corner_block_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t output_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t channel_tiles = get_compile_time_arg_val(3);  // input channels per pass / 32
constexpr uint32_t kernel_positions = get_compile_time_arg_val(4);
constexpr uint32_t tiled_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t geom_in_cb_index = get_compile_time_arg_val(6);
constexpr uint32_t geom_index_cb_index = get_compile_time_arg_val(7);
constexpr uint32_t geom_weight_cb_index = get_compile_time_arg_val(8);
constexpr uint32_t geom_kernel_cb_index = get_compile_time_arg_val(9);
// fp32 bits of the input height and width, and of each less one.
constexpr uint32_t IN_H_FP32 = get_compile_time_arg_val(10);
constexpr uint32_t IN_H_LESS_ONE_FP32 = get_compile_time_arg_val(11);
constexpr uint32_t IN_W_FP32 = get_compile_time_arg_val(12);
constexpr uint32_t IN_W_LESS_ONE_FP32 = get_compile_time_arg_val(13);
// The second reader's buffers, for the odd blocks; the ones above serve the even ones.
constexpr uint32_t odd_corner_block_cb_index = get_compile_time_arg_val(14);
constexpr uint32_t odd_scalar_cb_index = get_compile_time_arg_val(15);
constexpr uint32_t odd_geom_index_cb_index = get_compile_time_arg_val(16);
constexpr uint32_t odd_geom_weight_cb_index = get_compile_time_arg_val(17);
constexpr uint32_t odd_output_cb_index = get_compile_time_arg_val(18);
// The block's columns of one kernel position, the weight slices, the output accumulators and the
// bias.
constexpr uint32_t column_cb_index = get_compile_time_arg_val(19);
constexpr uint32_t conv_weight_cb_index = get_compile_time_arg_val(20);
constexpr uint32_t acc_cb_index = get_compile_time_arg_val(21);
constexpr uint32_t bias_cb_index = get_compile_time_arg_val(22);
constexpr uint32_t OUT_TILES = get_compile_time_arg_val(23);  // C_out / 32
constexpr bool HAS_BIAS = get_compile_time_arg_val(24);
constexpr bool RELU = get_compile_time_arg_val(25);
constexpr uint32_t CHUNKS = get_compile_time_arg_val(26);  // passes per kernel position
// The odd blocks' geometry inputs, staged by their reader; geom_in_cb holds the even ones'.
constexpr uint32_t odd_geom_in_cb_index = get_compile_time_arg_val(27);
constexpr uint32_t geom_scratch_cb_index = get_compile_time_arg_val(28);
constexpr uint32_t CORNERS = 4;
constexpr uint32_t BLOCK_TILES = CORNERS * channel_tiles;
constexpr uint32_t WEIGHT_SLICE_TILES = channel_tiles * OUT_TILES;
// Tiles per dest acquire: a half of half-sync dest holds four fp32 tiles.
constexpr uint32_t DEST_TILES = 4;

// Tiles of geom_in_cb, per block, and of geom_kernel_cb, once, in the reader's order.
constexpr uint32_t IN_DY = 0;
constexpr uint32_t IN_DX = 1;
constexpr uint32_t IN_MASK = 2;  // the logits
constexpr uint32_t IN_ORIGIN_Y = 3;
constexpr uint32_t IN_ORIGIN_X = 4;
constexpr uint32_t GEOM_IN_TILES = 5;
constexpr uint32_t KERNEL_Y = 0;
constexpr uint32_t KERNEL_X = 1;

constexpr uint32_t FP32_ZERO = 0x00000000u;
constexpr uint32_t FP32_ONE = 0x3F800000u;
constexpr uint32_t FP32_MINUS_ONE = 0xBF800000u;

namespace {

// The geometry's SFPU ops, as the compute API runs them but over GEOM_FACES. A geometry tile holds
// one kernel position per column, so with at most 16 positions faces 1 and 3 carry nothing the
// readers use, and the SFPU skips them.
constexpr VectorMode GEOM_FACES = kernel_positions <= 16 ? VectorMode::C : VectorMode::RC;

template <ckernel::BinaryOp OP>
inline void geom_binary(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, OP, 8 /* ITERATIONS */, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        GEOM_FACES)));
}
inline void geom_add(uint32_t a, uint32_t b, uint32_t out) { geom_binary<ckernel::BinaryOp::ADD>(a, b, out); }
inline void geom_sub(uint32_t a, uint32_t b, uint32_t out) { geom_binary<ckernel::BinaryOp::SUB>(a, b, out); }
inline void geom_mul(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8 /* ITERATIONS */, DST_ACCUM_MODE),
        a,
        b,
        out,
        GEOM_FACES)));
}
inline void geom_ge(uint32_t dst, uint32_t scalar) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_ge, (APPROX, 8 /* ITERATIONS */), dst, GEOM_FACES, scalar));
}
inline void geom_lt(uint32_t dst, uint32_t scalar) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_lt, (APPROX, 8 /* ITERATIONS */), dst, GEOM_FACES, scalar));
}
inline void geom_floor(uint32_t dst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_floor_, (APPROX, 8 /* ITERATIONS */), dst, GEOM_FACES));
}
inline void geom_mul_scalar(uint32_t dst, uint32_t scalar) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 8 /* ITERATIONS */, DST_ACCUM_MODE),
        dst,
        GEOM_FACES,
        scalar));
}
inline void geom_rsub_scalar(uint32_t dst, uint32_t scalar) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, RSUB_UNARY, 8 /* ITERATIONS */, DST_ACCUM_MODE),
        dst,
        GEOM_FACES,
        scalar));
}

inline void copy_values(uint32_t from, uint32_t to) {
    copy_dest_values_init();
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        copy_dest_value,
        (DataFormat::Float32, false /* APPROXIMATE */),
        from,
        to,
        0 /* unused */,
        GEOM_FACES)));
}

// dst *= (lo <= src < hi), a factor of 1.0 or 0.0; scratch is overwritten.
inline void zero_outside(uint32_t dst, uint32_t src, uint32_t scratch, uint32_t lo_fp32, uint32_t hi_fp32) {
    copy_values(src, scratch);
    unary_ge_tile_init();
    geom_ge(scratch, lo_fp32);
    mul_binary_tile_init();
    geom_mul(dst, scratch, dst);
    copy_values(src, scratch);
    unary_lt_tile_init();
    geom_lt(scratch, hi_fp32);
    mul_binary_tile_init();
    geom_mul(dst, scratch, dst);
}

// Loads one axis of the sample point into dest tiles 0 to 2 and leaves p = origin + kernel offset
// + offset in tile 0 and its floor p0 in tile 1.
inline void axis_point(uint32_t in_cb_index, uint32_t offset_tile, uint32_t origin_tile, uint32_t kernel_tile) {
    copy_init(in_cb_index);
    copy_tile(in_cb_index, offset_tile, 0);
    copy_tile(geom_kernel_cb_index, kernel_tile, 2);
    // The reader writes each row's window origin in column 0 only.
    unary_bcast_init<BroadcastType::COL>(in_cb_index);
    unary_bcast<BroadcastType::COL>(in_cb_index, origin_tile, 1);
    unary_bcast_uninit<BroadcastType::COL>(in_cb_index);

    add_binary_tile_init();
    geom_add(0, 1, 0);
    geom_add(0, 2, 0);
    copy_values(0, 1);
    rounding_op_tile_init();
    geom_floor(1);
}

// One axis's weights from the point axis_point leaves: the far weight l = p - p0 in tile 0 and the
// near weight (1 - l) in tile 3, each zeroed where its corner lies outside [0, size); p0 stays in
// tile 1 and tile 2 is scratch.
inline void axis_weights(uint32_t size_fp32, uint32_t size_less_one_fp32) {
    sub_binary_tile_init();
    geom_sub(0, 1, 0);
    copy_values(0, 3);
    binop_with_scalar_tile_init();
    geom_rsub_scalar(3, FP32_ONE);
    zero_outside(3, 1, 2, FP32_ZERO, size_fp32);
    zero_outside(0, 1, 2, FP32_MINUS_ONE, size_less_one_fp32);
}

// The block's geometry in four rounds of dest, each at most four fp32 tiles, its intermediates
// kept in fp32 L1 between rounds: the rows' weights times the mask, then the top-left corner's page
// y0 * W + x0 and the columns' weights, then the corners' weights, two a round. A corner's weight is
// its row weight times the mask times its column weight.
void compute_geometry(uint32_t in_cb_index, uint32_t index_cb_index, uint32_t weight_cb_index) {
    CircularBuffer geom_in_cb(in_cb_index);
    CircularBuffer index_cb(index_cb_index);
    CircularBuffer weight_cb(weight_cb_index);
    CircularBuffer scratch_cb(geom_scratch_cb_index);
    geom_in_cb.wait_front(GEOM_IN_TILES);
    reconfig_data_format_srca(in_cb_index);

    // Scratch: y0 * W, then the near and far rows' weights times the mask.
    tile_regs_acquire();
    axis_point(in_cb_index, IN_DY, IN_ORIGIN_Y, KERNEL_Y);
    axis_weights(IN_H_FP32, IN_H_LESS_ONE_FP32);
    copy_values(1, 2);
    binop_with_scalar_tile_init();
    geom_mul_scalar(2, IN_W_FP32);
    copy_init(in_cb_index);
    copy_tile(in_cb_index, IN_MASK, 1);
    sigmoid_tile_init();
    sigmoid_tile<GEOM_FACES>(1);  // the mask, from its logits
    mul_binary_tile_init();
    geom_mul(3, 1, 3);
    geom_mul(0, 1, 0);
    tile_regs_commit();
    scratch_cb.reserve_back(3);
    tile_regs_wait();
    pack_reconfig_data_format(geom_scratch_cb_index);
    pack_tile(2, geom_scratch_cb_index);
    pack_tile(3, geom_scratch_cb_index);
    pack_tile(0, geom_scratch_cb_index);
    tile_regs_release();
    scratch_cb.push_back(3);

    // The top-left corner's page; scratch: the near and far columns' weights.
    tile_regs_acquire();
    axis_point(in_cb_index, IN_DX, IN_ORIGIN_X, KERNEL_X);
    axis_weights(IN_W_FP32, IN_W_LESS_ONE_FP32);
    scratch_cb.wait_front(3);
    copy_init(geom_scratch_cb_index);
    copy_tile(geom_scratch_cb_index, 0, 2);
    scratch_cb.pop_front(1);
    add_binary_tile_init();
    geom_add(2, 1, 2);
    typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_int32, (APPROX, 8 /* ITERATIONS */), 2, GEOM_FACES));
    tile_regs_commit();
    geom_in_cb.pop_front(GEOM_IN_TILES);
    index_cb.reserve_back(1);
    scratch_cb.reserve_back(2);
    tile_regs_wait();
    pack_reconfig_data_format(index_cb_index);
    pack_tile(2, index_cb_index);
    pack_reconfig_data_format(geom_scratch_cb_index);
    pack_tile(3, geom_scratch_cb_index);
    pack_tile(0, geom_scratch_cb_index);
    tile_regs_release();
    index_cb.push_back(1);
    scratch_cb.push_back(2);

    // Scratch now holds near row, far row, near column, far column; corners 0 and 1 lie on the
    // near row, 2 and 3 on the far one.
    scratch_cb.wait_front(4);
    weight_cb.reserve_back(CORNERS);
    for (uint32_t row = 0; row < 2; ++row) {
        tile_regs_acquire();
        copy_init(geom_scratch_cb_index);
        copy_tile(geom_scratch_cb_index, row, 0);
        copy_tile(geom_scratch_cb_index, 2, 1);
        copy_tile(geom_scratch_cb_index, 3, 2);
        mul_binary_tile_init();
        geom_mul(0, 1, 3);
        geom_mul(0, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(weight_cb_index);
        pack_tile(3, weight_cb_index);
        pack_tile(0, weight_cb_index);
        tile_regs_release();
    }
    scratch_cb.pop_front(4);
    weight_cb.push_back(CORNERS);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(corner_block_cb_index, scalar_cb_index, column_cb_index);
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    CircularBuffer tiled_cb(tiled_cb_index);
    CircularBuffer column_cb(column_cb_index);
    CircularBuffer conv_weight_cb(conv_weight_cb_index);
    CircularBuffer acc_cb(acc_cb_index);
    // Written once by the reader and kept for every block.
    CircularBuffer(geom_kernel_cb_index).wait_front(2);
    // Every block's output tiles stay reserved while the positions accumulate into them; the last
    // pass hands each block's over as it finishes.
    acc_cb.reserve_back(num_blocks * OUT_TILES);

    auto geometry_of = [&](uint32_t block) {
        const bool odd = block & 1;
        compute_geometry(
            odd ? odd_geom_in_cb_index : geom_in_cb_index,
            odd ? odd_geom_index_cb_index : geom_index_cb_index,
            odd ? odd_geom_weight_cb_index : geom_weight_cb_index);
    };
    for (uint32_t block = 0; block < num_blocks && block < 2; ++block) {
        geometry_of(block);
    }
    if constexpr (HAS_BIAS) {
        CircularBuffer(bias_cb_index).wait_front(OUT_TILES);
    }

    // Bias, ReLU, bfloat16: a block's output, at the front of acc_cb, to the reader that owns the
    // block. It runs as each block finishes its last pass, so the readers write it out while the
    // other blocks still compute.
    auto finish_block = [&](uint32_t block) {
        const uint32_t out_index = (block & 1) ? odd_output_cb_index : output_cb_index;
        CircularBuffer output_cb(out_index);
        acc_cb.wait_front(OUT_TILES);
        if constexpr (HAS_BIAS) {
            reconfig_data_format(acc_cb_index, bias_cb_index);
        } else {
            reconfig_data_format_srca(acc_cb_index);
        }
        output_cb.reserve_back(OUT_TILES);
        pack_reconfig_data_format(out_index);
        for (uint32_t n0 = 0; n0 < OUT_TILES; n0 += DEST_TILES) {
            const uint32_t chunk = OUT_TILES - n0 < DEST_TILES ? OUT_TILES - n0 : DEST_TILES;
            if constexpr (HAS_BIAS) {
                add_bcast_rows_init(acc_cb_index, bias_cb_index);
            } else {
                copy_init(acc_cb_index);
            }
            tile_regs_acquire();
            for (uint32_t i = 0; i < chunk; ++i) {
                if constexpr (HAS_BIAS) {
                    add_tiles_bcast_rows(acc_cb_index, bias_cb_index, n0 + i, n0 + i, i);
                } else {
                    copy_tile(acc_cb_index, n0 + i, i);
                }
            }
            if constexpr (RELU) {
                relu_tile_init();
                for (uint32_t i = 0; i < chunk; ++i) {
                    relu_tile(i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < chunk; ++i) {
                pack_tile<true>(i, out_index, n0 + i);
            }
            tile_regs_release();
        }
        output_cb.push_back(OUT_TILES);
        acc_cb.pop_front(OUT_TILES);
    };

    for (uint32_t pass = 0; pass < kernel_positions * CHUNKS; ++pass) {
        const bool last_pass = pass + 1 == kernel_positions * CHUNKS;
        conv_weight_cb.wait_front(WEIGHT_SLICE_TILES);
        for (uint32_t block = 0; block < num_blocks; ++block) {
            const bool odd = block & 1;
            const uint32_t corner_cb_index = odd ? odd_corner_block_cb_index : corner_block_cb_index;
            const uint32_t scalar_index = odd ? odd_scalar_cb_index : scalar_cb_index;
            CircularBuffer corner_block_cb(corner_cb_index);
            CircularBuffer scalar_cb(scalar_index);

            // The block's columns of this pass: the four corners times their scalars, summed.
            reconfig_data_format_srca(corner_cb_index);
            pack_reconfig_data_format(tiled_cb_index);
            pack_reconfig_l1_acc(0);
            fast_tilize_init(corner_cb_index, BLOCK_TILES, tiled_cb_index);
            corner_block_cb.wait_front(BLOCK_TILES);
            tiled_cb.reserve_back(BLOCK_TILES);
            fast_tilize_block(corner_cb_index, BLOCK_TILES, tiled_cb_index);
            tiled_cb.push_back(BLOCK_TILES);
            corner_block_cb.pop_front(BLOCK_TILES);
            fast_tilize_uninit(corner_cb_index, tiled_cb_index, BLOCK_TILES);

            // The four corners' products sum in dest: ELWMUL adds into dest, which every acquire
            // starts at zero, so each column tile is packed once.
            bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(tiled_cb_index, scalar_index);
            tiled_cb.wait_front(BLOCK_TILES);
            scalar_cb.wait_front(CORNERS);
            column_cb.reserve_back(channel_tiles);
            for (uint32_t t0 = 0; t0 < channel_tiles; t0 += DEST_TILES) {
                const uint32_t tiles = channel_tiles - t0 < DEST_TILES ? channel_tiles - t0 : DEST_TILES;
                tile_regs_acquire();
                for (uint32_t corner = 0; corner < CORNERS; ++corner) {
                    for (uint32_t t = 0; t < tiles; ++t) {
                        mul_tiles_bcast<BroadcastType::COL>(
                            tiled_cb_index, scalar_index, corner * channel_tiles + t0 + t, corner, t);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < tiles; ++t) {
                    pack_tile(t, column_cb_index);
                }
                tile_regs_release();
            }
            column_cb.push_back(channel_tiles);
            scalar_cb.pop_front(CORNERS);
            tiled_cb.pop_front(BLOCK_TILES);

            // The block's output += its columns times the pass's weight slice: fp32 in dest over the
            // slice's rows, and in L1 across passes; the first pass overwrites. In the last pass
            // the blocks before this one have left acc_cb, so this one's tiles are at its front.
            column_cb.wait_front(channel_tiles);
            reconfig_data_format(conv_weight_cb_index, column_cb_index);
            pack_reconfig_data_format(acc_cb_index);
            for (uint32_t n0 = 0; n0 < OUT_TILES; n0 += DEST_TILES) {
                const uint32_t chunk = OUT_TILES - n0 < DEST_TILES ? OUT_TILES - n0 : DEST_TILES;
                matmul_block_init(column_cb_index, conv_weight_cb_index, 0, chunk, 1, channel_tiles);
                tile_regs_acquire();
                for (uint32_t k = 0; k < channel_tiles; ++k) {
                    matmul_block(
                        column_cb_index, conv_weight_cb_index, k, k * OUT_TILES + n0, 0, 0, chunk, 1, channel_tiles);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_l1_acc(pass == 0 ? 0 : 1);
                for (uint32_t i = 0; i < chunk; ++i) {
                    pack_tile<true>(i, acc_cb_index, (last_pass ? 0 : block * OUT_TILES) + n0 + i);
                }
                tile_regs_release();
            }
            pack_reconfig_l1_acc(0);
            column_cb.pop_front(channel_tiles);
            if (last_pass) {
                acc_cb.push_back(OUT_TILES);
                finish_block(block);
            }
            if (pass == 0 && block + 2 < num_blocks) {
                geometry_of(block + 2);
            }
        }
        conv_weight_cb.pop_front(WEIGHT_SLICE_TILES);
    }
}
