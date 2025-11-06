// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/transpose_wh_dest.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/cumsum.h"

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
// #include "ttnn/operations/experimental/reduction/integral_image/device/kernels/common.hpp"

#define APPROX false
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "debug/dprint.h"
#include "debug/dprint_tensix.h"

#include <cmath>

namespace {

constexpr uint32_t ONE_TILE{1};
constexpr uint32_t FIRST_TILE{0};
constexpr uint32_t WORKING_REG{0};

class ReadCBGuard {
    uint32_t cb;
    uint32_t tiles;

public:
    ReadCBGuard(uint32_t cb, uint32_t tiles) : cb{cb}, tiles{tiles} { cb_wait_front(cb, tiles); }
    ~ReadCBGuard() { cb_pop_front(cb, tiles); }

    ReadCBGuard(const ReadCBGuard&) = delete;  // can not allow to touch the object anyhow
    ReadCBGuard(ReadCBGuard&&) = delete;
    ReadCBGuard& operator=(const ReadCBGuard&) = delete;
    ReadCBGuard& operator=(ReadCBGuard&&) = delete;
    template <typename any_type_t>
    operator any_type_t() = delete;
};

class WriteCBGuard {
    uint32_t cb;
    uint32_t tiles;

public:
    WriteCBGuard(uint32_t cb, uint32_t tiles) : cb{cb}, tiles{tiles} { cb_reserve_back(cb, tiles); }
    ~WriteCBGuard() { cb_push_back(cb, tiles); }

    WriteCBGuard(const WriteCBGuard&) = delete;  // can not allow to touch the object anyhow
    WriteCBGuard(WriteCBGuard&&) = delete;
    WriteCBGuard& operator=(const WriteCBGuard&) = delete;
    WriteCBGuard& operator=(WriteCBGuard&&) = delete;
    template <typename any_type_t>
    operator any_type_t() = delete;
};

// FORCE_INLINE uint32_t portable_ilogb(uint32_t x) {
//     using std::frexp;

//     int exp = 0;
//     // frexp: x = m * 2^(exp), with 0.5 <= |m| < 1 (if x!=0)
//     // ilogb (base 2) = exp - 1
//     (void)frexp(x > uint32_t(0) ? x : -x, &exp);
//     return exp - 1;
// }

FORCE_INLINE uint32_t get_coord_from_tile_xy(uint32_t read_i, uint32_t write_i) {
    return ((write_i & 0x10) << 5)    // y_hi * 512
           | ((read_i & 0x10) << 4)   // x_hi * 256
           | ((write_i & 0x0F) << 4)  // y_lo * 16
           | (read_i & 0x0F);
}

// const uint32_t read_tile_id = get_tile_id(num_blocks_in_row, num_blocks_in_column, num_slices_along_channels,
// inner_tile_stride, channels_slice_i, row_chunk_i, column_block_i);
FORCE_INLINE uint32_t get_tile_id(
    uint32_t depth_blocks_num,
    uint32_t height_blocks_num,
    uint32_t channels_blocks_num,
    uint32_t inner_tile_stride,
    uint32_t channels_slice_i,
    uint32_t row_block_i,
    uint32_t column_block_i,
    uint32_t block_depth = 32) {
    const uint32_t tensor_face_block_size = channels_blocks_num * height_blocks_num;
    const uint32_t block_first_tile_id =
        block_depth * (tensor_face_block_size * column_block_i + depth_blocks_num * row_block_i + channels_slice_i);
    const uint32_t tile_id = block_first_tile_id + block_depth * tensor_face_block_size;
    return tile_id;
}

FORCE_INLINE constexpr uint32_t block_depth_ceil(uint32_t value, uint32_t block_depth = 32) {
    return (value + block_depth - 1) / block_depth;
}

// total tiles with double buffering (apart from 32t CBs, not enough memory) and default 32 block_depth: 204, each tile:
// 4 KB, total: 816 KB for 4-byte types per core.
struct IntImgComputeCTAs {
    const uint32_t start_cb;            // 2 tiles
    const uint32_t input_cb;            // 2 tiles
    const uint32_t acc_cb;              // 2 tiles
    const uint32_t cumsum_stage_0_cb;   // `block_size` tiles
    const uint32_t cumsum_stage_1_cb;   // `block_size` tiles
    const uint32_t cumsum_stage_2_cb;   // `block_size` tiles
    const uint32_t cumsum_stage_3_cb;   // `block_size` tiles
    const uint32_t output_cb;           // 2 tiles
    const uint32_t axis_2_buffer_cb;    // 2 tiles: covers entire propagation
    const uint32_t axis_3_buffer_0_cb;  // `block_size` tiles: each tile is spawned from broadcasting the last row of
                                        // upper block across all rows of a given tile - for the time being, their
                                        // spawning is forced to be done in the writer kernel.
    const uint32_t axis_3_buffer_1_cb;  // dual channel communication with the writer kernel is comprehensive and
                                        // properly synchronizes writer and compute kernels.
    const uint32_t tile_height;
    const uint32_t tile_width;
    const uint32_t block_depth;   // usually 32
    const uint32_t num_channels;  // axis 4/4
    const uint32_t input_height;  // axis 3/4
    const uint32_t input_depth;   // axis 2/4
    const uint32_t num_batches;   // axis 1/4
};

FORCE_INLINE constexpr IntImgComputeCTAs get_ctas() {
    return {
        get_compile_time_arg_val(0),
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16),
        get_compile_time_arg_val(17),
    };
}

FORCE_INLINE void cumsum_cube_axis_2(
    uint32_t cb_start,
    uint32_t cb_acc,
    uint32_t cb_input,
    uint32_t cb_cumsum_stage_0,
    uint32_t cb_axis_2_buffer,
    bool save_last_tile,
    uint32_t block_depth = 32) {
    ReadCBGuard start_cb_read_guard{cb_start, ONE_TILE};  // consume cb_start 1t

    bool enable_reload = false;

    for (uint32_t tile_i = 0; tile_i < block_depth; ++tile_i) {
        WriteCBGuard cumsum_stage_cb_write_guard{cb_cumsum_stage_0, ONE_TILE};  // produce cb_cumsum_stage_0 1-32t
        tile_regs_acquire();
        const uint32_t cb_op = enable_reload ? cb_acc : cb_start;
        cb_wait_front(cb_input, ONE_TILE);

        // copy_tile_init(cb_op);
        // copy_tile(cb_op, FIRST_TILE, WORKING_REG + 1);

        add_tiles_init(cb_input, cb_op);
        add_tiles(cb_input, cb_op, FIRST_TILE, FIRST_TILE, WORKING_REG);

        cb_pop_front(cb_input, ONE_TILE);
        if (enable_reload) {
            cb_pop_front(cb_acc, ONE_TILE);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_acc, ONE_TILE);
        pack_reconfig_data_format(cb_acc);
        pack_tile(WORKING_REG, cb_acc);
        cb_push_back(cb_acc, ONE_TILE);

        tile_regs_release();

        tile_regs_acquire();

        cb_wait_front(cb_acc, ONE_TILE);
        copy_tile_init(cb_acc);
        copy_tile(cb_acc, FIRST_TILE, WORKING_REG);

        tile_regs_commit();
        tile_regs_wait();

        pack_reconfig_data_format(cb_cumsum_stage_0);
        pack_tile(WORKING_REG, cb_cumsum_stage_0, FIRST_TILE);

        if ((tile_i == block_depth - 1) && save_last_tile) {
            WriteCBGuard axis_2_buffer_cb_guard{cb_axis_2_buffer, ONE_TILE};
            pack_reconfig_data_format(cb_axis_2_buffer);
            pack_tile(WORKING_REG, cb_axis_2_buffer, FIRST_TILE);
        }

        tile_regs_release();

        enable_reload = true;
    }

    cb_pop_front(cb_acc, ONE_TILE);
}

FORCE_INLINE void cumsum_cube_axis_3(
    uint32_t cb_cumsum_stage_wip, uint32_t cb_cumsum_output, uint32_t block_depth = 32) {
    for (uint32_t tile_i = 0; tile_i < block_depth; ++tile_i) {
        ReadCBGuard read_cumsum_guard{cb_cumsum_stage_wip, ONE_TILE};
        WriteCBGuard cumsum_output_write_guard{cb_cumsum_output, ONE_TILE};
        tile_regs_acquire();
        copy_tile_init(cb_cumsum_stage_wip);
        copy_tile(cb_cumsum_stage_wip, FIRST_TILE, WORKING_REG);

        cumsum_tile_init();
        cumsum_tile(WORKING_REG);

        tile_regs_commit();
        tile_regs_wait();

        pack_reconfig_data_format(cb_cumsum_output);
        pack_tile(WORKING_REG, cb_cumsum_output, FIRST_TILE);

        tile_regs_release();
    }
}

// TODO(jbbieniek): finish
FORCE_INLINE void propagate_tile_into_cube(
    uint32_t cb_axis_2_buffer,
    uint32_t cb_cumsum_stage_a,
    uint32_t cb_cumsum_stage_b,
    bool save_last_tile,
    uint32_t block_depth = 32) {
    cb_wait_front(cb_axis_2_buffer, ONE_TILE);
    for (uint32_t tile_i = 0; tile_i < block_depth; ++tile_i) {
        ReadCBGuard cb_cumsum_stage_0_guard{cb_cumsum_stage_a, ONE_TILE};
        WriteCBGuard cb_cumsum_stage_1_guard{cb_cumsum_stage_b, ONE_TILE};
        tile_regs_acquire();

        // binary_op_init_common(cb_axis_2_buffer, cb_cumsum_stage_a, cb_cumsum_stage_b);
        add_tiles_init(cb_axis_2_buffer, cb_cumsum_stage_a);
        add_tiles(cb_axis_2_buffer, cb_cumsum_stage_a, FIRST_TILE, FIRST_TILE, WORKING_REG);

        tile_regs_commit();
        tile_regs_wait();

        pack_reconfig_data_format(cb_cumsum_stage_b);
        pack_tile(WORKING_REG, cb_cumsum_stage_b, FIRST_TILE);
        if ((tile_i == block_depth - 1)) {  // when last tile in block gets propagated on, finally release the axis 2
                                            // buffer CB and save last tile only when there's a specific request. it
                                            // must be released to get ready for the processing of the next row chunk.
            cb_pop_front(cb_axis_2_buffer, ONE_TILE);
            if (save_last_tile) {
                WriteCBGuard axis_2_buffer_cb_guard{cb_axis_2_buffer, ONE_TILE};
                pack_reconfig_data_format(cb_axis_2_buffer);
                pack_tile(WORKING_REG, cb_axis_2_buffer, FIRST_TILE);
            }
        }

        tile_regs_release();
    }
}

FORCE_INLINE void get_and_propagate_adder_cube(
    uint32_t cb_cumsum_stage_X, uint32_t cb_axis_3_buffer_read, uint32_t cb_output, uint32_t block_depth = 32) {
    // there is the necessity to receive a block

    for (uint32_t tile_i = 0; tile_i < block_depth; ++tile_i) {
        ReadCBGuard cb_cumsum_stage_X_read_guard{cb_cumsum_stage_X, ONE_TILE};
        ReadCBGuard cb_axis_3_buffer_read_guard{cb_axis_3_buffer_read, ONE_TILE};
        WriteCBGuard cb_output_write_guard{cb_output, ONE_TILE};
        tile_regs_acquire();

        binary_op_init_common(cb_cumsum_stage_X, cb_axis_3_buffer_read, cb_output);
        UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(cb_cumsum_stage_X, cb_axis_3_buffer_read)));
        UNPACK(
            (llk_unpack_AB<BroadcastType::ROW>(cb_cumsum_stage_X, cb_axis_3_buffer_read, FIRST_TILE, FIRST_TILE, 31)));

        MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::ROW>()));
        MATH((llk_math_eltwise_binary<
              EltwiseBinaryType::ELWADD,
              BroadcastType::ROW,
              false,  // if anything, investigate later
              0,
              EltwiseBinaryReuseDestType::NONE>(WORKING_REG, false)));

        // add_tiles_init(cb_cumsum_stage_X, cb_output);
        // add_tiles(cb_cumsum_stage_X, cb_axis_3_buffer_read, FIRST_TILE, FIRST_TILE, WORKING_REG);

        tile_regs_wait();
        tile_regs_commit();

        pack_reconfig_data_format(cb_output);
        pack_tile(WORKING_REG, cb_output, FIRST_TILE);

        tile_regs_release();
    }
}

template <typename ctas_t>
FORCE_INLINE void perform_intimg_along_row_chunk(
    const ctas_t& ctas, uint32_t num_blocks_in_row, uint32_t rows_block_i) {
    for (uint32_t column_block_i = 0; column_block_i < num_blocks_in_row; ++column_block_i) {  // go along a row
        // consume: cb_start 1t, cb_acc 1t, cb_input 1t x32
        // produce: cb_cumsum_stage_0 32t
        const uint32_t block_depth = std::min(ctas.input_depth - column_block_i * ctas.block_depth, ctas.block_depth);
        const bool save_last_tile_after_cumsum_cube_axis_2 = (column_block_i == 0) && (num_blocks_in_row > 1);
        const bool save_last_tile_after_tile_into_cube_propagation =
            (column_block_i > 0) && (column_block_i != (num_blocks_in_row - 1));
        cumsum_cube_axis_2(
            ctas.start_cb,
            ctas.acc_cb,
            ctas.input_cb,
            ctas.cumsum_stage_0_cb,
            ctas.axis_2_buffer_cb,
            save_last_tile_after_cumsum_cube_axis_2,
            // (column_block_i == 0) && (num_blocks_in_row > 1),
            block_depth);
        if (column_block_i > 0) {
            // axis 2/4's propagation...
            // consume: cb_axis_2_buffer 1t, cb_cumsum_stage_0 32t
            // produce: cb_cumsum_stage_1 32t
            propagate_tile_into_cube(
                ctas.axis_2_buffer_cb,
                ctas.cumsum_stage_0_cb,
                ctas.cumsum_stage_1_cb,
                save_last_tile_after_tile_into_cube_propagation,
                // (column_block_i != (num_blocks_in_row - 1)),
                block_depth);  // working with cb_axis_2_buffer and cb_cumsum
            if (rows_block_i > 0) {
                // axis 3/4's propagation...
                // consume: cb_cumsum_stage_1 32t
                // produce: cb_cumsum_stage_2 32t
                cumsum_cube_axis_3(ctas.cumsum_stage_1_cb, ctas.cumsum_stage_2_cb, block_depth);
                // consume: cb_cumsum_stage_2 32tm axis_3_buffer_1_cb 32t
                // produce: cb_output 1t x32
                get_and_propagate_adder_cube(
                    ctas.cumsum_stage_2_cb, ctas.axis_3_buffer_0_cb, ctas.output_cb, block_depth);
            } else {
                // consume: cb_cumsum_stage_1 32t
                // produce: cb_output 1t x32
                cumsum_cube_axis_3(ctas.cumsum_stage_1_cb, ctas.output_cb, block_depth);
            }
        } else {
            // !!!! DIFFERENT PARAMS THAN THE CONDITION ABOVE !!!! (!!!!!!!!one stage less!!!!!!!!!)
            if (rows_block_i > 0) {
                // axis 3/4's propagation...
                // consume: cb_cumsum_stage_0 32t
                // produce: cumsummed down cube (cb_cumsum_stage_1 32t)
                cumsum_cube_axis_3(ctas.cumsum_stage_0_cb, ctas.cumsum_stage_1_cb, block_depth);
                // consume: cb_cumsum_stage_1 32t, axis_3_buffer_1_cb 32t
                // produce: cb_output 1t x32
                get_and_propagate_adder_cube(
                    ctas.cumsum_stage_1_cb, ctas.axis_3_buffer_0_cb, ctas.output_cb, block_depth);
            } else {
                // consume: cb_cumsum_stage_0 32t
                // produce: cumsummed down cube (cb_output 1t x32)
                cumsum_cube_axis_3(ctas.cumsum_stage_0_cb, ctas.output_cb, block_depth);
            }
        }
    }
}

}  // namespace

namespace NAMESPACE {

void MAIN {
    constexpr auto ctas{get_ctas()};

    constexpr uint32_t num_slices_along_channels = block_depth_ceil(
        ctas.num_channels, ctas.block_depth);  // block_depth is expected to be a power of 2 (the default is the regular
                                               // 32x32 tile's width/height size)
    constexpr uint32_t num_blocks_in_row = block_depth_ceil(ctas.input_depth, ctas.block_depth);
    constexpr uint32_t num_blocks_in_column = block_depth_ceil(ctas.input_height, ctas.block_depth);

    for (uint32_t channels_block_i = 0; channels_block_i < num_slices_along_channels; ++channels_block_i) {
        for (uint32_t rows_block_i = 0; rows_block_i < num_blocks_in_column; ++rows_block_i) {
            perform_intimg_along_row_chunk(ctas, num_blocks_in_row, rows_block_i);
        }
    }
}

}  // namespace NAMESPACE
