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

#define APPROX false
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

namespace {

constexpr uint32_t ONE_TILE{1};
constexpr uint32_t FIRST_TILE{0};
constexpr uint32_t WORKING_REG{0};

/**
 * @brief RAII guard for safely reading from a Circular Buffer (CB).
 *
 * `ReadCBGuard` automatically manages CB read synchronization in a scoped manner.
 * When constructed, it waits for `tiles` tiles to be available at the CB front
 * (`cb_wait_front`), ensuring data readiness before access. Upon destruction,
 * it automatically pops those tiles from the CB front (`cb_pop_front`),
 * signaling that the tiles have been consumed.
 *
 * This guarantees balanced `cb_wait_front`/`cb_pop_front` calls, even in the
 * presence of early returns or exceptions, preventing CB underflows and race
 * conditions.
 *
 * @note This class is strictly non-copyable and non-movable to prevent any
 *       double-pop or premature release of CB resources.
 *
 * Example usage:
 * @code
 * {
 *     ReadCBGuard guard(cb_id, num_tiles);
 *     // Safely read from CB: data is guaranteed ready.
 *     // ...
 * } // Automatically pops the tiles on scope exit.
 * @endcode
 */
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

/**
 * @brief RAII guard for safely writing to a Circular Buffer (CB).
 *
 * `WriteCBGuard` automatically manages CB write synchronization in a scoped manner.
 * When constructed, it reserves space for `tiles` tiles at the CB back
 * (`cb_reserve_back`), ensuring sufficient room for writing. Upon destruction,
 * it automatically pushes those tiles to the CB back (`cb_push_back`),
 * making them visible to downstream consumers.
 *
 * This ensures balanced `cb_reserve_back`/`cb_push_back` calls and prevents
 * buffer overflows or mismatched producer-consumer behavior.
 *
 * @note Like `ReadCBGuard`, this class is non-copyable and non-movable to ensure
 *       one-to-one ownership of the CB reservation.
 *
 * Example usage:
 * @code
 * {
 *     WriteCBGuard guard(cb_id, num_tiles);
 *     // Safely write to CB: space is guaranteed reserved.
 *     // ...
 * } // Automatically pushes tiles to the CB on scope exit.
 * @endcode
 */
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

FORCE_INLINE uint32_t get_coord_from_tile_xy(uint32_t read_i, uint32_t write_i) {
    return ((write_i & 0x10) << 5)    // y_hi * 512
           | ((read_i & 0x10) << 4)   // x_hi * 256
           | ((write_i & 0x0F) << 4)  // y_lo * 16
           | (read_i & 0x0F);
}

FORCE_INLINE constexpr uint32_t block_depth_ceil(uint32_t value, uint32_t block_depth = 32) {
    return (value + block_depth - 1) / block_depth;
}

struct IntImgComputeCTAs {
    const uint32_t start_cb;
    const uint32_t input_cb;
    const uint32_t acc_cb;
    const uint32_t cumsum_stage_0_cb;
    const uint32_t cumsum_stage_1_cb;
    const uint32_t cumsum_stage_2_cb;
    const uint32_t output_cb;
    const uint32_t axis_2_buffer_cb;    // covers entire propagation
    const uint32_t axis_3_buffer_0_cb;  // each tile is spawned from broadcasting the last row of
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
    const uint32_t cores_x;
    const uint32_t cores_y;
};

FORCE_INLINE constexpr IntImgComputeCTAs get_ctas() {
    return {
        get_compile_time_arg_val(0),  get_compile_time_arg_val(1),  get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),  get_compile_time_arg_val(4),  get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),  get_compile_time_arg_val(7),  get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),  get_compile_time_arg_val(10), get_compile_time_arg_val(11),
        get_compile_time_arg_val(12), get_compile_time_arg_val(13), get_compile_time_arg_val(14),
        get_compile_time_arg_val(15), get_compile_time_arg_val(16), get_compile_time_arg_val(17),
        get_compile_time_arg_val(18),
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
    ReadCBGuard start_cb_read_guard{cb_start, ONE_TILE};

    bool enable_reload = false;

    binary_op_init_common(cb_input, cb_acc, cb_cumsum_stage_0);
    for (uint32_t tile_i = 0; tile_i < block_depth; ++tile_i) {
        WriteCBGuard cumsum_stage_cb_write_guard{cb_cumsum_stage_0, ONE_TILE};
        tile_regs_acquire();
        const uint32_t cb_op = enable_reload ? cb_acc : cb_start;
        cb_wait_front(cb_input, ONE_TILE);

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

        add_tiles_init(cb_cumsum_stage_X, cb_output);
        add_tiles(cb_cumsum_stage_X, cb_axis_3_buffer_read, FIRST_TILE, FIRST_TILE, WORKING_REG);

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
            block_depth);
        if (column_block_i > 0) {
            // axis 2/4's propagation
            propagate_tile_into_cube(
                ctas.axis_2_buffer_cb,
                ctas.cumsum_stage_0_cb,
                ctas.cumsum_stage_1_cb,
                save_last_tile_after_tile_into_cube_propagation,
                block_depth);  // working with cb_axis_2_buffer
            if (rows_block_i > 0) {
                // axis 3/4's propagation
                cumsum_cube_axis_3(ctas.cumsum_stage_1_cb, ctas.cumsum_stage_2_cb, block_depth);
                get_and_propagate_adder_cube(
                    ctas.cumsum_stage_2_cb, ctas.axis_3_buffer_1_cb, ctas.output_cb, block_depth);
            } else {
                cumsum_cube_axis_3(ctas.cumsum_stage_1_cb, ctas.output_cb, block_depth);
            }
        } else {
            if (rows_block_i > 0) {
                // axis 3/4's propagation
                cumsum_cube_axis_3(ctas.cumsum_stage_0_cb, ctas.cumsum_stage_1_cb, block_depth);
                get_and_propagate_adder_cube(
                    ctas.cumsum_stage_1_cb, ctas.axis_3_buffer_1_cb, ctas.output_cb, block_depth);
            } else {
                cumsum_cube_axis_3(ctas.cumsum_stage_0_cb, ctas.output_cb, block_depth);
            }
        }
    }
}

}  // namespace

namespace NAMESPACE {

void MAIN {
    constexpr auto ctas{get_ctas()};

    constexpr uint32_t num_blocks_in_row = block_depth_ceil(ctas.input_depth, ctas.block_depth);
    constexpr uint32_t num_blocks_in_column = block_depth_ceil(ctas.input_height, ctas.block_depth);

    for (uint32_t rows_block_i = 0; rows_block_i < num_blocks_in_column; ++rows_block_i) {
        perform_intimg_along_row_chunk(ctas, num_blocks_in_row, rows_block_i);
    }
}

}  // namespace NAMESPACE
