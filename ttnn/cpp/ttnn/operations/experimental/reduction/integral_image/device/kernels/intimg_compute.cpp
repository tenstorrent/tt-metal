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

FORCE_INLINE uint32_t get_coord_from_tile_xy(uint32_t read_i, uint32_t write_i) {
    return ((write_i & 0x10) << 5)    // y_hi * 512
           | ((read_i & 0x10) << 4)   // x_hi * 256
           | ((write_i & 0x0F) << 4)  // y_lo * 16
           | (read_i & 0x0F);
}

// all static data
struct IntImgComputeCTAs {
    const uint32_t start_cb;
    const uint32_t input_cb;
    const uint32_t acc_cb;
    const uint32_t before_adder_propagation_stage_cb;
    const uint32_t output_cb;
    const uint32_t to_bot_tile_cb;
    const uint32_t from_top_tile_cb;
    const uint32_t axis_3_buffer_0_cb;
    const uint32_t axis_3_buffer_1_cb;  // dual channel communication with the writer kernel is comprehensive and
                                        // properly synchronizes writer and compute kernels.
    const uint32_t tile_height;
    const uint32_t tile_width;
    const uint32_t num_batches;
    const uint32_t input_depth;
    const uint32_t input_height;
    const uint32_t input_width;
    const uint32_t num_tiles_along_channels;
    const uint32_t num_tiles_along_height;
    const uint32_t top_semaphore_id;
    const uint32_t bot_semaphore_id;
};

// all per-core runtime data
struct IntImgComputeRTAs {
    const uint32_t input_base_addr;
    // const uint32_t zero_tile_base_addr;
    const uint32_t output_base_addr;
    const uint32_t starting_tile_along_channels;
    const uint32_t num_tiles_along_channels_per_core;
    const uint32_t starting_tile_along_height;
    const uint32_t num_tiles_along_height_per_core;
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

FORCE_INLINE const IntImgComputeRTAs get_rtas() {
    return {
        get_arg_val<uint32_t>(0),
        get_arg_val<uint32_t>(1),
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(5),
        // get_arg_val<uint32_t>(6),
    };
}

FORCE_INLINE void cumsum_tile_axis_2_and_3(
    uint32_t cb_start, uint32_t cb_acc, uint32_t cb_input, uint32_t cb_next_stage) {
    ReadCBGuard start_cb_read_guard{cb_start, ONE_TILE};

    bool enable_reload = false;

    // for (uint32_t tile_i = 0; tile_i < row_depth; ++tile_i) {
    WriteCBGuard next_stage_cb_write_guard{cb_next_stage, ONE_TILE};
    tile_regs_acquire();
    const uint32_t cb_op = enable_reload ? cb_acc : cb_start;
    cb_wait_front(cb_input, ONE_TILE);

    add_tiles_init(cb_input, cb_op);
    add_tiles(cb_input, cb_op, FIRST_TILE, FIRST_TILE, WORKING_REG);

    cumsum_tile_init();
    cumsum_tile(WORKING_REG);

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

        pack_reconfig_data_format(cb_next_stage);
        pack_tile(WORKING_REG, cb_next_stage, FIRST_TILE);

        // if ((tile_i == row_depth - 1) && save_last_tile) {
        //     WriteCBGuard axis_2_buffer_cb_guard{cb_axis_2_buffer, ONE_TILE};
        //     pack_reconfig_data_format(cb_axis_2_buffer);
        //     pack_tile(WORKING_REG, cb_axis_2_buffer, FIRST_TILE);
        // }

        tile_regs_release();

        enable_reload = true;
        // }

        cb_pop_front(cb_acc, ONE_TILE);
}

FORCE_INLINE void cumsum_tile_axis_2(
    uint32_t cb_start, uint32_t cb_acc, uint32_t cb_input, uint32_t cb_cumsum_stage_0) {
    ReadCBGuard start_cb_read_guard{cb_start, ONE_TILE};  // consume cb_start

    bool enable_reload = false;

    for (uint32_t tile_i = 0; tile_i < block_depth; ++tile_i) {
        WriteCBGuard cumsum_stage_cb_write_guard{cb_cumsum_stage_0, ONE_TILE};  // produce cb_cumsum_stage_0
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

FORCE_INLINE void cumsum_tile_axis_3(uint32_t cb_cumsum_stage_wip, uint32_t cb_cumsum_output) {
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

FORCE_INLINE void get_and_propagate_adder_tile_from_reader(
    uint32_t cb_before_adder_propagation_tile,
    uint32_t cb_axis_3_buffer_read,
    uint32_t cb_to_bot_stage_tile,
    uint32_t cb_output) {
    ReadCBGuard cb_before_adder_propagation_tile_guard{
        cb_before_adder_propagation_tile, ONE_TILE};                           // tile from above to propagate onto
    ReadCBGuard cb_axis_3_buffer_read_guard{cb_axis_3_buffer_read, ONE_TILE};  // propagation tile
    WriteCBGuard cb_to_bot_stage_tile_guard{cb_to_bot_stage_tile, ONE_TILE};   // tile that gets propagated lower
    WriteCBGuard cb_output_guard{
        cb_output,
        ONE_TILE};  // same tile as above, but the one that goes straight to the output (compute -> writer -> DRAM)
    tile_regs_acquire();

    add_tiles_init(cb_before_adder_propagation_tile, cb_axis_3_buffer_read);
    add_tiles(cb_before_adder_propagation_tile, cb_axis_3_buffer_read, FIRST_TILE, FIRST_TILE, WORKING_REG);

    tile_regs_wait();
    tile_regs_commit();

    pack_reconfig_data_format(cb_to_bot_stage_tile);
    pack_tile(WORKING_REG, cb_to_bot_stage_tile, FIRST_TILE);
    pack_reconfig_data_format(cb_output);
    pack_tile(WORKING_REG, cb_output, FIRST_TILE);

    tile_regs_release();
}

template <typename ctas_t, typename rtas_t>
FORCE_INLINE void perform_intimg_along_assigned_row_chunks_by_height(
    const ctas_t& ctas, const rtas_t& rtas, uint32_t ending_tile_along_height) {
    // REMEMBER TO PUSH THE TILE ALSO TO THE CB THAT SERVES AS A STAGE BETWEEN CORES, NOT ONLY TO THE
    // OUTPUT!!!!!!!!!!!!!!
    for (uint32_t column_tile_i = rtas.starting_tile_along_height; column_tile_i < ending_tile_along_height;
         ++column_tile_i) {  // TODO(jbbieniekTT): this algorithm works correctly when `ending_tile_along_height -
                             // starting_tile_along_height == 1`, which is the case for input shapes up to [1, any, 160,
                             // any]
        bool propagation_enabled = (column_tile_i != ctas.num_tiles_along_height - 1);
        uint32_t cb_next_stage = propagation_enabled ? ctas.before_adder_propagation_stage_cb : ctas.output_cb;
        for (uint32_t row_tile_i = 0; row_tile_i < ctas.input_depth; ++row_tile_i) {
            cumsum_tile_axis_2_and_3(ctas.start_cb, ctas.acc_cb, ctas.input_cb, cb_next_stage);
            // if (column_tile_i != ctas.num_tiles_along_height - 1) { // propagate if currently processed row chunk is
            // not located at the "bottom" of the tensor, that is, last row chunk when seen along tile-height
            //     // axis 3/4's propagation...

            //     // WAIT FOR THE READER AND PERFORM LOCAL CUMSUMS
            //     cumsum_tile_axis_2_and_3(
            //         ctas.start_cb, ctas.acc_cb, ctas.input_cb, ctas.before_adder_propagation_stage_cb,
            //         ctas.input_depth);
            //     // APPLY THE ADDER TILE SPAWNED BY THE READER, THEN PA
            //     get_and_propagate_adder_tile_from_reader(
            //         ctas.before_adder_propagation_stage_cb, ctas.axis_3_buffer_1_cb, ctas.to_bot_stage_tile,
            //         ctas.output_cb);
            // } else {
            //     cumsum_tile_axis_2_and_3(ctas.start_cb, ctas.acc_cb, ctas.input_cb, ctas.output_cb,
            //     ctas.input_depth);
            // }
            if (propagation_enabled) {
                get_and_propagate_adder_tile_from_reader(
                    ctas.before_adder_propagation_stage_cb,
                    ctas.axis_3_buffer_1_cb,
                    ctas.to_bot_tile_cb,
                    ctas.output_cb);
            }
        }
    }
}

}  // namespace

namespace NAMESPACE {

void MAIN {
    const auto rtas = get_rtas();
    constexpr auto ctas = get_ctas();

    const uint32_t ending_tile_along_channels =
        rtas.starting_tile_along_channels + rtas.num_tiles_along_channels_per_core;
    const uint32_t ending_tile_along_height = rtas.starting_tile_along_height + rtas.num_tiles_along_height_per_core;
    for (uint32_t channels_slice_i = rtas.starting_tile_along_channels; channels_slice_i < ending_tile_along_channels;
         ++channels_slice_i) {
        // for (uint32_t column_tile_i = rtas.starting_tile_along_height; column_tile_i < ending_tile_along_height;
        //      ++column_tile_i) {
        perform_intimg_along_assigned_row_chunks_by_height(ctas, rtas, ending_tile_along_height);
        // }
    }
}

}  // namespace NAMESPACE
