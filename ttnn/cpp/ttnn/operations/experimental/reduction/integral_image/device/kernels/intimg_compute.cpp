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
constexpr uint32_t CB_INVALID{static_cast<uint32_t>(-1)};

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
    const uint32_t cumsum_axis_3;
    const uint32_t before_adder_propagation_cb;
    const uint32_t output_cb;
    const uint32_t to_bot_core_cb;
    // const uint32_t from_top_core_cb;
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

// FORCE_INLINE void cumsum_tile_axis_2_and_3(
//     uint32_t cb_start, uint32_t cb_acc, uint32_t cb_input, uint32_t cb_next_stage) {
//     ReadCBGuard start_cb_read_guard{cb_start, ONE_TILE};

//     bool enable_reload = false;

//     // for (uint32_t tile_i = 0; tile_i < row_depth; ++tile_i) {
//     WriteCBGuard next_stage_cb_write_guard{cb_next_stage, ONE_TILE};
//     tile_regs_acquire();
//     const uint32_t cb_op = enable_reload ? cb_acc : cb_start;
//     cb_wait_front(cb_input, ONE_TILE);

//     add_tiles_init(cb_input, cb_op);
//     add_tiles(cb_input, cb_op, FIRST_TILE, FIRST_TILE, WORKING_REG);

//     cumsum_tile_init();
//     cumsum_tile(WORKING_REG);

//     cb_pop_front(cb_input, ONE_TILE);
//     if (enable_reload) {
//         cb_pop_front(cb_acc, ONE_TILE);
//     }

//         tile_regs_commit();
//         tile_regs_wait();

//         cb_reserve_back(cb_acc, ONE_TILE);
//         pack_reconfig_data_format(cb_acc);
//         pack_tile(WORKING_REG, cb_acc);
//         cb_push_back(cb_acc, ONE_TILE);

//         tile_regs_release();

//         tile_regs_acquire();

//         cb_wait_front(cb_acc, ONE_TILE);
//         copy_tile_init(cb_acc);
//         copy_tile(cb_acc, FIRST_TILE, WORKING_REG);

//         tile_regs_commit();
//         tile_regs_wait();

//         pack_reconfig_data_format(cb_next_stage);
//         pack_tile(WORKING_REG, cb_next_stage, FIRST_TILE);

//         // if ((tile_i == row_depth - 1) && save_last_tile) {
//         //     WriteCBGuard axis_2_buffer_cb_guard{cb_axis_2_buffer, ONE_TILE};
//         //     pack_reconfig_data_format(cb_axis_2_buffer);
//         //     pack_tile(WORKING_REG, cb_axis_2_buffer, FIRST_TILE);
//         // }

//         tile_regs_release();

//         enable_reload = true;
//         // }

//         cb_pop_front(cb_acc, ONE_TILE);
// }

FORCE_INLINE void cumsum_tile_axis_2(
    uint32_t cb_start, uint32_t cb_acc, uint32_t cb_input, uint32_t cb_pass_to_cumsum_3, uint32_t row_depth) {
    // uint32_t cb_start, uint32_t cb_acc, uint32_t cb_input, uint32_t cb_pass_to_cumsum_3, bool
    // should_pass_to_propagation, uint32_t cb_pass_to_propagation) {
    ReadCBGuard start_cb_read_guard{cb_start, ONE_TILE};  // consume cb_start

    bool enable_reload = false;

    for (uint32_t tile_i = 0; tile_i < row_depth; ++tile_i) {
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

        cumsum_tile_init();
        cumsum_tile(WORKING_REG);

        tile_regs_commit();
        tile_regs_wait();

        {
            WriteCBGuard cb_pass_to_cumsum_3_guard{cb_pass_to_cumsum_3, ONE_TILE};  // produce cb_cumsum_stage_0
            pack_reconfig_data_format(cb_pass_to_cumsum_3);
            pack_tile(WORKING_REG, cb_pass_to_cumsum_3, FIRST_TILE);
        }

        // if ((tile_i == block_depth - 1) && save_last_tile) {
        //     WriteCBGuard axis_2_buffer_cb_guard{cb_axis_2_buffer, ONE_TILE};
        //     pack_reconfig_data_format(cb_axis_2_buffer);
        //     pack_tile(WORKING_REG, cb_axis_2_buffer, FIRST_TILE);
        // }

        // if (should_pass_to_propagation) {
        //     WriteCBGuard cb_pass_to_propagation_guard{cb_pass_to_propagation, ONE_TILE};
        //     pack_reconfig_data_format(cb_pass_to_propagation);
        //     pack_tile(WORKING_REG, cb_pass_to_propagation, FIRST_TILE);
        // }

        tile_regs_release();

        enable_reload = true;
    }

    cb_pop_front(cb_acc, ONE_TILE);
}

// FORCE_INLINE void cumsum_tile_axis_2_and_3(
//     uint32_t row_depth, uint32_t cb_start, uint32_t cb_acc, uint32_t cb_input, uint32_t cb_next_stage, bool
//     should_pass_to_propagation, uint32_t cb_pass_to_propagation = CB_INVALID) { ReadCBGuard
//     start_cb_read_guard{cb_start, ONE_TILE};  // consume cb_start

//     bool enable_reload = false;

//     for (uint32_t row_tile_i = 0; row_tile_i < row_depth; ++row_tile_i) {
//             // DPRINT << "COMPUTE row_tile_i: " << row_tile_i << " column_tile_i: " << column_tile_i << ENDL();
//         DPRINT << row_tile_i << "  00000000000" << ENDL();
//         tile_regs_acquire();
//         const uint32_t cb_op = enable_reload ? cb_acc : cb_start;
//         DPRINT << row_tile_i << "  11111111 xxxxxx" << ENDL();
//         cb_wait_front(cb_input, ONE_TILE);
//         DPRINT << row_tile_i << "  11111111 yyyyyyyyy" << ENDL();

//         add_tiles_init(cb_input, cb_op);
//         DPRINT << row_tile_i << "  1111111111 zzzzzzzzzz" << ENDL();
//         add_tiles(cb_input, cb_op, FIRST_TILE, FIRST_TILE, WORKING_REG);

//         DPRINT << row_tile_i << "  2222222222" << ENDL();

//         cb_pop_front(cb_input, ONE_TILE);
//         if (enable_reload) {
//             cb_pop_front(cb_acc, ONE_TILE);
//         }

//         tile_regs_commit();
//         tile_regs_wait();

//         cb_reserve_back(cb_acc, ONE_TILE);
//         pack_reconfig_data_format(cb_acc);
//         pack_tile(WORKING_REG, cb_acc);
//         cb_push_back(cb_acc, ONE_TILE);
//         DPRINT << row_tile_i << "  3333333333333" << ENDL();

//         tile_regs_release();

//         tile_regs_acquire();

//         cb_wait_front(cb_acc, ONE_TILE);
//         copy_tile_init(cb_acc);
//         copy_tile(cb_acc, FIRST_TILE, WORKING_REG);

//         DPRINT << row_tile_i << "  44444444444444444 aaaaaaaa" << ENDL();

//         cumsum_tile_init();
//         DPRINT << row_tile_i << "  44444444444444444 bbbbbbbbb" << ENDL();
//         cumsum_tile(WORKING_REG);
//         DPRINT << row_tile_i << "  44444444444444444 ccccccccc" << ENDL();

//         tile_regs_commit();
//         tile_regs_wait();
//         DPRINT << row_tile_i << "  555555555" << ENDL();

//         { // must always copy to some next stage to pass it further - in other words, it is produced
//             WriteCBGuard cb_next_stage_guard{cb_next_stage, ONE_TILE};
//             pack_reconfig_data_format(cb_next_stage);
//             pack_tile(WORKING_REG, cb_next_stage, FIRST_TILE);
//         }

//         // if ((tile_i == block_depth - 1) && save_last_tile) {
//         //     WriteCBGuard axis_2_buffer_cb_guard{cb_axis_2_buffer, ONE_TILE};
//         //     pack_reconfig_data_format(cb_axis_2_buffer);
//         //     pack_tile(WORKING_REG, cb_axis_2_buffer, FIRST_TILE);
//         // }

//         // !!!!!!!!!!! DO NOT FORGET THAT IT WOULDN'T OCCUR FOR EVERY FIRST ROW CHUNK (ALL CHUNKS AT THE TOP OF THE
//         INPUT TENSOR) !!!!!!!!!! if (should_pass_to_propagation) { //
//             WriteCBGuard cb_pass_to_propagation_guard{cb_pass_to_propagation, ONE_TILE};
//             pack_reconfig_data_format(cb_pass_to_propagation);
//             pack_tile(WORKING_REG, cb_pass_to_propagation, FIRST_TILE);
//         }

//         tile_regs_release();

//         enable_reload = true;
//         DPRINT << row_tile_i << "  66666666" << ENDL();
//     }

//     cb_pop_front(cb_acc, ONE_TILE);
// }

template <typename ctas_t>
FORCE_INLINE void get_and_propagate_adder_tile_from_reader(
    // uint32_t cb_before_adder_propagation_tile, // from compute
    // uint32_t cb_axis_3_buffer_read, // from reader (buffer 1)
    // uint32_t cb_to_bot_core,
    // uint32_t cb_output,
    const ctas_t& ctas,
    bool should_pass_to_lower_core) {
    // ReadCBGuard cb_before_adder_propagation_tile_guard{
    //     ctas.before_adder_propagation_tile_cb, ONE_TILE};                           // tile from above to propagate
    //     onto
    // ReadCBGuard cb_axis_3_buffer_read_guard{ctas.axis_3_buffer_1_cb, ONE_TILE};  // propagation tile
    // WriteCBGuard cb_to_bot_core_guard{cb_to_bot_core, ONE_TILE};   // tile that gets propagated lower
    // WriteCBGuard cb_output_guard{
    //     ctas.output_cb,
    //     ONE_TILE};  // same tile as above, but the one that goes straight to the output (compute -> writer -> DRAM)
    tile_regs_acquire();

    {
        ReadCBGuard cb_before_adder_propagation_guard{
            ctas.before_adder_propagation_cb, ONE_TILE};  // tile from above to propagate onto
        ReadCBGuard cb_axis_3_buffer_read_guard{ctas.axis_3_buffer_1_cb, ONE_TILE};  // propagation tile

        add_tiles_init(ctas.before_adder_propagation_cb, ctas.axis_3_buffer_1_cb);
        add_tiles(ctas.before_adder_propagation_cb, ctas.axis_3_buffer_1_cb, FIRST_TILE, FIRST_TILE, WORKING_REG);
    }
    // add_tiles_init(ctas.before_adder_propagation_tile_cb, ctas.axis_3_buffer_1_cb);
    // add_tiles(ctas.before_adder_propagation_tile_cb, ctas.axis_3_buffer_1_cb, FIRST_TILE, FIRST_TILE, WORKING_REG);

    tile_regs_commit();
    tile_regs_wait();

    if (should_pass_to_lower_core) {
        WriteCBGuard cb_to_bot_core_guard{ctas.to_bot_core_cb, ONE_TILE};  // tile that gets propagated lower
        pack_reconfig_data_format(ctas.to_bot_core_cb);
        pack_tile(WORKING_REG, ctas.to_bot_core_cb, FIRST_TILE);
    }
    // pack_reconfig_data_format(cb_to_bot_core);
    // pack_tile(WORKING_REG, cb_to_bot_core, FIRST_TILE);
    {
        WriteCBGuard cb_output_guard{
            ctas.output_cb,
            ONE_TILE};  // same tile as above, but the one that goes straight to the output (compute -> writer -> DRAM)
        pack_reconfig_data_format(ctas.output_cb);
        pack_tile(WORKING_REG, ctas.output_cb, FIRST_TILE);
    }
    // pack_reconfig_data_format(ctas.output_cb);
    // pack_tile(WORKING_REG, ctas.output_cb, FIRST_TILE);

    tile_regs_release();
}

// template <typename ctas_t>
// FORCE_INLINE void cumsum_tile_axis_2_and_3_with_propagation(
//     const ctas_t& ctas, bool should_propagate_adder_tile_from_reader, bool should_pass_to_lower_core) {
//     // uint32_t row_depth, uint32_t cb_start, uint32_t cb_acc, uint32_t cb_input, uint32_t cb_next_stage, bool
//     // should_pass_to_propagation, uint32_t cb_pass_to_propagation = CB_INVALID) {
//     ReadCBGuard start_cb_read_guard{ctas.start_cb, ONE_TILE};  // consume cb_start

//     bool enable_reload = false;

//     for (uint32_t row_tile_i = 0; row_tile_i < ctas.input_depth; ++row_tile_i) {
//         DPRINT << row_tile_i << "  BEGIN" << ENDL();
//         // DPRINT << "COMPUTE row_tile_i: " << row_tile_i << " column_tile_i: " << column_tile_i << ENDL();
//         // DPRINT << row_tile_i << "  00000000000" << ENDL();
//         tile_regs_acquire();
//         const uint32_t cb_op = enable_reload ? ctas.acc_cb : ctas.start_cb;
//         cb_wait_front(ctas.input_cb, ONE_TILE);

//         add_tiles_init(ctas.input_cb, cb_op);
//         add_tiles(ctas.input_cb, cb_op, FIRST_TILE, FIRST_TILE, WORKING_REG);

//         // DPRINT << row_tile_i << "  2222222222" << ENDL();

//         cb_pop_front(ctas.input_cb, ONE_TILE);
//         if (enable_reload) {
//             cb_pop_front(ctas.acc_cb, ONE_TILE);
//         }

//         tile_regs_commit();
//         tile_regs_wait();

//         cb_reserve_back(ctas.acc_cb, ONE_TILE);
//         pack_reconfig_data_format(ctas.acc_cb);
//         pack_tile(WORKING_REG, ctas.acc_cb);
//         cb_push_back(ctas.acc_cb, ONE_TILE);
//         // DPRINT << row_tile_i << "  3333333333333" << ENDL();

//         tile_regs_release();

//         tile_regs_acquire();

//         cb_wait_front(ctas.acc_cb, ONE_TILE);
//         copy_tile_init(ctas.acc_cb);
//         copy_tile(ctas.acc_cb, FIRST_TILE, WORKING_REG);

//         // DPRINT << row_tile_i << "  44444444444444444 aaaaaaaa" << ENDL();

//         // cumsum_tile_init();
//         // cumsum_tile(WORKING_REG);

//         tile_regs_commit();
//         tile_regs_wait();
//         // DPRINT << row_tile_i << "  555555555" << ENDL();

//         if (should_propagate_adder_tile_from_reader) {
//             // pack to cb_before_adder_propagation, it will then be passed to cb_output and cb_to_bot_core by
//             // get_and_propagate_adder_tile_from_reader
//             {
//                 WriteCBGuard cb_before_adder_propagation_guard{ctas.before_adder_propagation_cb, ONE_TILE};
//                 pack_reconfig_data_format(ctas.before_adder_propagation_cb);
//                 pack_tile(WORKING_REG, ctas.before_adder_propagation_cb, FIRST_TILE);
//             }

//             tile_regs_release();
//             // DPRINT << row_tile_i << "  55555 aaaaa" << ENDL();

//             // // pack to cb_to_bot_core
//             // if (should_pass_to_lower_core) {
//             //     WriteCBGuard cb_to_bot_core_guard{cb_to_bot_core, ONE_TILE};
//             //     pack_reconfig_data_format(cb_to_bot_core);
//             //     pack_tile(WORKING_REG, cb_to_bot_core, ONE_TILE);
//             // }

//             // wait for tile spawned by reader from broadcasting last row of top tile passed by top writer onto all
//             rows get_and_propagate_adder_tile_from_reader(ctas, should_pass_to_lower_core);
//             // DPRINT << row_tile_i << "  5555555 bbbbbbb" << ENDL();
//         } else {
//             // pack to cb_output
//             {
//                 WriteCBGuard cb_output_guard{ctas.output_cb, ONE_TILE};
//                 pack_reconfig_data_format(ctas.output_cb);
//                 pack_tile(WORKING_REG, ctas.output_cb, FIRST_TILE);
//             }

//             // DPRINT << row_tile_i << "  5555555 cccccc" << ENDL();
//             // pack to cb_to_bot_core
//             if (should_pass_to_lower_core) {
//                 WriteCBGuard cb_to_bot_core_guard{ctas.to_bot_core_cb, ONE_TILE};
//                 pack_reconfig_data_format(ctas.to_bot_core_cb);
//                 pack_tile(WORKING_REG, ctas.to_bot_core_cb, FIRST_TILE);
//             }

//             tile_regs_release();
//         }

//         enable_reload = true;
//         // DPRINT << row_tile_i << "  66666666" << ENDL();
//         DPRINT << row_tile_i << "  END" << ENDL();
//     }

//     cb_pop_front(ctas.acc_cb, ONE_TILE);
// }

FORCE_INLINE void cumsum_tile_axis_3(
    uint32_t cb_previous_stage,
    uint32_t cb_next_stage,
    bool should_pass_to_lower_core = false,
    uint32_t cb_to_bot_core = CB_INVALID) {
    ReadCBGuard cb_previous_stage_guard{cb_previous_stage, ONE_TILE};
    WriteCBGuard cb_next_stage_guard{cb_next_stage, ONE_TILE};

    tile_regs_acquire();

    copy_tile_init(cb_previous_stage);
    copy_tile(cb_previous_stage, FIRST_TILE, WORKING_REG);

    cumsum_tile_init();
    cumsum_tile(WORKING_REG);

    tile_regs_commit();
    tile_regs_wait();

    pack_reconfig_data_format(cb_next_stage);
    pack_tile(WORKING_REG, cb_next_stage, FIRST_TILE);

    if (should_pass_to_lower_core) {
        WriteCBGuard cb_to_bot_core_guard{cb_to_bot_core, ONE_TILE};
        pack_reconfig_data_format(cb_to_bot_core);
        pack_tile(WORKING_REG, cb_to_bot_core, FIRST_TILE);
    }

    tile_regs_release();
}

// template <typename ctas_t>
// FORCE_INLINE void cumsum_tile_axis_2_and_3_and_pass_further(
//     const ctas_t& ctas, bool should_propagate_adder_tile_from_reader, bool should_pass_to_lower_core) {
//     // uint32_t row_depth, uint32_t cb_start, uint32_t cb_acc, uint32_t cb_input, uint32_t cb_next_stage, bool
//     // should_pass_to_propagation, uint32_t cb_pass_to_propagation = CB_INVALID) {
//     ReadCBGuard start_cb_read_guard{ctas.start_cb, ONE_TILE};  // consume cb_start

//     bool enable_reload = false;

//     for (uint32_t row_tile_i = 0; row_tile_i < ctas.input_depth; ++row_tile_i) {
//         DPRINT << row_tile_i << "  BEGIN" << ENDL();
//         // DPRINT << "COMPUTE row_tile_i: " << row_tile_i << " column_tile_i: " << column_tile_i << ENDL();
//         // DPRINT << row_tile_i << "  00000000000" << ENDL();
//         tile_regs_acquire();
//         const uint32_t cb_op = enable_reload ? ctas.acc_cb : ctas.start_cb;
//         cb_wait_front(ctas.input_cb, ONE_TILE);

//         add_tiles_init(ctas.input_cb, cb_op);
//         add_tiles(ctas.input_cb, cb_op, FIRST_TILE, FIRST_TILE, WORKING_REG);

//         // DPRINT << row_tile_i << "  2222222222" << ENDL();

//         cb_pop_front(ctas.input_cb, ONE_TILE);
//         if (enable_reload) {
//             cb_pop_front(ctas.acc_cb, ONE_TILE);
//         }

//         tile_regs_commit();
//         tile_regs_wait();

//         cb_reserve_back(ctas.acc_cb, ONE_TILE);
//         pack_reconfig_data_format(ctas.acc_cb);
//         pack_tile(WORKING_REG, ctas.acc_cb);
//         cb_push_back(ctas.acc_cb, ONE_TILE);
//         // DPRINT << row_tile_i << "  3333333333333" << ENDL();

//         tile_regs_release();

//         tile_regs_acquire();

//         cb_wait_front(ctas.acc_cb, ONE_TILE);
//         copy_tile_init(ctas.acc_cb);
//         copy_tile(ctas.acc_cb, FIRST_TILE, WORKING_REG);

//         // DPRINT << row_tile_i << "  44444444444444444 aaaaaaaa" << ENDL();

//         // cumsum_tile_init();
//         // cumsum_tile(WORKING_REG);

//         tile_regs_commit();
//         tile_regs_wait();
//         // DPRINT << row_tile_i << "  555555555" << ENDL();

//         if (should_propagate_adder_tile_from_reader) {
//             // pack to cb_before_adder_propagation, it will then be passed to cb_output and cb_to_bot_core by
//             // get_and_propagate_adder_tile_from_reader
//             {
//                 WriteCBGuard cb_before_adder_propagation_guard{ctas.before_adder_propagation_cb, ONE_TILE};
//                 pack_reconfig_data_format(ctas.before_adder_propagation_cb);
//                 pack_tile(WORKING_REG, ctas.before_adder_propagation_cb, FIRST_TILE);
//             }

//             tile_regs_release();
//             // DPRINT << row_tile_i << "  55555 aaaaa" << ENDL();

//             // // pack to cb_to_bot_core
//             // if (should_pass_to_lower_core) {
//             //     WriteCBGuard cb_to_bot_core_guard{cb_to_bot_core, ONE_TILE};
//             //     pack_reconfig_data_format(cb_to_bot_core);
//             //     pack_tile(WORKING_REG, cb_to_bot_core, ONE_TILE);
//             // }

//             // wait for tile spawned by reader from broadcasting last row of top tile passed by top writer onto all
//             rows get_and_propagate_adder_tile_from_reader(ctas, should_pass_to_lower_core);
//             // DPRINT << row_tile_i << "  5555555 bbbbbbb" << ENDL();
//         } else {
//             // pack to cb_output
//             {
//                 WriteCBGuard cb_output_guard{ctas.output_cb, ONE_TILE};
//                 pack_reconfig_data_format(ctas.output_cb);
//                 pack_tile(WORKING_REG, ctas.output_cb, FIRST_TILE);
//             }

//             // DPRINT << row_tile_i << "  5555555 cccccc" << ENDL();
//             // pack to cb_to_bot_core
//             if (should_pass_to_lower_core) {
//                 WriteCBGuard cb_to_bot_core_guard{ctas.to_bot_core_cb, ONE_TILE};
//                 pack_reconfig_data_format(ctas.to_bot_core_cb);
//                 pack_tile(WORKING_REG, ctas.to_bot_core_cb, FIRST_TILE);
//             }

//             tile_regs_release();
//         }

//         enable_reload = true;
//         // DPRINT << row_tile_i << "  66666666" << ENDL();
//         DPRINT << row_tile_i << "  END" << ENDL();
//     }

//     cb_pop_front(ctas.acc_cb, ONE_TILE);
// }

// FORCE_INLINE void cumsum_tile_axis_3(uint32_t cb_cumsum_stage_wip, uint32_t cb_cumsum_output) {
//     for (uint32_t tile_i = 0; tile_i < block_depth; ++tile_i) {
//         ReadCBGuard read_cumsum_guard{cb_cumsum_stage_wip, ONE_TILE};
//         WriteCBGuard cumsum_output_write_guard{cb_cumsum_output, ONE_TILE};
//         tile_regs_acquire();
//         copy_tile_init(cb_cumsum_stage_wip);
//         copy_tile(cb_cumsum_stage_wip, FIRST_TILE, WORKING_REG);

//         cumsum_tile_init();
//         cumsum_tile(WORKING_REG);

//         tile_regs_commit();
//         tile_regs_wait();

//         pack_reconfig_data_format(cb_cumsum_output);
//         pack_tile(WORKING_REG, cb_cumsum_output, FIRST_TILE);

//         tile_regs_release();
//     }
// }

template <typename ctas_t, typename rtas_t>
FORCE_INLINE void perform_intimg_along_channel_slice(
    const ctas_t& ctas, const rtas_t& rtas, uint32_t ending_tile_along_height) {
    // REMEMBER TO PUSH THE TILE ALSO TO THE CB THAT SERVES AS A STAGE BETWEEN CORES, NOT ONLY TO THE
    // OUTPUT!!!!!!!!!!!!!!
    for (uint32_t column_tile_i = rtas.starting_tile_along_height; column_tile_i < ending_tile_along_height;
         ++column_tile_i) {  // TODO(jbbieniekTT): this algorithm works correctly when `ending_tile_along_height -
                             // starting_tile_along_height == 1`, which is the case for input shapes up to [1, any, 160,
                             // any]
                             // bool propagation_enabled = (column_tile_i != ctas.num_tiles_along_height - 1);
        // uint32_t cb_next_stage = propagation_enabled ? ctas.before_adder_propagation_stage_cb : ctas.output_cb;
        // for (uint32_t row_tile_i = 0; row_tile_i < ctas.input_depth; ++row_tile_i) {
        // DPRINT << "COMPUTE row_tile_i: " << row_tile_i << " column_tile_i: " << column_tile_i << ENDL();
        const bool should_pass_to_lower_core = (column_tile_i < (ctas.num_tiles_along_height - 1));
        const bool should_propagate_adder_tile_from_reader = (column_tile_i > 0);
        // cumsum_tile_axis_2_and_3_with_propagation(
        //     ctas, should_propagate_adder_tile_from_reader, should_pass_to_lower_core);
        // cumsum_tile_axis_2_and_3_and_pass_further(ctas);
        cumsum_tile_axis_2(ctas.start_cb, ctas.acc_cb, ctas.input_cb, ctas.output_cb, ctas.input_depth);
        // if (should_propagate_adder_tile_from_reader) {
        //     cumsum_tile_axis_3(ctas.cumsum_axis_3, ctas.before_adder_propagation_cb);
        //     get_and_propagate_adder_tile_from_reader(ctas, should_pass_to_lower_core);
        //     // if (should_pass_to_lower_core) {
        //     //     //
        //     // }
        // } else {
        //     cumsum_tile_axis_3(ctas.cumsum_axis_3, ctas.output_cb, true, ctas.to_bot_core_cb);
        //     //
        //     // if (should_pass_to_lower_core) {
        //     //     //
        //     // }
        // }
        // if (column_tile_i == 0) {
        //     // output + lower (should_pass_to_lower_core)
        //     cumsum_tile_axis_2_and_3_with_propagation(ctas.input_depth, ctas.start_cb, ctas.acc_cb,
        //     ctas.input_cb, ctas.output_cb, true, ctas.to_bot_core_cb); // goes to writer DPRINT << "MMMMMMMM" <<
        //     ENDL();
        // } else if (column_tile_i == ctas.num_tiles_along_height - 1) {
        //     // propagate + output (should_propagate_adder_tile_from_reader)
        //     cumsum_tile_axis_2_and_3_with_propagation(ctas.input_depth, ctas.start_cb, ctas.acc_cb,
        //     ctas.input_cb, ctas.before_adder_propagation_cb, false); // tile still in compute
        //     get_and_propagate_adder_tile_from_reader(ctas.before_adder_propagation_cb, ctas.axis_3_buffer_1_cb,
        //     ctas.to_bot_core_cb, ctas.output_cb); // wait until reader has spawned a tile that it had received
        //     from the core that processes the row chunk right above it and apply (add) it to the designated stage
        //     before propagation, then pass to both stages, one designated to be sent to the lower core and another
        //     to the DRAM. DPRINT << "NNNNNNNNNNNN" << ENDL();
        // } else {
        //     // propagate + output + lower (should_propagate_adder_tile_from_reader + should_pass_to_lower_core)
        //     cumsum_tile_axis_2_and_3_with_propagation(ctas.input_depth, ctas.start_cb, ctas.acc_cb,
        //     ctas.input_cb, ctas.before_adder_propagation_cb, true, ctas.to_bot_core_cb); // tile both remains in
        //     compute and goes to the writer kernel of the current core to be sent to the reader kernel of the core
        //     that processes the next lower row chunk.
        //     get_and_propagate_adder_tile_from_reader(ctas.before_adder_propagation_cb, ctas.axis_3_buffer_1_cb,
        //     ctas.to_bot_core_cb, ctas.output_cb); DPRINT << "OOOOOOOOOOOOOO" << ENDL();
        // }

        // // !!!! DIFFERENT PARAMS THAN THE CONDITION ABOVE !!!! (!!!!!!!!one stage less!!!!!!!!!)
        // if (rows_block_i > 0) {
        //     // axis 3/4's propagation...
        //     // consume: cb_cumsum_stage_0 32t
        //     // produce: cumsummed down cube (cb_cumsum_stage_1 32t)
        //     cumsum_cube_axis_3(ctas.cumsum_stage_0_cb, ctas.cumsum_stage_1_cb, block_depth);
        //     // consume: cb_cumsum_stage_1 32t, axis_3_buffer_1_cb 32t
        //     // produce: cb_output 1t x32
        //     get_and_propagate_adder_cube(
        //         ctas.cumsum_stage_1_cb, ctas.axis_3_buffer_1_cb, ctas.output_cb, block_depth);
        // } else {
        //     // consume: cb_cumsum_stage_0 32t
        //     // produce: cumsummed down cube (cb_output 1t x32)
        //     cumsum_cube_axis_3(ctas.cumsum_stage_0_cb, ctas.output_cb, block_depth);
        // }

        // cumsum_tile_axis_2_and_3(ctas.start_cb, ctas.acc_cb, ctas.input_cb, cb_next_stage);
        // // if (column_tile_i != ctas.num_tiles_along_height - 1) { // propagate if currently processed row chunk
        // is
        // // not located at the "bottom" of the tensor, that is, last row chunk when seen along tile-height
        // //     // axis 3/4's propagation...

        // //     // WAIT FOR THE READER AND PERFORM LOCAL CUMSUMS
        // //     cumsum_tile_axis_2_and_3(
        // //         ctas.start_cb, ctas.acc_cb, ctas.input_cb, ctas.before_adder_propagation_stage_cb,
        // //         ctas.input_depth);
        // //     // APPLY THE ADDER TILE SPAWNED BY THE READER, THEN PA
        // //     get_and_propagate_adder_tile_from_reader(
        // //         ctas.before_adder_propagation_stage_cb, ctas.axis_3_buffer_1_cb, ctas.to_bot_core_cb,
        // //         ctas.output_cb);
        // // } else {
        // //     cumsum_tile_axis_2_and_3(ctas.start_cb, ctas.acc_cb, ctas.input_cb, ctas.output_cb,
        // //     ctas.input_depth);
        // // }
        // if (propagation_enabled) {
        //     get_and_propagate_adder_tile_from_reader(
        //         ctas.before_adder_propagation_stage_cb,
        //         ctas.axis_3_buffer_1_cb,
        //         ctas.to_bot_core_cb,
        //         ctas.output_cb);
        // }
        // }
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
        perform_intimg_along_channel_slice(ctas, rtas, ending_tile_along_height);
        // }
    }
}

}  // namespace NAMESPACE
