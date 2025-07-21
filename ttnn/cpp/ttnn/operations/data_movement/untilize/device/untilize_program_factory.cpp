// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_program_factory.hpp"

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

uint32_t get_largest_divisor(uint32_t dividend, uint32_t starting_divisor, uint32_t divisor_factor = 1) {
    for (uint32_t curr_div = starting_divisor; curr_div > 0; curr_div--) {
        if (dividend % (curr_div * divisor_factor) == 0) {
            return curr_div;
        }
    }
    return 1;
}

operation::ProgramWithCallbacks untilize_multi_core_sub_core_grids(
    const Tensor& a,
    Tensor& output,
    bool use_pack_untilize,
    bool fp32_dest_acc_en,
    const CoreRangeSet& sub_core_grids) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    IDevice* device = a.device();

    uint32_t ntiles = a.physical_volume() / TILE_HW;
    uint32_t ncores = sub_core_grids.num_cores();
    for (uint32_t core_id = ncores; core_id >= 1; core_id--) {
        if (ntiles % ncores == 0) {
            break;
        } else {
            ncores--;
        }
    }

    TT_ASSERT(ntiles % (ncores) == 0);

    uint32_t max_tiles = 1;
    uint32_t ntiles_per_block = ntiles / ncores;
    uint32_t stick_s = a.padded_shape()[-1];
    uint32_t ntiles_per_row = stick_s / TILE_WIDTH;
    uint32_t stick_size = stick_s * output.element_size();
    uint32_t ntiles_per_column = ntiles / ntiles_per_row;
    uint32_t starting_tile = ntiles_per_block;
    if (ntiles_per_row > max_tiles) {
        starting_tile = max_tiles;
    }
    ntiles_per_block = get_largest_divisor(ntiles_per_row, starting_tile);
    TT_ASSERT(
        ntiles_per_row % ntiles_per_block == 0 and ntiles_per_block >= 1 and ntiles_per_block <= ntiles_per_row and
        ntiles % ntiles_per_block == 0);

    uint32_t nblocks = (ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = input_single_tile_size;

    auto cores = corerange_to_cores(sub_core_grids, ncores, true);
    auto all_cores = num_cores_to_corerangeset_in_subcoregrids(cores[0], ncores, sub_core_grids, true);
    uint32_t nblocks_per_core = nblocks / ncores;

    bool row_major = true;
    bool src_block_sharded = false;
    uint32_t num_rows_block = 0, block_row_size = 0, output_row_size = 0, last_block_row_size_unpadded = 0,
             num_output_rows_unpadded = 0;
    CoreCoord end_core;
    std::vector<CoreCoord> cores_with_rtargs;

    uint32_t num_input_tiles = ntiles_per_block * 2;
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0, program, all_cores, input_single_tile_size, num_input_tiles, input_cb_data_format, nullptr);

    uint32_t num_output_tiles = ntiles_per_block * 2;
    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        all_cores,
        output_single_tile_size,
        num_output_tiles,
        output_cb_data_format,
        nullptr);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM;
    std::vector<uint32_t> reader_ct_args = {(uint32_t)src0_is_dram};

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    bool out_is_dram = dst_buffer->buffer_type() == BufferType::DRAM;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(stick_size) : 0;
    std::vector<uint32_t> writer_ct_args = {
        (uint32_t)out_is_dram,
        (uint32_t)stick_size_is_power_of_two,
        (uint32_t)log2_stick_size,
    };

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    /** compute
     */
    std::vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
        (uint32_t)src0_cb_index,
        (uint32_t)output_cb_index};

    std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize || a.dtype() == DataType::UINT16) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel =
            std::string("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
    }

    auto untilize_kernel_id = CreateKernel(
        program,
        compute_kernel,
        all_cores,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args});

    uint32_t tile_start_id = 0;
    uint32_t offset_within_stick = 0;

    auto nsticks_per_core = ntiles_per_column * TILE_HEIGHT;

    for (uint32_t i = 0; i < cores.size(); i++) {
        CoreCoord core = cores[i];

        // reader runtime args
        auto ntiles_per_core = ntiles_per_block * nblocks_per_core;
        const std::array reader_rt_args = {
            src0_buffer->address(),  // src_addr
            ntiles_per_core,         // ntiles
            tile_start_id            // start_id
        };

        const std::array writer_rt_args = {
            dst_buffer->address(),               // dst_addr
            nsticks_per_core,                    // nsticks
            stick_size,                          // block_size_nbytes
            ntiles_per_core,                     // ntiles_per_core
            TILE_WIDTH * output.element_size(),  // tile_width_size
            std::uint32_t{0},                    // start stick id = 0, since parallelizing on height
            offset_within_stick};

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);
        cores_with_rtargs.push_back(core);
        tile_start_id += ntiles_per_core;
        offset_within_stick += ntiles_per_core * TILE_WIDTH * output.element_size();
    }

    auto override_runtime_arguments_callback = [reader_kernel_id = reader_kernel_id,
                                                writer_kernel_id = writer_kernel_id,
                                                cb_src0 = cb_src0,
                                                cb_output = cb_output,
                                                cores_with_rtargs](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();
        {
            auto& runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            for (const CoreCoord& core : cores_with_rtargs) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer->address();
            }
        }

        {
            auto& runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (const CoreCoord& core : cores_with_rtargs) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks untilize_multi_core_parallelize_column(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    IDevice* device = a.device();

    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t ntiles = a.physical_volume() / TILE_HW;
    uint32_t ncores_x = grid_size.x;
    uint32_t ncores_y = grid_size.y;
    // uint32_t ncores_x = 2;

    ncores_x = get_largest_divisor(ntiles, ncores_x);
    // uint32_t ncores_y = 1;
    ncores_y = get_largest_divisor(ntiles, ncores_y, ncores_x);

    TT_ASSERT(ntiles % (ncores_x * ncores_y) == 0);
    uint32_t ntiles_per_block = ntiles / (ncores_x * ncores_y);

    // TODO increase block size to increase untilize performance, currently each untilize block is a single tile
    // uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 -
    // a.device()->allocator()->get_base_allocator_addr(HalMemType::L1); uint32_t max_tiles =
    // (max_l1_size / (input_single_tile_size + output_single_tile_size))/2;  // 2 CBs, double buffering each
    uint32_t max_tiles = 1;

    uint32_t stick_s = a.padded_shape()[-1];
    uint32_t ntiles_per_row = stick_s / TILE_WIDTH;
    uint32_t stick_size = stick_s * output.element_size();
    uint32_t ntiles_per_column = ntiles / ntiles_per_row;
    uint32_t starting_tile = ntiles_per_block;
    if (ntiles_per_row > max_tiles) {
        starting_tile = max_tiles;
    }
    ntiles_per_block = get_largest_divisor(ntiles_per_row, starting_tile);
    TT_ASSERT(
        ntiles_per_row % ntiles_per_block == 0 and ntiles_per_block >= 1 and ntiles_per_block <= ntiles_per_row and
        ntiles % ntiles_per_block == 0);

    uint32_t nblocks = (ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = input_single_tile_size;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(CoreCoord(ncores_x, ncores_y), nblocks);

    bool row_major = true;
    bool src_block_sharded = false;
    uint32_t num_rows_block = 0, block_row_size = 0, output_row_size = 0, last_block_row_size_unpadded = 0,
             num_output_rows_unpadded = 0;
    CoreCoord end_core;
    std::vector<CoreCoord> cores_with_rtargs;

    uint32_t num_input_tiles = ntiles_per_block * 2;
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0, program, all_cores, input_single_tile_size, num_input_tiles, input_cb_data_format, nullptr);

    uint32_t num_output_tiles = ntiles_per_block * 2;
    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        all_cores,
        output_single_tile_size,
        num_output_tiles,
        output_cb_data_format,
        nullptr);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM;
    std::vector<uint32_t> reader_ct_args = {(uint32_t)src0_is_dram};

    auto unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    bool out_is_dram = dst_buffer->buffer_type() == BufferType::DRAM;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(stick_size) : 0;
    std::vector<uint32_t> writer_ct_args = {
        (uint32_t)out_is_dram,
        (uint32_t)stick_size_is_power_of_two,
        (uint32_t)log2_stick_size,
    };

    auto unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    /** compute
     */
    std::vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
        (uint32_t)src0_cb_index,
        (uint32_t)output_cb_index};
    std::vector<uint32_t> compute_args_cliff = {
        (uint32_t)nblocks_per_core_cliff,
        (uint32_t)ntiles_per_block,  // per_block_ntiles
        (uint32_t)src0_cb_index,
        (uint32_t)output_cb_index};

    std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize || a.dtype() == DataType::UINT16) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel =
            std::string("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
    }

    if (core_range.ranges().size() > 0) {
        auto untilize_kernel_id = CreateKernel(
            program,
            compute_kernel,
            core_range,
            ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args});
    }
    if (core_range_cliff.ranges().size() > 0) {
        auto untilize_cliff_kernel_id = CreateKernel(
            program,
            compute_kernel,
            core_range_cliff,
            ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args_cliff});
    }

    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        // unequal case with cliff
        ncores_full -= 1;
        full_cores = core_range;
    }
    uint32_t tile_start_id = 0;
    uint32_t offset_within_stick = 0;
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);

    auto nsticks_per_core = ntiles_per_column * TILE_HEIGHT;

    for (uint32_t i = 0; i < cores.size(); i++) {
        CoreCoord core = cores[i];
        if (!full_cores.contains(core)) {
            continue;
        }
        // reader runtime args
        auto ntiles_per_core = ntiles_per_block * nblocks_per_core;
        const std::array reader_rt_args = {
            src0_buffer->address(),  // src_addr
            ntiles_per_core,         // ntiles
            tile_start_id            // start_id
        };

        const std::array writer_rt_args = {
            dst_buffer->address(),               // dst_addr
            nsticks_per_core,                    // nsticks
            stick_size,                          // block_size_nbytes
            ntiles_per_core,                     // ntiles_per_core
            TILE_WIDTH * output.element_size(),  // tile_width_size
            std::uint32_t{0},                    // start stick id = 0, since parallelizing on height
            offset_within_stick};

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
        cores_with_rtargs.push_back(core);
        tile_start_id += ntiles_per_core;
        offset_within_stick += ntiles_per_core * TILE_WIDTH * output.element_size();
    }
    if (ncores_full < ncores) {
        // last core is the cliff core with nblocks_per_core_cliff blocks
        CoreCoord core = row_major ? CoreCoord{ncores_full % ncores_x, ncores_full / ncores_x}
                                   : CoreCoord{ncores_full / ncores_y, ncores_full % ncores_y};
        // reader runtime args
        auto ntiles_per_core_cliff = ntiles_per_block * nblocks_per_core_cliff;
        const std::array reader_rt_args = {
            src0_buffer->address(),  // src_addr
            ntiles_per_core_cliff,   // ntiles
            tile_start_id            // start_id
        };

        const std::array writer_rt_args = {
            dst_buffer->address(),               // dst_addr
            nsticks_per_core,                    // nsticks
            stick_size,                          // block_size_nbytes
            ntiles_per_core_cliff,               // ntiles_per_core
            TILE_WIDTH * output.element_size(),  // tile_width_size
            std::uint32_t{0},                    // start stick id = 0, since parallelizing on height
            offset_within_stick};
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
    }

    auto override_runtime_arguments_callback = [reader_kernel_id = unary_reader_kernel_id,
                                                writer_kernel_id = unary_writer_kernel_id,
                                                cb_src0 = cb_src0,
                                                cb_output = cb_output,
                                                cores_with_rtargs](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();
        {
            auto& runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            for (const CoreCoord& core : cores_with_rtargs) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer->address();
            }
        }

        {
            auto& runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (const CoreCoord& core : cores_with_rtargs) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks untilize_multi_core_block(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    tt::tt_metal::Program program{};
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();

    uint32_t a_tile_width = a.tensor_spec().tile().get_width();
    uint32_t a_tile_height = a.tensor_spec().tile().get_height();

    uint32_t num_tiles_per_row = a.padded_shape()[-1] / a_tile_width;
    uint32_t num_tiles_per_col = a.padded_shape()[-2] / a_tile_height;

    uint32_t num_blocks = (a.padded_shape()[-1] * a.padded_shape()[-2]) / (a_tile_height * a_tile_width);

    auto
        [ncores,
         all_cores,
         core_range,
         cliff_row_core_range,
         cliff_col_core_range,
         cliff_col_row_core_range,
         nblocks_per_core,
         single_block_size,
         single_block_size_cliff_row,
         single_block_size_cliff_col,
         has_cliff_row,
         has_cliff_col,
         full_cores_per_row,
         full_cores_per_col] =
            ttnn::split_blocks_for_tilize_wh(grid_size, num_blocks, num_tiles_per_row, num_tiles_per_col);

    uint32_t total_tiles_per_row = full_cores_per_row * single_block_size + has_cliff_row * single_block_size_cliff_row;
    uint32_t row_size_bytes;

    uint32_t el_size = a.element_size();
    if (a.dtype() == DataType::BFLOAT8_B) {
        row_size_bytes = input_shape[-1] * output.element_size();
        el_size = output.element_size();
    } else {
        row_size_bytes = input_shape[-1] * a.element_size();
    }

    if (core_range.size() > 0) {
        create_cb(
            tt::CBIndex::c_0, program, core_range, input_single_tile_size, single_block_size, input_cb_data_format);

        create_cb(
            tt::CBIndex::c_16, program, core_range, output_single_tile_size, single_block_size, output_cb_data_format);
    }

    if (has_cliff_col && has_cliff_row) {
        create_cb(
            tt::CBIndex::c_0,
            program,
            cliff_col_row_core_range,
            input_single_tile_size,
            single_block_size_cliff_row,
            input_cb_data_format);

        create_cb(
            tt::CBIndex::c_16,
            program,
            cliff_col_row_core_range,
            output_single_tile_size,
            single_block_size_cliff_row,
            output_cb_data_format);
    }

    if (has_cliff_row) {
        create_cb(
            tt::CBIndex::c_0,
            program,
            cliff_row_core_range,
            input_single_tile_size,
            single_block_size_cliff_row,
            input_cb_data_format);

        create_cb(
            tt::CBIndex::c_16,
            program,
            cliff_row_core_range,
            output_single_tile_size,
            single_block_size_cliff_row,
            output_cb_data_format);
    }

    if (has_cliff_col) {
        auto [src3_cb_index, cb_src3] = create_cb(
            tt::CBIndex::c_0,
            program,
            cliff_col_core_range,
            input_single_tile_size,
            single_block_size,
            input_cb_data_format);

        auto [output3_cb_index, cb_output3] = create_cb(
            tt::CBIndex::c_16,
            program,
            cliff_col_core_range,
            output_single_tile_size,
            single_block_size,
            output_cb_data_format);
    }

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // reader

    uint32_t src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    uint32_t num_tiles_2d = a.padded_shape()[-1] * a.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp",
        all_cores,
        ReaderDataMovementConfig({src0_is_dram, num_tiles_2d, third_dim, total_tiles_per_row}));

    // writer

    uint32_t out_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = row_size_bytes;
    uint32_t stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(stick_size) : 0;

    uint32_t total_num_rows = output.logical_shape()[-2];
    std::map<std::string, std::string> writer_defines = {
        {"STICK_SIZE_IS_POW2", std::to_string((uint32_t)(stick_size_is_power_of_two))}};

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
        "writer_unary_stick_layout_wh_multicore.cpp",
        all_cores,
        WriterDataMovementConfig(
            {out_is_dram, log2_stick_size, total_num_rows, third_dim, TILE_HEIGHT}, writer_defines));

    // compute

    if (core_range.size() > 0) {
        auto untilize_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp",
            core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = {single_block_size, single_block_size, third_dim}});
    }
    if (has_cliff_col && has_cliff_row) {
        auto tilize_col_row_cliff_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp",
            cliff_col_row_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = {single_block_size_cliff_col, single_block_size_cliff_row, third_dim}});
    }
    if (has_cliff_row) {
        auto tilize_row_cliff_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp",
            cliff_row_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = {single_block_size, single_block_size_cliff_row, third_dim}});
    }

    if (has_cliff_col) {
        auto tilize_col_cliff_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp",
            cliff_col_core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = {single_block_size_cliff_col, single_block_size, third_dim}});
    }

    // RUNTIME ARGS
    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
    uint32_t start_row_id = 0;
    uint32_t start_column_id = 0;
    uint32_t tile_start_id = 0;
    uint32_t single_block_size_row_arg;
    uint32_t single_block_size_col_arg;

    uint32_t total_row_cores = full_cores_per_row;
    if (has_cliff_row) {
        total_row_cores++;
    }
    uint32_t cores_col_count = 1;

    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];

        if (has_cliff_col && has_cliff_row && i == ncores - 1) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size_cliff_col;

        } else if (has_cliff_row && i != 0 && ((i + 1) % (full_cores_per_row + 1)) == 0) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size;

        } else if (i < total_row_cores * full_cores_per_col) {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size;

        } else {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size_cliff_col;
        }

        //  writer runtime args
        std::vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            row_size_bytes,
            TILE_WIDTH * el_size * single_block_size_row_arg,
            start_row_id,
            start_column_id,
            single_block_size_row_arg,
            single_block_size_col_arg,
        };

        // reader runtime args
        const std::array reader_rt_args = {
            src0_buffer->address(), tile_start_id, single_block_size_row_arg, single_block_size_col_arg};
        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        uint32_t end_column_id = start_column_id + single_block_size_row_arg * TILE_WIDTH * el_size;
        start_column_id = end_column_id % row_size_bytes;
        if (end_column_id % row_size_bytes == 0 && end_column_id != 0) {
            start_row_id += single_block_size_col_arg * TILE_HEIGHT;
        }

        if (start_column_id == 0) {
            tile_start_id = cores_col_count * single_block_size_col_arg * total_tiles_per_row;
            cores_col_count++;
        } else {
            tile_start_id += single_block_size_row_arg;
        }
    }

    auto override_runtime_args_callback =
        [reader_kernel_id = unary_reader_kernel_id, writer_kernel_id = unary_writer_kernel_id, cores = cores](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_tensors,
                                              const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors.at(0).buffer();
            auto dst_buffer = output_tensors.at(0).buffer();

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);

            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = src_buffer->address();
                }
                {
                    auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = dst_buffer->address();
                }
            }
        };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    tt::tt_metal::IDevice* device = a.device();
    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    ShardSpec shard_spec = a.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];

    uint32_t num_tiles_per_block = shard_width / tile_width;
    uint32_t num_blocks_per_core = shard_height / tile_height;
    uint32_t num_tiles_per_shard = num_tiles_per_block * num_blocks_per_core;

    // Input CB
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0,
        program,
        shard_spec.grid,
        input_single_tile_size,
        num_tiles_per_shard,
        input_cb_data_format,
        src0_buffer);

    // Output CB
    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        shard_spec.grid,
        output_single_tile_size,
        num_tiles_per_shard,
        output_cb_data_format,
        dst_buffer);

    // Reader compile-time args
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index};

    // Reader kernel
    KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        shard_spec.grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Writer compile-time args
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)output_cb_index};

    // Writer kernel
    KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
        shard_spec.grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Compute compile-time args
    std::vector<uint32_t> compute_compile_time_args = {
        (uint32_t)num_blocks_per_core,
        (uint32_t)num_tiles_per_block,
        (uint32_t)src0_cb_index,
        (uint32_t)output_cb_index};

    // Compute kernel
    std::string compute_kernel;
    if (num_tiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize || a.dtype() == DataType::UINT16) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel =
            std::string("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
        compute_kernel =
            std::string("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    }
    KernelHandle untilize_kernel_id = CreateKernel(
        program,
        compute_kernel,
        shard_spec.grid,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_compile_time_args});

    // Run-time args
    auto cores =
        corerange_to_cores(shard_spec.grid, std::nullopt, shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    for (uint32_t i = 0; i < cores.size(); ++i) {
        CoreCoord core = cores[i];

        // Reader run-time args
        uint32_t num_tiles_to_read = num_tiles_per_block * num_blocks_per_core;
        std::vector<uint32_t> reader_run_time_args = {num_tiles_to_read};

        // Writer run-time args
        uint32_t num_tiles_to_write = num_tiles_per_block * num_blocks_per_core;
        std::vector<uint32_t> writer_run_time_args = {num_tiles_to_write};

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_run_time_args);
    }

    auto override_runtime_args_callback = [reader_kernel_id = unary_reader_kernel_id,
                                           writer_kernel_id = unary_writer_kernel_id,
                                           cb_src0 = cb_src0,
                                           cb_output = cb_output](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks untilize_multi_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    tt::tt_metal::IDevice* device = a.device();
    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t tensor_width = a.padded_shape()[-1];
    uint32_t tensor_height = a.physical_volume() / tensor_width;

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_volume = tile_height * tile_width;

    bool input_is_sharded = a.is_sharded();
    bool output_is_sharded = output.is_sharded();

    uint32_t num_tiles_per_row = tensor_width / tile_width;
    uint32_t num_tiles_per_col = tensor_height / tile_height;

    auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_compute_cores,
         compute_core_range,
         full_compute_core_range,
         cliff_compute_core_range,
         num_rows_per_full_core,
         num_rows_per_cliff_core] = ttnn::split_blocks_for_tilize(grid_size, num_tiles_per_col);

    constexpr uint32_t threshold_row_block = 32;
    if (!input_is_sharded and !output_is_sharded) {
        if (num_tiles_per_row > threshold_row_block) {
            if (num_tiles_per_col > threshold_row_block || num_tiles_per_row > num_tiles_per_col) {
                uint32_t num_blocks_block = (a.padded_shape()[-1] * a.padded_shape()[-2]) / (TILE_HEIGHT * TILE_WIDTH);

                auto
                    [ncores_block,
                     all_cores_block,
                     core_range_block,
                     cliff_row_core_range,
                     cliff_col_core_range,
                     cliff_col_row_core_range,
                     nblocks_per_core_block,
                     single_block_size,
                     single_block_size_cliff_row,
                     single_block_size_cliff_col,
                     has_cliff_row,
                     has_cliff_col,
                     full_cores_per_row,
                     full_cores_per_col] =
                        ttnn::split_blocks_for_tilize_wh(
                            grid_size, num_blocks_block, num_tiles_per_row, num_tiles_per_col);
                if (num_compute_cores < ncores_block) {
                    return untilize_multi_core_block(a, output, use_pack_untilize, fp32_dest_acc_en);
                }
            }
        }
    }

    // TODO (#23449): This memory calculation is a) outdated/inaccurate and needs to be fixed and b) needs
    // to be moved up a few layers as the available memory may be different upon a program cache hit

    // Determine how much L1 space we can use for input and output CBs,
    // ensuring that we don't intrude into other L1 storage space
    uint32_t max_l1_size =
        device->l1_size_per_core() / 2 - device->allocator()->get_base_allocator_addr(HalMemType::L1);

    // Determine the max number of tiles that can be in any CB at a given time (1 input CB + 1 output CB = 2 total CBs)
    uint32_t max_tiles_per_cb = max_l1_size / (input_single_tile_size + output_single_tile_size);

    // TODO : currently multi_core parallelization on column only works for single tile height tensors.
    // Need to debug this to work on wide tensors that are higher than a single tile

    // If the input is interleaved and an entire row of tiles can't fit in a CB at once
    if (!input_is_sharded && num_tiles_per_row > max_tiles_per_cb) {
        // If the output is also interleaved and the tensor is only a single tile high, we can
        // parellize the work column wise. Otherwise we have to resort to the single core implementation,
        // as the current default multi core implementation processes an entire row of tiles at once.
        if (!output_is_sharded && num_tiles_per_col == 1) {
            return untilize_multi_core_parallelize_column(a, output, use_pack_untilize, fp32_dest_acc_en);
        } else {
            return untilize_single_core(a, output, use_pack_untilize, fp32_dest_acc_en);
        }
    }

    // Default values are for interleaved input.
    // Cliff core applicable interleaved input only, it is the only core not processing the
    // same number of rows (blocks) as all other cores.
    uint32_t num_input_blocks_across_width = 1;
    uint32_t num_tiles_per_input_block = num_tiles_per_row;
    uint32_t num_input_blocks_per_full_core = num_rows_per_full_core;
    uint32_t num_input_blocks_per_cliff_core = num_rows_per_cliff_core;
    if (input_is_sharded) {
        ShardSpec input_shard_spec = a.shard_spec().value();
        uint32_t input_shard_height = input_shard_spec.shape[0];
        uint32_t input_shard_width = input_shard_spec.shape[1];

        num_compute_cores = input_shard_spec.grid.num_cores();
        compute_core_range = input_shard_spec.grid;
        full_compute_core_range = input_shard_spec.grid;
        cliff_compute_core_range = CoreRangeSet();

        // Note: Accounting for uneven input shards
        num_input_blocks_across_width = tt::div_up(tensor_width, input_shard_width);
        num_tiles_per_input_block = input_shard_width / tile_width;
        num_input_blocks_per_full_core = input_shard_height / tile_height;
        num_input_blocks_per_cliff_core = 0;
    }

    // Input CB
    uint32_t input_cb_num_tiles;
    if (input_is_sharded) {
        // Have compute core untilize the entire shard at once
        input_cb_num_tiles = num_tiles_per_input_block * num_input_blocks_per_full_core;
    } else {
        if (num_input_blocks_per_full_core == 1) {
            // No need to double buffer if the core is only processing a single block
            input_cb_num_tiles = num_tiles_per_input_block;
        } else {
            // Double buffer if the core is processing 2+ blocks
            input_cb_num_tiles = num_tiles_per_input_block * 2;
        }
    }
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0,
        program,
        compute_core_range,
        input_single_tile_size,
        input_cb_num_tiles,
        input_cb_data_format,
        input_is_sharded ? src0_buffer : nullptr);

    // Output CB
    uint32_t output_cb_num_tiles;
    if (num_input_blocks_per_full_core == 1) {
        // No need to double buffer if the core is only processing a single block
        output_cb_num_tiles = num_tiles_per_input_block;
    } else {
        // Double buffer if the core is processing 2+ blocks
        output_cb_num_tiles = num_tiles_per_input_block * 2;
    }
    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        compute_core_range,
        output_single_tile_size,
        output_cb_num_tiles,
        output_cb_data_format);

    // Reader compile-time args and kernel
    KernelHandle unary_reader_kernel_id;
    if (input_is_sharded) {
        // Sharded input
        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index};
        unary_reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
            compute_core_range,
            tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    } else {
        // Interleaved input
        bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM;
        std::vector<uint32_t> reader_compile_time_args = {
            (uint32_t)src0_is_dram,
            (uint32_t)src0_cb_index,
        };
        unary_reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
            "reader_unary_start_id.cpp",
            compute_core_range,
            ReaderDataMovementConfig(reader_compile_time_args));
    }

    // Writer compute defines
    std::map<std::string, std::string> writer_compute_defines;
    if (output_is_sharded) {
        writer_compute_defines["SHARDED"] = "1";
    }

    // Writer compile-time args
    bool output_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    uint32_t output_num_blocks_across_width = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        uint32_t output_shard_width = output.shard_spec().value().shape[1];
        output_num_blocks_across_width = tensor_width / output_shard_width;
    }
    uint32_t output_stick_size = tensor_width * output.element_size() / output_num_blocks_across_width;
    bool output_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_stick_size);
    uint32_t output_log_base_2_of_page_size =
        output_stick_size_is_power_of_two ? (std::bit_width(output_stick_size) - 1) : 0;
    uint32_t output_element_size = output.element_size();
    uint32_t num_cols_per_input_block = num_tiles_per_input_block * tile_width;
    uint32_t num_cols_per_output_block = tensor_width / output_num_blocks_across_width;
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)output_is_dram,
        (uint32_t)output_cb_index,
        (uint32_t)output_stick_size,
        (uint32_t)output_stick_size_is_power_of_two,
        (uint32_t)output_log_base_2_of_page_size,
        (uint32_t)tile_height,
        (uint32_t)num_tiles_per_input_block,
        (uint32_t)output_num_blocks_across_width,
        (uint32_t)output_element_size,
        (uint32_t)num_cols_per_input_block,
        (uint32_t)num_cols_per_output_block,
    };
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output, writer_compile_time_args);
    }

    // Writer kernel
    KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multi_core.cpp",
        compute_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_compute_defines));

    // Compute kernel file
    std::string compute_kernel;
    if (num_tiles_per_input_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize || a.dtype() == DataType::UINT16) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel = std::string(
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp");
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
        compute_kernel = std::string(
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/"
            "pack_untilize_variable_num_blocks.cpp");
    }

    // Compute compile-time args and kernel
    // Note: This condition is always true for sharded input
    KernelHandle untilize_kernel_id = 0;
    if (full_compute_core_range.ranges().size() > 0) {
        std::vector<uint32_t> compute_compile_time_args = {
            (uint32_t)num_tiles_per_input_block, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};
        untilize_kernel_id = CreateKernel(
            program,
            compute_kernel,
            full_compute_core_range,
            ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_compile_time_args});
    }

    // Compute Cliff compile_time args and kernel
    // Note: This condition is always false for sharded input (sharded input will never have a cliff core)
    KernelHandle untilize_cliff_kernel_id = 0;
    if (cliff_compute_core_range.ranges().size() > 0) {
        std::vector<uint32_t> compute_compile_time_args_cliff = {
            (uint32_t)num_tiles_per_input_block, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};
        untilize_cliff_kernel_id = CreateKernel(
            program,
            compute_kernel,
            cliff_compute_core_range,
            ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_compile_time_args_cliff});
    }

    // Run-time arg assignment
    // Note: This variable is only applicable to interleaved input
    uint32_t tile_start_index = 0;

    // Run-time args (full cores)
    // Note: For sharded input, these are the only cores used
    bool is_row_major = input_is_sharded ? a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR : true;
    std::vector<CoreCoord> full_cores = corerange_to_cores(full_compute_core_range, std::nullopt, is_row_major);
    for (uint32_t i = 0; i < full_cores.size(); ++i) {
        CoreCoord core = full_cores[i];
        uint32_t height_wise_input_block_start_index =
            (i / num_input_blocks_across_width) * num_input_blocks_per_full_core;
        uint32_t width_wise_input_block_index = i % num_input_blocks_across_width;

        // Handle uneven input sharding width wise (writer run-time arg)
        uint32_t num_unpadded_cols_per_input_block = num_cols_per_input_block;
        if (input_is_sharded) {
            bool is_last_input_shard_in_row = width_wise_input_block_index == num_input_blocks_across_width - 1;
            if (is_last_input_shard_in_row) {
                uint32_t input_shard_width = a.shard_spec().value().shape[1];
                num_unpadded_cols_per_input_block =
                    num_cols_per_input_block - (tt::round_up(tensor_width, input_shard_width) - tensor_width);
            }
        }

        // Handle uneven input sharding height wise (reader, compute, writer run-time arg)
        uint32_t num_input_blocks_to_process = num_input_blocks_per_full_core;
        if (input_is_sharded) {
            uint32_t input_shard_height = a.shard_spec().value().shape[0];
            uint32_t height_wise_shard_index = i / num_input_blocks_across_width;
            uint32_t num_shards_height_wise = tt::div_up(tensor_height, input_shard_height);
            bool is_last_input_shard_in_col = height_wise_shard_index == num_shards_height_wise - 1;
            if (is_last_input_shard_in_col) {
                num_input_blocks_to_process =
                    num_input_blocks_per_full_core -
                    (tt::round_up(tensor_height, input_shard_height) - tensor_height) / tile_height;
            }
        }

        // Reader run-time args
        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        std::vector<uint32_t> reader_run_time_args;
        if (input_is_sharded) {
            // Sharded input
            reader_run_time_args = {num_tiles_to_read};
        } else {
            // Interleaved input
            reader_run_time_args = {
                src0_buffer->address(),
                num_tiles_to_read,
                tile_start_index,
            };
        }

        // Writer run-time args
        uint32_t input_block_global_col_index = width_wise_input_block_index * num_cols_per_input_block;
        uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
        uint32_t num_cols_already_processed_in_first_output_block =
            input_block_global_col_index % num_cols_per_output_block;
        std::vector<uint32_t> writer_run_time_args = {
            dst_buffer->address(),
            num_input_blocks_to_process,
            height_wise_input_block_start_index,
            num_unpadded_cols_per_input_block,
            width_wise_output_block_start_index,
            num_cols_already_processed_in_first_output_block};
        if (output_is_sharded) {
            shard_builder::extend_sharding_run_time_args(output, writer_run_time_args);
        }

        // Compute run-time args
        std::vector<uint32_t> compute_run_time_args = {num_input_blocks_to_process};

        // Set run-time arg
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, untilize_kernel_id, core, compute_run_time_args);

        // Update index of first tile to read
        tile_start_index += num_tiles_per_input_block * num_input_blocks_per_full_core;
    }

    // Run-time args (cliff core)
    // Note: Only applicable if input is interleaved (sharded input will never have a cliff core)
    std::vector<CoreCoord> cliff_cores = corerange_to_cores(cliff_compute_core_range, std::nullopt, is_row_major);
    if (cliff_cores.size() > 0) {
        // There should only ever be 0 or 1 cliff cores
        CoreCoord cliff_core = cliff_cores[0];
        uint32_t height_wise_input_block_start_index = full_cores.size() * num_input_blocks_per_full_core;
        uint32_t width_wise_input_block_index = 0;

        // Handle uneven input sharding width wise (writer run-time arg)
        // Note: Since cliff core is only applicable to interleaved input, this core
        // will never process an uneven shard (or any shard for that matter)
        uint32_t num_unpadded_cols_per_input_block = num_cols_per_input_block;

        // Handle uneven input sharding height wise (reader, compute, writer run-time arg)
        // Note: Since cliff core is only applicable to interleaved input, this core
        // will never process an uneven shard (or any shard for that matter)
        uint32_t num_input_blocks_to_process = num_input_blocks_per_cliff_core;

        // Writer run-time args
        uint32_t input_block_global_col_index = width_wise_input_block_index * num_cols_per_input_block;
        uint32_t width_wise_output_block_start_index = input_block_global_col_index / num_cols_per_output_block;
        uint32_t num_cols_already_processed_in_first_output_block =
            input_block_global_col_index % num_cols_per_output_block;
        std::vector<uint32_t> writer_run_time_args = {
            dst_buffer->address(),
            num_input_blocks_to_process,
            height_wise_input_block_start_index,
            num_unpadded_cols_per_input_block,
            width_wise_output_block_start_index,
            num_cols_already_processed_in_first_output_block};
        if (output_is_sharded) {
            shard_builder::extend_sharding_run_time_args(output, writer_run_time_args);
        }

        // Reader run-time args (always reading interleaved input as cliff core does not exist for sharded input)
        uint32_t num_tiles_to_read = num_tiles_per_input_block * num_input_blocks_to_process;
        std::vector<uint32_t> reader_run_time_args = {
            src0_buffer->address(),
            num_tiles_to_read,
            tile_start_index,
        };

        // Compute run-time args
        std::vector<uint32_t> compute_run_time_args = {num_input_blocks_to_process};

        // Set run-time args
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, cliff_core, reader_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, cliff_core, writer_run_time_args);
        tt::tt_metal::SetRuntimeArgs(program, untilize_cliff_kernel_id, cliff_core, compute_run_time_args);
    }

    std::vector<CoreCoord> cores_with_run_time_args;
    cores_with_run_time_args.reserve(full_cores.size() + cliff_cores.size());
    cores_with_run_time_args.insert(cores_with_run_time_args.end(), full_cores.begin(), full_cores.end());
    cores_with_run_time_args.insert(cores_with_run_time_args.end(), cliff_cores.begin(), cliff_cores.end());

    auto override_runtime_arguments_callback = [reader_kernel_id = unary_reader_kernel_id,
                                                writer_kernel_id = unary_writer_kernel_id,
                                                cb_src0 = cb_src0,
                                                cb_output = cb_output,
                                                cores_with_run_time_args](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        bool input_is_sharded = input_tensors.at(0).is_sharded();
        bool output_is_sharded = output_tensors.at(0).is_sharded();

        // Reader
        if (input_is_sharded) {
            // Sharded input
            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        } else {
            // Interleaved input
            auto& runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            for (const CoreCoord& core : cores_with_run_time_args) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer->address();
            }
        }

        // Writer
        auto& runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
        for (const CoreCoord& core : cores_with_run_time_args) {
            auto& runtime_args = runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks untilize_single_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    tt::tt_metal::Program program{};

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    tt::tt_metal::IDevice* device = a.device();
    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_volume = tile_height * tile_width;

    bool input_is_sharded = a.memory_config().is_sharded();
    bool output_is_sharded = output.memory_config().is_sharded();

    uint32_t num_tiles = a.physical_volume() / tile_volume;
    uint32_t num_blocks_across_height = a.physical_volume() / a.padded_shape()[-1] / tile_height;
    uint32_t num_columns_of_blocks = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        num_columns_of_blocks = a.padded_shape()[-1] / output.shard_spec().value().shape[1];
    }
    uint32_t num_tiles_per_column_row = a.padded_shape()[-1] / num_columns_of_blocks / tile_width;

    // Determine how much L1 space we can use for input and output CBs,
    // ensuring that we don't intrude into other L1 storage space
    uint32_t max_l1_size =
        a.device()->l1_size_per_core() / 2 - a.device()->allocator()->get_base_allocator_addr(HalMemType::L1);

    // Determine the max number of tiles that can be in any CB at a given time (1 input CB + 1 output CB = 2 total CBs)
    uint32_t max_tiles_per_cb = max_l1_size / (input_single_tile_size + output_single_tile_size);

    // Determine how many tiles each block will store.
    // Currently we require that the number of tiles in a row is divisible by the number of blocks in a row, or
    // equivalently the number of tiles in a row is divisible by the number of tiles in a block.
    uint32_t num_tiles_per_block = num_tiles_per_column_row;
    if (num_tiles_per_block > max_tiles_per_cb) {
        for (uint32_t i = max_tiles_per_cb; i > 0; --i) {
            if (num_tiles_per_column_row % i == 0) {
                num_tiles_per_block = i;
                break;
            }
        }
    }

    uint32_t num_blocks_per_column_row = num_tiles_per_column_row / num_tiles_per_block;
    uint32_t output_single_block_width_size = num_tiles_per_block * TILE_WIDTH * output.element_size();
    uint32_t num_total_sticks = a.physical_volume() / a.padded_shape()[-1] * num_columns_of_blocks;
    uint32_t output_stick_size = a.physical_volume() * output.element_size() / num_total_sticks;

    // Input CB
    uint32_t input_cb_num_tiles = num_tiles_per_block;
    auto [src0_cb_index, cb_src0] =
        create_cb(tt::CBIndex::c_0, program, core, input_single_tile_size, input_cb_num_tiles, input_cb_data_format);

    // Output CB
    uint32_t output_cb_num_tiles = num_tiles_per_block;
    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16, program, core, output_single_tile_size, output_cb_num_tiles, output_cb_data_format);

    // Reader compute defines
    std::map<std::string, std::string> reader_compute_defines;
    if (input_is_sharded) {
        reader_compute_defines["SHARDED"] = "1";
    }

    // Writer compute defines
    std::map<std::string, std::string> writer_compute_defines;
    if (output_is_sharded) {
        writer_compute_defines["SHARDED"] = "1";
    }

    // Reader compile-time args
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src0_is_dram,
        (uint32_t)src0_cb_index,
    };
    if (input_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(a, reader_compile_time_args);
    }

    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "reader_unary_start_id.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_compute_defines));

    // Writer compile-time args
    bool output_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(output_stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::bit_width(output_stick_size) - 1) : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)output_is_dram,
        (uint32_t)output_cb_index,
        (uint32_t)output_stick_size,
        (uint32_t)stick_size_is_power_of_two,
        (uint32_t)log2_stick_size,
        (uint32_t)tile_height,
        (uint32_t)num_blocks_across_height,
        (uint32_t)num_columns_of_blocks,
        (uint32_t)num_blocks_per_column_row,
        (uint32_t)num_tiles_per_block,
        (uint32_t)output_single_block_width_size,
    };
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output, writer_compile_time_args);
    }

    // Untilized writer
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_single_core.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_compute_defines));

    // Compute file path
    std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (num_tiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize || a.dtype() == DataType::UINT16) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel =
            std::string("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
    }

    // Compute compile-time args
    uint32_t num_blocks = num_columns_of_blocks * num_blocks_per_column_row * num_blocks_across_height;
    std::vector<uint32_t> compute_compile_time_args = {
        (uint32_t)num_blocks, (uint32_t)num_tiles_per_block, (uint32_t)src0_cb_index, (uint32_t)output_cb_index};

    // Compute kernel
    auto untilize_kernel_id = tt::tt_metal::CreateKernel(
        program,
        compute_kernel,
        core,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_compile_time_args});

    // Reader run-time args
    uint32_t start_page_id = 0;
    std::vector<uint32_t> reader_run_time_args = {
        src0_buffer->address(),
        num_tiles,
        start_page_id,
    };
    if (input_is_sharded) {
        shard_builder::extend_sharding_run_time_args(a, reader_run_time_args);
    }

    // Writer run-time args
    std::vector<uint32_t> writer_run_time_args = {dst_buffer->address()};
    if (output_is_sharded) {
        shard_builder::extend_sharding_run_time_args(output, writer_run_time_args);
    }

    // Set run-time args
    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_run_time_args);
    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_run_time_args);

    auto override_runtime_args_callback = [reader_kernel_id = unary_reader_kernel_id,
                                           writer_kernel_id = unary_writer_kernel_id](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::detail
