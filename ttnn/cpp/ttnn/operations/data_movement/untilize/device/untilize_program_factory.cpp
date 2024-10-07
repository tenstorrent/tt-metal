// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_program_factory.hpp"

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace ttnn::operations::data_movement::detail {


uint32_t get_largest_divisor(uint32_t dividend, uint32_t starting_divisor, uint32_t divisor_factor = 1) {
    for (uint32_t curr_div = starting_divisor; curr_div > 0; curr_div--) {
        if (dividend % (curr_div*divisor_factor) == 0) {
            return curr_div;
        }
    }
    return 1;
}

operation::ProgramWithCallbacks untilize_multi_core_parallelize_column(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {

    auto program = tt::tt_metal::CreateProgram();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    Device* device = a.device();

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t ntiles = a.volume() / TILE_HW;
    uint32_t ncores_x = grid_size.x;
    //uint32_t ncores_x = 2;

    ncores_x = get_largest_divisor(ntiles, ncores_x);
    uint32_t ncores_y = grid_size.y;
    //uint32_t ncores_y = 1;
    ncores_y = get_largest_divisor(ntiles, ncores_y, ncores_x);

    TT_ASSERT(ntiles % (ncores_x * ncores_y) == 0);
    uint32_t ntiles_per_block = ntiles/(ncores_x * ncores_y);

    //TODO increase block size to increase untilize performance, currently each untilize block is a single tile
    //uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 - a.device()->get_base_allocator_addr(HalMemType::L1);
    //uint32_t max_tiles = (max_l1_size / (input_single_tile_size + output_single_tile_size))/2;  // 2 CBs, double buffering each
    uint32_t max_tiles = 1;



    uint32_t stick_s = a.get_legacy_shape()[-1];
    uint32_t ntiles_per_row = stick_s / TILE_WIDTH;
    uint32_t stick_size = stick_s * output.element_size();
    uint32_t ntiles_per_column = ntiles / ntiles_per_row;
    uint32_t starting_tile = ntiles_per_block;
    if (ntiles_per_row > max_tiles) {
        starting_tile = max_tiles;
    }
    ntiles_per_block = get_largest_divisor(ntiles_per_row, starting_tile);
    TT_ASSERT(ntiles_per_row % ntiles_per_block == 0 and ntiles_per_block >= 1 and ntiles_per_block <= ntiles_per_row and ntiles % ntiles_per_block == 0);

    uint32_t nblocks = (ntiles / ntiles_per_block);
    uint32_t block_size_nbytes =  input_single_tile_size;

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
        tt::CB::c_in0,
        program,
        all_cores,
        input_single_tile_size,
        num_input_tiles,
        input_cb_data_format,
        nullptr);

    uint32_t num_output_tiles =  ntiles_per_block * 2;
    auto [output_cb_index, cb_output] = create_cb(
        tt::CB::c_out0,
        program,
        all_cores,
        output_single_tile_size,
        num_output_tiles,
        output_cb_data_format,
         nullptr);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_ct_args = {(uint32_t)src0_is_dram};

    auto unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    bool out_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
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
    vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
    };
    vector<uint32_t> compute_args_cliff = {
        (uint32_t)nblocks_per_core_cliff ,
        (uint32_t)ntiles_per_block,  // per_block_ntiles
    };

    std::string compute_kernel("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel = std::string("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
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
        if (!full_cores.core_coord_in_core_ranges(core)) {
            continue;
        }
        // reader runtime args
        vector<uint32_t> reader_rt_args;
        auto ntiles_per_core = ntiles_per_block * nblocks_per_core;
        reader_rt_args = {
            src0_buffer->address(),               // src_addr
            ntiles_per_core,  // ntiles
            tile_start_id                         // start_id
        };

        std::vector<uint32_t> writer_rt_args;
        writer_rt_args = {
            dst_buffer->address(),           // dst_addr
            nsticks_per_core,  // nsticks
            stick_size,               // block_size_nbytes
            ntiles_per_core,                // ntiles_per_core
            TILE_WIDTH * output.element_size(),               // tile_width_size
            0, //start stick id = 0, since parallelizing on height
            offset_within_stick
        };

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
        std::vector<uint32_t> reader_rt_args;
        auto ntiles_per_core_cliff = ntiles_per_block * nblocks_per_core_cliff;
        reader_rt_args = {
            src0_buffer->address(),               // src_addr
            ntiles_per_core_cliff,  // ntiles
            tile_start_id                         // start_id
        };

        std::vector<uint32_t> writer_rt_args;
        writer_rt_args = {
            dst_buffer->address(),           // dst_addr
            nsticks_per_core,  // nsticks
            stick_size,               // block_size_nbytes
            ntiles_per_core_cliff,                // ntiles_per_core
            TILE_WIDTH * output.element_size(),               // tile_width_size
            0, //start stick id = 0, since parallelizing on height
            offset_within_stick
        };
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
    }

    auto override_runtime_arguments_callback = [reader_kernel_id = unary_reader_kernel_id,
                                                writer_kernel_id = unary_writer_kernel_id,
                                                cb_src0 = cb_src0,
                                                cb_output = cb_output,
                                                cores_with_rtargs](
                                                   const void* operation,
                                                   ProgramHandle program,
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



operation::ProgramWithCallbacks untilize_multi_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    auto program = tt::tt_metal::CreateProgram();

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    Device* device = a.device();

    uint32_t ntiles = a.volume() / TILE_HW;
    uint32_t stick_s = a.get_legacy_shape()[-1];
    uint32_t ntiles_per_block = a.get_legacy_shape()[-1] / TILE_WIDTH;
    uint32_t nblocks = ceil((float)ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = a.get_legacy_shape()[-1] * output.element_size();

    uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 - a.device()->get_base_allocator_addr(HalMemType::L1);
    uint32_t max_tiles = (max_l1_size / (input_single_tile_size + output_single_tile_size));  // 2 CBs, double buffering each

    // TODO : currently multi_core parallelization on column only works for single tile height tensors.
    // Need to debug this to work on wide tensors that are higher than a single tile
    if (ntiles_per_block > max_tiles) {
        if(!src_sharded and !out_sharded) {
            uint32_t ntiles_height = ntiles / ntiles_per_block;
            if(ntiles_height == 1) {
                return untilize_multi_core_parallelize_column(a, output, use_pack_untilize, fp32_dest_acc_en);
            }
            else {
                return untilize_single_core(a, output, use_pack_untilize, fp32_dest_acc_en);
            }
        }
    }
    auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, nblocks);
    uint32_t ncores_x = grid_size.x;
    uint32_t ncores_y = std::ceil(static_cast<float>(ncores) / ncores_x);

    bool row_major = true;
    bool src_block_sharded = false;
    uint32_t num_rows_block = 0, block_row_size = 0, output_row_size = 0, last_block_row_size_unpadded = 0,
             num_output_rows_unpadded = 0;
    CoreCoord end_core;
    std::vector<CoreCoord> cores_with_rtargs;

    if (src_sharded) {
        auto shard_spec = a.shard_spec().value();
        src_block_sharded = a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
        row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
        ncores_y = device->compute_with_storage_grid_size().y;
        all_cores = shard_spec.grid;
        uint32_t num_cores = all_cores.num_cores();
        ncores = num_cores;
        core_range = all_cores;
        core_range_cliff = CoreRangeSet({});
        ntiles_per_block = shard_spec.shape[1] / TILE_WIDTH;
        nblocks_per_core = shard_spec.shape[0] / TILE_HEIGHT;
        nblocks_per_core_cliff = 0;

        num_rows_block = shard_spec.shape[0];
        block_row_size = shard_spec.shape[1] * output.element_size();  // in0_block_w * TILE_WIDTH * dtype_nbytes
        output_row_size = output.get_legacy_shape()[-1] * output.element_size();  // output row size bytes
        last_block_row_size_unpadded =
            block_row_size -
            (tt::round_up(output.get_legacy_shape()[-1], shard_spec.shape[1]) - output.get_legacy_shape()[-1]) *
                output.element_size();
        uint32_t num_output_rows = output.volume() / output.get_legacy_shape()[-1];
        num_output_rows_unpadded =
            num_rows_block - (tt::round_up(num_output_rows, shard_spec.shape[0]) - num_output_rows);
        end_core = (*shard_spec.grid.ranges().begin()).end_coord;
    }

    uint32_t num_input_tiles = src_sharded ? ntiles_per_block * nblocks_per_core : ntiles_per_block * 2;
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CB::c_in0,
        program,
        all_cores,
        input_single_tile_size,
        num_input_tiles,
        input_cb_data_format,
        src_sharded ? a.buffer() : nullptr);

    uint32_t num_output_tiles = out_sharded ? ntiles_per_block * nblocks_per_core : ntiles_per_block * 2;
    auto [output_cb_index, cb_output] = create_cb(
        tt::CB::c_out0,
        program,
        all_cores,
        output_single_tile_size,
        num_output_tiles,
        output_cb_data_format,
        out_sharded ? output.buffer() : nullptr);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    KernelHandle unary_reader_kernel_id;

    if (src_sharded) {
        std::vector<uint32_t> reader_ct_args = {(std::uint32_t)src0_cb_index};

        unary_reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    } else {
        bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
        vector<uint32_t> reader_ct_args = {(uint32_t)src0_is_dram};

        unary_reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
            all_cores,
            ReaderDataMovementConfig(reader_ct_args));
    }

    /** writer
     */
    KernelHandle unary_writer_kernel_id;
    if (out_sharded) {
        std::vector<uint32_t> writer_ct_args = {(std::uint32_t)output_cb_index};
        unary_writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    } else {
        bool out_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
        if (src_block_sharded) {
            vector<uint32_t> writer_ct_args = {
                (uint32_t)out_is_dram, (uint32_t)(input_cb_data_format == tt::DataFormat::Float32)};
            unary_writer_kernel_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp",
                all_cores,
                WriterDataMovementConfig(writer_ct_args));
        } else {
            bool stick_size_is_power_of_two = is_power_of_two_at_least_32(block_size_nbytes);
            uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(block_size_nbytes) : 0;
            vector<uint32_t> writer_ct_args = {
                (uint32_t)out_is_dram,
                (uint32_t)stick_size_is_power_of_two,
                (uint32_t)log2_stick_size,
            };

            unary_writer_kernel_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
                "writer_unary_stick_layout_split_rows_interleaved.cpp",
                all_cores,
                WriterDataMovementConfig(writer_ct_args));
        }
    }

    /** compute
     */
    vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
    };
    vector<uint32_t> compute_args_cliff = {
        (uint32_t)nblocks_per_core_cliff,
        (uint32_t)ntiles_per_block,  // per_block_ntiles
    };

    std::string compute_kernel("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel = std::string("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
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

    // 1D distribution of blocks across all cores
    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        // unequal case with cliff
        ncores_full -= 1;
        full_cores = core_range;
    }
    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);
    for (uint32_t i = 0; i < cores.size(); i++) {
        CoreCoord core = cores[i];
        if (!full_cores.core_coord_in_core_ranges(core)) {
            continue;
        }
        // reader runtime args
        vector<uint32_t> reader_rt_args;

        if (src_sharded) {
            reader_rt_args = {
                ntiles_per_block * nblocks_per_core  // ntiles
            };
        } else {
            reader_rt_args = {
                src0_buffer->address(),               // src_addr
                ntiles_per_block * nblocks_per_core,  // ntiles
                tile_start_id                         // start_id
            };
        }
        // log_debug("reader[{}]: {},{} = {} ({})", src0_buffer->address(), core.x, core.y, tile_start_id,
        // ntiles_per_block * nblocks_per_core);

        // writer runtime args
        vector<uint32_t> writer_rt_args;
        if (out_sharded) {
            writer_rt_args = {
                ntiles_per_block * nblocks_per_core  // ntiles
            };
        } else {
            if (src_block_sharded) {
                uint32_t block_start_row_offset;
                uint32_t block_start_row_id_offset;
                uint32_t row_size_unpadded = block_row_size;
                uint32_t num_rows_unpadded = num_rows_block;
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }

                writer_rt_args = {
                    dst_buffer->address(),  // dst_addr
                    num_rows_block,
                    block_row_size,
                    1,
                    1,
                    1,
                    output_row_size,
                    row_size_unpadded,
                    num_rows_unpadded,
                    block_start_row_id_offset,
                    block_start_row_offset};
            } else {
                writer_rt_args = {
                    dst_buffer->address(),           // dst_addr
                    nblocks_per_core * TILE_HEIGHT,  // nblocks per core
                    block_size_nbytes,               // block_size_nbytes
                    ntiles_per_block,                // ntiles_per_block
                    block_size_nbytes,               // block_size_nbytes
                    1,                               // full blocks in a row
                    0,
                    0,
                    row_start_id};
            }
        }
        // log_debug("writer[{}]: {},{} = {} {}", dst_buffer->address(), core.x, core.y, block_size_nbytes,
        // row_start_id);

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);

        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
        cores_with_rtargs.push_back(core);
        tile_start_id += ntiles_per_block * nblocks_per_core;
        row_start_id += TILE_HEIGHT * nblocks_per_core;
    }
    if (ncores_full < ncores) {
        // last core is the cliff core with nblocks_per_core_cliff blocks
        CoreCoord core = row_major ? CoreCoord{ncores_full % ncores_x, ncores_full / ncores_x}
                                   : CoreCoord{ncores_full / ncores_y, ncores_full % ncores_y};
        // reader runtime args
        vector<uint32_t> reader_rt_args;

        if (src_sharded) {
            reader_rt_args = {
                ntiles_per_block * nblocks_per_core_cliff  // ntiles
            };
        } else {
            reader_rt_args = {
                src0_buffer->address(),                               // src_addr
                (uint32_t)ntiles_per_block * nblocks_per_core_cliff,  // ntiles
                tile_start_id                                         // start_id
            };
        }
        // log_debug("reader: {},{} = {} ({})", core.x, core.y, tile_start_id, ntiles_per_block *
        // nblocks_per_core_cliff);

        // writer runtime args
        vector<uint32_t> writer_rt_args;
        if (out_sharded) {
            writer_rt_args = {
                ntiles_per_block * nblocks_per_core_cliff  // ntiles
            };
        } else {
            if (src_block_sharded) {
                uint32_t block_start_row_offset;
                uint32_t block_start_row_id_offset;
                uint32_t row_size_unpadded = block_row_size;
                uint32_t num_rows_unpadded = num_rows_block;
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }
                writer_rt_args = {
                    dst_buffer->address(),  // dst_addr
                    num_rows_block,
                    block_row_size,
                    1,
                    1,
                    1,
                    output_row_size,
                    row_size_unpadded,
                    num_rows_unpadded,
                    block_start_row_id_offset,
                    block_start_row_offset};
            } else {
                writer_rt_args = {
                    dst_buffer->address(),                 // dst_addr
                    nblocks_per_core_cliff * TILE_HEIGHT,  // nsticks
                    block_size_nbytes,                     // stick_size_nbytes
                    ntiles_per_block,                      // ntiles_per_block
                    block_size_nbytes,                     // block_width_nbytes
                    1,                                     // full blocks in a row
                    0,                                     // UNUSED
                    0,                                     // UNUSED
                    row_start_id};
            }
        }
        // log_debug("writer: {},{} = {} {}", core.x, core.y, block_size_nbytes, row_start_id);

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);

        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
        cores_with_rtargs.push_back(core);
    }
    auto override_runtime_arguments_callback = [reader_kernel_id = unary_reader_kernel_id,
                                                writer_kernel_id = unary_writer_kernel_id,
                                                cb_src0 = cb_src0,
                                                cb_output = cb_output,
                                                cores_with_rtargs](
                                                   const void* operation,
                                                   ProgramHandle program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        bool src_sharded = input_tensors.at(0).memory_config().is_sharded();
        bool out_sharded = output_tensors.at(0).memory_config().is_sharded();

        if (src_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        } else {
            auto& runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            for (const CoreCoord& core : cores_with_rtargs) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer->address();
            }
        }

        if (out_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
        } else {
            auto& runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (const CoreCoord& core : cores_with_rtargs) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks untilize_single_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    auto program = tt::tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;

    uint32_t num_sticks = a.volume() / a.get_legacy_shape()[-1];
    uint32_t stick_size = a.get_legacy_shape()[-1] * output.element_size();

    uint32_t stick_s = a.get_legacy_shape()[-1];
    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 - a.device()->get_base_allocator_addr(HalMemType::L1);
    uint32_t max_tiles = max_l1_size / (input_single_tile_size + output_single_tile_size);  // 2 CBs
    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for (uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }
    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * output.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;
    uint32_t num_leftover_tiles = num_tiles_in_row % num_tiles_per_block;
    uint32_t leftover_width_in_row = num_leftover_tiles * output.element_size();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device* device = a.device();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    auto cb_src0_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * input_single_tile_size, {{src0_cb_index, input_cb_data_format}})
                              .set_page_size(src0_cb_index, input_single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    auto cb_output_config = tt::tt_metal::CircularBufferConfig(
                                num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
                                .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Writer compile-time args
    vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        num_sticks,
        stick_size,
        num_tiles_per_block,
        block_width_size,
        num_full_blocks_in_row,
        num_leftover_tiles,
        leftover_width_in_row,
        0};

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_is_dram};

    bool out_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::bit_width(stick_size) - 1): 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)out_is_dram,
        (std::uint32_t)stick_size_is_power_of_two,
        (std::uint32_t)log2_stick_size,
    };

    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Untilized writer
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_interleaved.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles / num_tiles_per_block),  // per_core_block_cnt
        uint32_t(num_tiles_per_block)               // per_core_block_tile_cnt
    };

    std::string compute_kernel("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (num_tiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel = std::string("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
    }

    auto untilize_kernel_id = tt::tt_metal::CreateKernel(
        program,
        compute_kernel,
        core,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args});

    tt::tt_metal::SetRuntimeArgs(
        program, unary_reader_kernel_id, core, {src0_buffer->address(), uint32_t(num_tiles), 0});

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);

    auto override_runtime_args_callback = [reader_kernel_id = unary_reader_kernel_id,
                                           writer_kernel_id = unary_writer_kernel_id](
                                              const ProgramHandle program,
                                              const std::vector<Buffer*>& input_buffers,
                                              const std::vector<Buffer*>& output_buffers) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

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

    return {program, override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::detail
