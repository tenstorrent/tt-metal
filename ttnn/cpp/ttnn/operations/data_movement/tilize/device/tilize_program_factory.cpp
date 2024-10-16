// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks tilize_single_core(const Tensor& a, Tensor& output) {
    tt::tt_metal::Program program{};

    CoreRange core({0, 0}, {0, 0});

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device* device = a.device();
    auto output_shape = output.get_legacy_shape();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t num_tiles = a.volume() / TILE_HW;

    auto width = a.get_legacy_shape()[-1];
    uint32_t stick_s = width;
    uint32_t num_sticks = a.volume() / width;
    uint32_t stick_size = stick_s * a.element_size();  // Assuming bfloat16 dataformat

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
    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;
    uint32_t num_leftover_tiles = num_tiles_in_row % num_tiles_per_block;
    uint32_t leftover_width_in_row = num_leftover_tiles * a.element_size();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;

    auto src0_cb_config = tt::tt_metal::CircularBufferConfig(num_input_tiles * input_single_tile_size,
                                                             {{src0_cb_index, input_cb_data_format}})
                              .set_page_size(src0_cb_index, input_single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, src0_cb_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    auto cb_output_config = tt::tt_metal::CircularBufferConfig(num_output_tiles * output_single_tile_size,
                                                               {{output_cb_index, output_cb_data_format}})
                                .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        num_sticks,
        stick_size,
        num_tiles_per_block,
        block_width_size,
        num_full_blocks_in_row,
        num_leftover_tiles,
        leftover_width_in_row,
        0  // row_start_id
    };

    // Reader compile-time args
    uint32_t src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> reader_compile_time_args = {src0_is_dram, stick_size_is_power_of_two, log2_stick_size};

    uint32_t out_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index, out_is_dram};

    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id =
        tt::tt_metal::CreateKernel(program,
                                   "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                   "reader_unary_stick_layout_split_rows_interleaved.cpp",
                                   core,
                                   tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Tilized writer
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_args = {
        num_tiles / num_tiles_per_block,  // per_core_block_cnt
        num_tiles_per_block               // per_core_block_tile_cnt
    };

    auto tilize_kernel_id = tt::tt_metal::CreateKernel(program,
                                                       "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
                                                       core,
                                                       tt::tt_metal::ComputeConfig{.compile_args = compute_args});

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles, 0});

    auto override_runtime_args_callback = [reader_kernel_id = unary_reader_kernel_id,
                                           writer_kernel_id = unary_writer_kernel_id](
                                              const Program& program,
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

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks tilize_multi_core_interleaved(const Tensor& a, Tensor& output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    int32_t ntiles = a.volume() / TILE_HW;
    uint32_t ntiles_per_block = a.get_legacy_shape()[-1] / TILE_WIDTH;
    uint32_t nblocks = std::ceil((float)ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = a.get_legacy_shape()[-1] * a.element_size();

    Device* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, nblocks);

    create_cb(tt::CB::c_in0, program, all_cores, input_single_tile_size, ntiles_per_block, input_cb_data_format);

    auto [output_cb_index, _] =
        create_cb(tt::CB::c_out0, program, all_cores, output_single_tile_size, ntiles_per_block, output_cb_data_format);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    uint32_t src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    uint32_t stick_size_is_power_of_two = is_power_of_two_at_least_32(block_size_nbytes);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (uint32_t)std::log2(block_size_nbytes) : 0;
    std::vector<uint32_t> reader_ct_args = {src0_is_dram, stick_size_is_power_of_two, log2_stick_size};
    KernelHandle unary_reader_kernel_id =
        CreateKernel(program,
                     "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                     "reader_unary_stick_layout_split_rows_interleaved.cpp",
                     all_cores,
                     ReaderDataMovementConfig(reader_ct_args));

    /** writer
     */
    uint32_t out_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_ct_args = {output_cb_index, out_is_dram};
    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    /** compute
     */
    vector<uint32_t> compute_args = {nblocks_per_core, ntiles_per_block};
    vector<uint32_t> compute_args_cliff = {nblocks_per_core_cliff, ntiles_per_block};

    if (core_range.ranges().size() > 0) {
        auto tilize_kernel_id = CreateKernel(program,
                                             "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
                                             core_range,
                                             ComputeConfig{.compile_args = compute_args});
    }
    if (core_range_cliff.size() > 0) {
        auto tilize_cliff_kernel_id = CreateKernel(program,
                                                   "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
                                                   core_range_cliff,
                                                   ComputeConfig{.compile_args = compute_args_cliff});
    }

    // 1D distribution of blocks across cores
    bool has_cliff = core_range_cliff.size() > 0;

    uint32_t ncores_full = ncores - has_cliff;
    uint32_t ncores_x = grid_size.x;
    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];

        // reader runtime args
        vector<uint32_t> reader_rt_args = {src0_buffer->address(),
                                           nblocks_per_core * TILE_HEIGHT,
                                           block_size_nbytes,
                                           ntiles_per_block,
                                           block_size_nbytes,
                                           1,  // full blocks in row
                                           0,  // num leftover tiles
                                           0,  // leftover width in row
                                           row_start_id};

        // writer runtime args
        vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            ntiles_per_block * nblocks_per_core,  // ntiles per core
            tile_start_id                         // start id
        };

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        tile_start_id += ntiles_per_block * nblocks_per_core;
        row_start_id += TILE_HEIGHT * nblocks_per_core;
    }
    if (has_cliff) {
        // the last core is a cliff core with nblocks_per_core_cliff blocks
        const CoreCoord& core = cores.back();

        // reader runtime args
        vector<uint32_t> reader_rt_args = {src0_buffer->address(),
                                           nblocks_per_core_cliff * TILE_HEIGHT,
                                           block_size_nbytes,
                                           ntiles_per_block,
                                           block_size_nbytes,
                                           1,  // full blocks in row
                                           0,  // num leftover tiles
                                           0,  // leftover width in row
                                           row_start_id};

        // writer runtime args
        vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            ntiles_per_block * nblocks_per_core_cliff,  // ntiles per core
            tile_start_id                               // start id
        };

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
    }

    auto override_runtime_args_callback = [reader_kernel_id = unary_reader_kernel_id,
                                           writer_kernel_id = unary_writer_kernel_id,
                                           cores = cores](const Program& program,
                                                          const std::vector<Buffer*>& input_buffers,
                                                          const std::vector<Buffer*>& output_buffers) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

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

operation::ProgramWithCallbacks tilize_multi_core_sharded(const Tensor& input, Tensor& output) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t num_tiles = input.volume() / TILE_HW;

    tt::tt_metal::Device* device = input.device();

    auto shard_spec = input.shard_spec().value();
    uint32_t num_tiles_per_shard = shard_spec.shape[0] * shard_spec.shape[1] / TILE_HW;
    uint32_t num_tiles_per_row = shard_spec.shape[1] / TILE_WIDTH;
    auto all_cores = shard_spec.grid;
    uint32_t num_cores_x = device->compute_with_storage_grid_size().x;
    uint32_t num_cores = all_cores.num_cores();

    auto [src0_cb_index, cb_src0] = create_cb(tt::CB::c_in0,
                                              program,
                                              all_cores,
                                              input_single_tile_size,
                                              num_tiles_per_shard,
                                              input_cb_data_format,
                                              input.buffer());

    auto [output_cb_index, cb_output] = create_cb(tt::CB::c_out0,
                                                  program,
                                                  all_cores,
                                                  output_single_tile_size,
                                                  num_tiles_per_shard,
                                                  output_cb_data_format,
                                                  output.buffer());

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index};

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_args = {uint32_t(num_tiles_per_shard / num_tiles_per_row), uint32_t(num_tiles_per_row)};

    auto untilize_kernel_id = tt::tt_metal::CreateKernel(program,
                                                         "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
                                                         all_cores,
                                                         tt::tt_metal::ComputeConfig{.compile_args = compute_args});

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, {num_tiles_per_shard});

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, all_cores, {num_tiles_per_shard});

    auto override_runtime_arguments_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cb_output](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors.at(0).buffer();

            auto dst_buffer = output_tensors.at(0).buffer();

            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);

            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks tilize_multi_core(const Tensor& a, Tensor& output) {
    if (a.memory_config().is_sharded()) {
        return tilize_multi_core_sharded(a, output);
    } else {
        return tilize_multi_core_interleaved(a, output);
    }
}

}  // namespace ttnn::operations::data_movement::detail
