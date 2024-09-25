// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <math.h>

#include "ttnn/deprecated/tt_dnn/op_library/cb_utils.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks tilize_with_val_padding_single_core(
    const Tensor& a, Tensor& output, const float pad_value) {
    auto output_shape = output.get_shape().with_tile_padding();

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device* device = a.device();

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    int32_t num_tiles = output.volume() / TILE_HW;

    auto true_input_shape = a.get_shape().with_tile_padding();
    auto true_output_shape = output.get_shape().with_tile_padding();

    auto input_w = true_input_shape.rank() >= 4 ? true_input_shape[-4] : 1;
    auto input_z = true_input_shape.rank() >= 3 ? true_input_shape[-3] : 1;
    auto input_y = true_input_shape.rank() >= 2 ? true_input_shape[-2] : 1;
    auto input_x = true_input_shape[-1];

    auto output_w = true_output_shape.rank() >= 4 ? true_output_shape[-4] : 1;
    auto output_z = true_output_shape.rank() >= 3 ? true_output_shape[-3] : 1;
    auto output_y = true_output_shape.rank() >= 2 ? true_output_shape[-2] : 1;
    auto output_x = true_output_shape[-1];

    uint32_t unpadded_row_size_bytes = input_x * a.element_size();  // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output_x * a.element_size();   // Assuming bfloat16 dataformat

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = output_x / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 - L1_UNRESERVED_BASE;
    // Memory usage is 2 CBs of width W, plus buffer of size alignment + (W * datum size)
    uint32_t max_X = (max_l1_size - alignment) / (a.element_size() * TILE_HEIGHT * 2 + a.element_size());
    uint32_t max_tiles = max_X / TILE_WIDTH;

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

    uint32_t block_width = num_tiles_per_block * TILE_WIDTH;
    uint32_t block_row_size = block_width * a.element_size();
    uint32_t num_blocks_w_output = padded_row_size_bytes / block_row_size;
    uint32_t num_blocks_w_input = unpadded_row_size_bytes / block_row_size;

    // Leftover size if input is not divisible by block size
    uint32_t block_row_leftover_size = unpadded_row_size_bytes - num_blocks_w_input * block_row_size;

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_output - num_blocks_w_input - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (output_y - input_y) / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_Z_diff_blocks = (output_z - input_z) * output_y / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_W_diff_blocks =
        (output_w - input_w) * output_z * output_y / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t num_leftover_Y = input_y - input_y / TILE_HEIGHT * TILE_HEIGHT;

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    assert(num_input_tiles > 0);
    tt::tt_metal::CircularBufferConfig src0_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * input_single_tile_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, input_single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, src0_cb_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        input_w,
        padded_W_diff_blocks,
        input_z,
        padded_Z_diff_blocks,
        input_y,
        padded_Y_diff_blocks,
        num_leftover_Y,
        input_x,
        unpadded_row_size_bytes,
        padded_row_size_bytes,
        packed_pad_value,
        num_blocks_w_input,
        num_blocks_w_output,
        num_blocks_w_diff,
        block_row_size,
        block_row_leftover_size};

    // Reader compile-time args
    uint32_t src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_row_size_bytes;
    uint32_t stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> reader_compile_time_args = {src0_is_dram, stick_size_is_power_of_two, log2_stick_size};

    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_dims_split_rows.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Tilized writer
    uint32_t out_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig({output_cb_index, out_is_dram}));

    vector<uint32_t> compute_kernel_args = {uint32_t(num_tiles / num_tiles_per_block), uint32_t(num_tiles_per_block)};

    auto tilize_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);

    tt::tt_metal::SetRuntimeArgs(
        program, unary_writer_kernel_id, core, {dst_buffer->address(), (uint32_t)num_tiles, 0});

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

operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_interleaved(
    const Tensor& a, Tensor& output, const float pad_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    Device* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();

    uint32_t num_blocks = output.volume() / output.get_shape().with_tile_padding()[-1] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = output.get_shape().with_tile_padding()[-1] / TILE_WIDTH;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, num_blocks);

    bool has_cliff = core_range_cliff.size() > 0;

    uint32_t unpadded_row_size_bytes = a.get_shape().with_tile_padding()[-1] * a.element_size();     // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output.get_shape().with_tile_padding()[-1] * a.element_size();  // Assuming bfloat16 dataformat

    auto [src0_cb_index, cb_src0] =
        create_cb(tt::CB::c_in0, program, all_cores, input_single_tile_size, num_tiles_per_row, input_cb_data_format);

    auto [output_cb_index, cb_output] = create_cb(
        tt::CB::c_out0, program, all_cores, output_single_tile_size, num_tiles_per_row, output_cb_data_format);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    uint32_t src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_row_size_bytes;
    uint32_t stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(stick_size) : 0;

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_dims_split_rows_multicore.cpp",
        all_cores,
        ReaderDataMovementConfig({src0_is_dram, stick_size_is_power_of_two, log2_stick_size}));

    /** writer
     */
    uint32_t out_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig({output_cb_index, out_is_dram}));

    /** compute
     */
    if (core_range.size() > 0) {
        auto tilize_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
            core_range,
            ComputeConfig{.compile_args = {nblocks_per_core, num_tiles_per_row}});
    }
    if (has_cliff) {
        auto tilize_cliff_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
            core_range_cliff,
            ComputeConfig{.compile_args = {nblocks_per_core_cliff, num_tiles_per_row}});
    }

    /* RUNTIME ARGS */

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    // 1D distribution of blocks across cores
    auto core_assignments = ttnn::distribute_work(
        output.get_shape(),
        output.get_shape().with_tile_padding().padding(),
        ncores,
        nblocks_per_core,
        has_cliff,
        nblocks_per_core_cliff);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    uint32_t ncores_x = grid_size.x;

    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // reader runtime args
        vector<uint32_t> reader_rt_args = {
            src0_buffer->address(),
            unpadded_row_size_bytes,
            padded_row_size_bytes,
            packed_pad_value,
            row_start_id,
            static_cast<unsigned int>(assignment.size()),
        };

        uint32_t nblocks_per_core = 0;

        for (const auto& el : assignment) {
            nblocks_per_core += el.block_count();
            row_start_id += el.data_row_count();
            reader_rt_args.push_back(el.n_data);
            reader_rt_args.push_back(el.n_mixed);
            reader_rt_args.push_back(el.n_pads);
            reader_rt_args.push_back(el.times);
        }

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core;

        // writer runtime args
        vector<uint32_t> writer_rt_args = {dst_buffer->address(), num_tiles_per_core, tile_start_id};

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        tile_start_id += num_tiles_per_core;
    }

    auto override_runtime_args_callback =
        [reader_kernel_id = unary_reader_kernel_id, writer_kernel_id = unary_writer_kernel_id, cores = cores](
            const Program& program,
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

// This purely supports input width shard -> output width shard for now
operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_sharded(
    const Tensor& a, Tensor& output, const float pad_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    Device* device = a.device();

    auto input_shard_spec = a.shard_spec().value();
    auto output_shard_spec = output.shard_spec().value();

    auto all_cores = output_shard_spec.grid;

    uint32_t num_batches = output.volume() / (output.get_shape().with_tile_padding()[-2] * output.get_shape().with_tile_padding()[-1]);

    uint32_t num_input_rows = input_shard_spec.shape[0];
    uint32_t input_shard_width_bytes = input_shard_spec.shape[1] * a.element_size();
    uint32_t ntiles_per_core = output_shard_spec.shape[0] * output_shard_spec.shape[1] / TILE_HW;
    uint32_t ntiles_per_batch = ntiles_per_core / num_batches;
    uint32_t ntiles_per_block = output_shard_spec.shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = output_shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t num_padded_rows = output.get_shape().with_tile_padding()[-2] - a.get_shape().with_tile_padding()[-2];

    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CB::c_in1,
        program,
        all_cores,
        input_shard_width_bytes,
        num_input_rows,
        input_cb_data_format,
        src_sharded ? a.buffer() : nullptr);

    auto [src1_cb_index, cb_src1] = create_cb(
        tt::CB::c_in0, program, all_cores, input_single_tile_size, ntiles_per_batch * 2, input_cb_data_format);

    auto [src2_cb_index, cb_src2] =
        create_cb(tt::CB::c_in2, program, all_cores, input_shard_width_bytes, 1, input_cb_data_format);

    auto [output_cb_index, cb_output] = create_cb(
        tt::CB::c_out0,
        program,
        all_cores,
        output_single_tile_size,
        ntiles_per_core,
        output_cb_data_format,
        out_sharded ? output.buffer() : nullptr);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    KernelHandle unary_reader_kernel_id;
    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
    };

    unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_height_width_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    /** writer
     */
    KernelHandle unary_writer_kernel_id;
    bool out_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    vector<uint32_t> writer_ct_args = {
        output_cb_index,
    };
    unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    /** compute
     */
    vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
    };

    auto tilize_kernel_id = CreateKernel(
        program, "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp", all_cores, ComputeConfig{.compile_args = compute_args});

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    vector<uint32_t> reader_rt_args = {
        num_input_rows,
        input_shard_width_bytes,
        (num_input_rows / num_batches) * input_shard_width_bytes,
        ntiles_per_batch,
        num_padded_rows,
        num_batches,
        packed_pad_value};
    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, reader_rt_args);

    vector<uint32_t> writer_rt_args = {ntiles_per_core};
    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, all_cores, writer_rt_args);

    auto override_runtime_arguments_callback = [reader_kernel_id = unary_reader_kernel_id,
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

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks tilize_with_val_padding_multi_core(
    const Tensor& a, Tensor& output, const float pad_value) {
    if (a.memory_config().is_sharded()) {
        return tilize_with_val_padding_multi_core_sharded(a, output, pad_value);
    } else {
        return tilize_with_val_padding_multi_core_interleaved(a, output, pad_value);
    }
}

}  // namespace ttnn::operations::data_movement::detail
