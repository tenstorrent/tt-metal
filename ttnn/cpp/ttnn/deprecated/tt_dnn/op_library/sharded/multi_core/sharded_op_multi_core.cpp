// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/sharded/sharded_op.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

// Utility function
uint32_t calculate_starting_idx_h(const Tensor& tensor, uint32_t num_slices, uint32_t slice_index) {
    if (num_slices <= 1) {
        return 0;
    }

    uint32_t num_tiles_height = tensor.volume() / tensor.get_legacy_shape()[-1] / TILE_HEIGHT;
    uint32_t num_tiles_width = tensor.get_legacy_shape()[-1] / TILE_WIDTH;
    uint32_t total_num_tiles = num_tiles_height * num_tiles_width;

    uint32_t num_tiles_per_slice = total_num_tiles / num_slices;
    uint32_t starting_tile_in_slice = num_tiles_per_slice * slice_index;
    return starting_tile_in_slice;
}

operation::ProgramWithCallbacks interleaved_to_sharded_multi_core(
    const Tensor& input, const Tensor& output, uint32_t num_slices, uint32_t slice_index) {
    std::shared_ptr<tt_metal::Program> program = tt_metal::CreateProgram();

    uint32_t num_units, num_units_per_shard, input_unit_size, output_unit_size, num_units_per_shard_width,
        num_units_per_shard_height, num_units_offset, num_units_per_row, num_units_per_shard_height_last,
        num_units_per_shard_width_last, padded_offset_bytes;

    tt_metal::Device* device = input.device();

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    auto shard_spec = output.shard_spec().value();
    auto shard_strategy = output.memory_config().memory_layout;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    CoreCoord end_core = (*shard_spec.grid.ranges().rbegin()).end_coord;
    if (input.get_layout() == Layout::TILE) {
        num_units = input.volume() / TILE_HW;
        input_unit_size = tt_metal::detail::TileSize(input_cb_data_format);
        output_unit_size = tt_metal::detail::TileSize(output_cb_data_format);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.get_legacy_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = input.volume() / input.get_legacy_shape()[-1] / TILE_HEIGHT / num_slices;
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
        padded_offset_bytes = (num_units_per_shard_width - num_units_per_shard_width_last) * input_unit_size;
    } else {
        num_units = (input.volume() / input.get_legacy_shape()[-1] / shard_spec.shape[0]) *
                    (input.get_legacy_shape()[-1] / shard_spec.shape[1]);
        input_unit_size = shard_spec.shape[1] * input.element_size();
        output_unit_size = shard_spec.shape[1] * output.element_size();
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.get_legacy_shape()[-1] * input.element_size();
        num_units_offset = 1;
        uint32_t num_units_height = input.volume() / input.get_legacy_shape()[-1];
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        // TODO: Use a different variable name. Units refers to pages, but this is being used as size
        num_units_per_shard_width_last =
            input_unit_size - (round_up(num_units_per_row, input_unit_size) - num_units_per_row);
        padded_offset_bytes = align(input_unit_size, input.buffer()->alignment());
    }

    bool convert_df = input_cb_data_format != output_cb_data_format;

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    auto all_cores = shard_spec.grid;
    uint32_t input_cb_index = CB::c_in0;
    uint32_t scratch_cb_index = CB::c_in1;
    uint32_t out_cb_index = input_cb_index;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t output_page_size = align(output_unit_size, dst_buffer->alignment());
    if (convert_df) {
        out_cb_index = CB::c_out0;
        uint32_t input_page_size = align(input_unit_size, src_buffer->alignment());
        tt_metal::CircularBufferConfig input_cb_out_config =
            tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{input_cb_index, input_cb_data_format}})
                .set_page_size(input_cb_index, input_page_size);
        auto cb_input = tt_metal::CreateCircularBuffer(program, all_cores, input_cb_out_config);
    }
    tt_metal::CircularBufferConfig output_cb_out_config =
        tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, output_page_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_out_config);
    if (src_is_dram && input_unit_size % DRAM_ALIGNMENT != 0) {
        uint32_t scratch_cb_page_size = align(input_unit_size, DRAM_ALIGNMENT);
        tt_metal::CircularBufferConfig scratch_cb_out_config =
            tt_metal::CircularBufferConfig(1 * scratch_cb_page_size, {{scratch_cb_index, input_cb_data_format}})
                .set_page_size(scratch_cb_index, scratch_cb_page_size);
        auto cb_scratch = tt_metal::CreateCircularBuffer(program, all_cores, scratch_cb_out_config);
    }

    tt_metal::KernelHandle unary_reader_kernel_id;
    if (input.get_layout() == Layout::TILE) {
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)input_cb_index, (std::uint32_t)src_is_dram, all_cores.num_cores()};

        unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    } else {
        bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(num_units_per_row);
        uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(num_units_per_row) : 0;
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)input_cb_index,
            (std::uint32_t)scratch_cb_index,
            (std::uint32_t)src_is_dram,
            (std::uint32_t)src_stick_size_is_power_of_two,
            (std::uint32_t)src_log2_stick_size};

        unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/sharded/kernels/dataflow/"
            "reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    }

    std::vector<uint32_t> writer_compile_time_args = {out_cb_index};
    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt_metal::KernelHandle compute_kernel_id = 0;
    if (convert_df) {
        compute_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/sharded/kernels/compute/eltwise_copy.cpp",
            all_cores,
            tt_metal::ComputeConfig{});
    }

    uint32_t starting_idx_h = calculate_starting_idx_h(input, num_slices, slice_index);
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;

    const auto cores = corerange_to_cores(shard_spec.grid, std::nullopt, rm_orientation);
    for (const auto& core : cores) {
        uint32_t curr_num_units_per_shard = num_units_per_shard;
        if (input.get_layout() == Layout::TILE) {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = num_units_per_shard_width;
            uint32_t padded_offset = 0;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core == end_core) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core == end_core) {
                    shard_width = num_units_per_shard_width_last;
                    padded_offset = padded_offset_bytes;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                        padded_offset = padded_offset_bytes;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                        padded_offset = padded_offset_bytes;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            curr_num_units_per_shard = shard_height * num_units_per_shard_width;
            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {src_buffer->address(),
                 shard_height,
                 shard_width,
                 padded_offset,
                 num_units_offset,
                 curr_num_units_per_shard,
                 curr_idx_h + curr_idx_w,
                 starting_idx_h});
            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = input_unit_size;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                    curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                        curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                        curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                    }
                }
            }

            bool aligned = src_is_dram ? curr_idx_w % DRAM_ALIGNMENT == 0 : true;
            uint32_t aligned_width_offset, aligned_shard_width, aligned_offset;
            if (!aligned) {
                aligned_width_offset = round_down(curr_idx_w, DRAM_ALIGNMENT);
                aligned_offset = curr_idx_w - aligned_width_offset;
                aligned_shard_width = aligned_offset + shard_width;
            } else {
                aligned_width_offset = curr_idx_w;
                aligned_shard_width = shard_width;
                aligned_offset = 0;
            }

            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {src_buffer->address(),
                 num_units_per_row,
                 shard_height,
                 shard_width,
                 padded_offset_bytes,
                 static_cast<uint32_t>(aligned),
                 aligned_width_offset,
                 aligned_shard_width,
                 aligned_offset,
                 curr_idx_h});
            curr_idx_w += input_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {curr_num_units_per_shard});
        if (convert_df) {
            tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {curr_num_units_per_shard});
        }
    }

    auto override_runtime_arguments_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, cb_output, cores, num_slices](
            const void* operation,
            std::shared_ptr<Program>  program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors.at(0).buffer();
            auto dst_buffer = output_tensors.at(0).buffer();

            bool partial_op = num_slices > 1;
            uint32_t starting_idx_h = 0;
            if (partial_op) {
                uint32_t runtime_slice_index = static_cast<const ttnn::operations::data_movement::InterleavedToShardedPartialDeviceOperation*>(operation)->slice_index;
                starting_idx_h = calculate_starting_idx_h(input_tensors.at(0), num_slices, runtime_slice_index);
            }

            auto& runtime_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
            for (const auto& core : cores) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer->address();
                if (partial_op) {
                    runtime_args[7] = starting_idx_h;
                }
            }
            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
        };

    return {.program = program, .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks sharded_to_interleaved_multi_core(
    const Tensor& input, const Tensor& output, uint32_t num_slices, uint32_t slice_index) {
    std::shared_ptr<tt_metal::Program> program = tt_metal::CreateProgram();

    uint32_t num_units, num_units_per_shard, input_unit_size, output_unit_size, num_units_per_shard_width,
        num_units_per_shard_height, num_units_offset, num_units_per_row, num_units_per_shard_height_last,
        num_units_per_shard_width_last;

    tt_metal::Device* device = input.device();

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    CoreCoord end_core = (*shard_spec.grid.ranges().rbegin()).end_coord;
    if (output.get_layout() == Layout::TILE) {
        num_units = input.volume() / TILE_HW;
        input_unit_size = tt_metal::detail::TileSize(input_cb_data_format);
        output_unit_size = tt_metal::detail::TileSize(output_cb_data_format);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = output.get_legacy_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = output.volume() / output.get_legacy_shape()[-1] / TILE_HEIGHT / num_slices;
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
    } else {
        num_units = (output.volume() / output.get_legacy_shape()[-1] / shard_spec.shape[0]) *
                    (input.get_legacy_shape()[-1] / shard_spec.shape[1]);
        input_unit_size = shard_spec.shape[1] * input.element_size();
        output_unit_size = shard_spec.shape[1] * output.element_size();
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = output.get_legacy_shape()[-1] * output.element_size();
        num_units_offset = 1;
        uint32_t num_units_height = input.volume() / input.get_legacy_shape()[-1];
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            output_unit_size - (round_up(num_units_per_row, output_unit_size) - num_units_per_row);
    }

    bool convert_df = input_cb_data_format != output_cb_data_format;

    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t out_cb_index = src0_cb_index;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t input_page_size = align(input_unit_size, input.buffer()->alignment());
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, input_page_size)
            .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);
    if (convert_df) {
        out_cb_index = CB::c_out0;
        uint32_t output_page_size = align(output_unit_size, output.buffer()->alignment());
        tt_metal::CircularBufferConfig output_cb_out_config =
            tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
                .set_page_size(out_cb_index, output_page_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_out_config);
    }

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index};

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    tt_metal::KernelHandle unary_writer_kernel_id;
    if (input.get_layout() == Layout::TILE) {
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)out_cb_index, (std::uint32_t)dst_is_dram};

        unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    } else {
        bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(num_units_per_row);
        uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(num_units_per_row) : 0;
        std::vector<uint32_t> writer_compile_time_args = {
            (std::uint32_t)out_cb_index,
            (std::uint32_t)dst_is_dram,
            (std::uint32_t)dst_stick_size_is_power_of_two,
            (std::uint32_t)dst_log2_stick_size};

        unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/sharded/kernels/dataflow/"
            "writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }
    if (convert_df) {
        vector<uint32_t> compute_kernel_args = {num_units_per_shard};

        auto eltwise_unary_kernel_group_1 = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/eltwise_copy.cpp",
            all_cores,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args});
    }

    tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, {num_units_per_shard});

    uint32_t starting_idx_h = calculate_starting_idx_h(output, num_slices, slice_index);
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;

    const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);
    uint32_t padded_shard_width = align(output_unit_size, dst_buffer->alignment());
    for (const auto& core : cores) {
        if (input.get_layout() == Layout::TILE) {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = num_units_per_shard_width;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {dst_buffer->address(),
                 num_units_per_shard_height,
                 num_units_per_shard_width,
                 shard_height,
                 shard_width,
                 num_units_offset,
                 num_units_per_shard,
                 curr_idx_h + curr_idx_w,
                 starting_idx_h});
            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = output_unit_size;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {dst_buffer->address(),
                 num_units_per_row,
                 shard_height,
                 shard_width,
                 padded_shard_width,
                 curr_idx_w,
                 curr_idx_h});
            curr_idx_w += output_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
    }
    auto override_runtime_arguments_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cores, num_slices](
            const void* operation,
            std::shared_ptr<Program>  program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors[0].buffer();

            Buffer* dst_buffer = nullptr;
            uint32_t starting_idx_h = 0;
            const bool partial_op = num_slices > 1 || (num_slices == 1 && output_tensors.size() == 0);
            if (partial_op) {
                // If we have num_slices > 1, it means that our op is S->I partial.
                // And currently we store output tensors there as input[1]
                // If we have num_slices == 1, and output_tensors.size() == 0,
                // it also means we are in S->I partial and must read from output from inputs[1]
                dst_buffer = input_tensors.at(1).buffer();

                // Calculate starting_idx_h
                uint32_t runtime_slice_index = static_cast<const ttnn::operations::data_movement::ShardedToInterleavedPartialDeviceOperation*>(operation)->slice_index;
                starting_idx_h = calculate_starting_idx_h(input_tensors.at(1), num_slices, runtime_slice_index);
            } else {
                dst_buffer = output_tensors.at(0).buffer();
            }
            // TODO: Make these common args instead
            auto& runtime_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);
            for (const auto& core : cores) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = dst_buffer->address();
                if (partial_op) {
                    runtime_args[8] = starting_idx_h;
                }
            }
            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        };

    return {.program = program, .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

std::unordered_map<CoreCoord, std::vector<PageStride>> get_core_page_ranges(
    Buffer* input_buffer, Buffer* output_buffer) {
    auto output_buffer_page_mapping = generate_buffer_page_mapping(*output_buffer);
    auto input_buffer_page_mapping = generate_buffer_page_mapping(*input_buffer);

    const auto& output_shard_to_host_mapping = output_buffer_page_mapping.dev_page_to_host_page_mapping_;
    const auto& input_page_to_local_page_mapping = input_buffer_page_mapping.host_page_to_local_shard_page_mapping_;
    const auto& host_page_to_input_page_mapping = input_buffer_page_mapping.host_page_to_dev_page_mapping_;

    auto output_cores = output_buffer_page_mapping.all_cores_;
    // First get output_core to vector< pair<input_core, input_page> (num_pages_in_output)
    std::vector<std::vector<std::optional<std::pair<CoreCoord, uint32_t>>>> output_core_to_vector_input_core_page(
        output_cores.size());

    for (uint32_t output_page_id = 0; output_page_id < output_buffer->num_dev_pages(); output_page_id++) {
        auto output_core_id = output_buffer_page_mapping.dev_page_to_core_mapping_[output_page_id];
        TT_ASSERT(output_core_id < output_cores.size());
        auto host_page = output_shard_to_host_mapping[output_page_id];
        std::optional<std::pair<CoreCoord, uint32_t>> mapped_page = std::nullopt;
        if (host_page.has_value()) {
            auto input_page = host_page_to_input_page_mapping[host_page.value()];
            auto local_input_page = input_page_to_local_page_mapping[host_page.value()];
            auto input_core =
                input_buffer_page_mapping.all_cores_[input_buffer_page_mapping.dev_page_to_core_mapping_[input_page]];
            mapped_page = std::make_optional<std::pair<CoreCoord, uint32_t>>({input_core, local_input_page});
        }
        output_core_to_vector_input_core_page[output_core_id].push_back(mapped_page);
    }

    // now compress to output_core to vector<pair<input_core, input_page_range> (num_page_ranges_in_output)
    std::unordered_map<CoreCoord, std::vector<PageStride>> ret_map;
    ret_map.reserve(output_cores.size());

    auto output_core_host_page_indices = output_buffer_page_mapping.core_host_page_indices_;
    auto device = input_buffer->device();
    auto full_grid = device->compute_with_storage_grid_size();
    CoreCoord end_core = (*output_buffer->shard_spec().grid().ranges().rbegin()).end_coord;
    uint32_t output_core_id = 0;
    for (auto output_core : output_cores) {
        ret_map.try_emplace(output_core, std::vector<PageStride>{});

        const auto& input_cores_with_pages = output_core_to_vector_input_core_page[output_core_id];
        auto it = input_cores_with_pages.begin();
        const auto end = input_cores_with_pages.end();

        while (it != end) {
            // hit padding, will see how many consecutive pages has padding to make a padded range
            if (!it->has_value()) {
                auto consecutive_it = it + 1;
                auto last_it_consec = it;
                while (consecutive_it != end) {
                    if (consecutive_it->has_value()) {
                        break;
                    }
                    last_it_consec = consecutive_it;
                    consecutive_it = consecutive_it + 1;
                }
                uint32_t stride_size = std::distance(it, last_it_consec) + 1;
                ret_map[output_core].push_back(PageStride{
                    .start_core = output_core,
                    .start_data = 0,
                    .stride_size = stride_size,
                    .stride = Stride{.core = {0, 0}, .data = 0},
                    .num_strides = 1,
                    .skip = true});
                it += stride_size;
            } else {
                const auto start_core = it->value().first;
                const auto start_page = it->value().second;
                auto expected_next_page = start_page + 1;
                Stride stride = Stride{.core = {0, 0}, .data = 0};
                if ((it + 1) == end) {
                    ret_map[output_core].push_back(PageStride{
                        .start_core = start_core,
                        .start_data = it->value().second,
                        .stride_size = 1,
                        .stride = stride,
                        .num_strides = 1,
                        .skip = false});
                    it = end;
                } else {
                    // first get a single stride, go through the number of consecutive pages in the same core
                    auto consecutive_it = it + 1;
                    auto last_it_consec = it;
                    while (consecutive_it != end and consecutive_it->has_value()) {
                        auto next_input_page = *(consecutive_it);
                        auto curr_input_page = *(last_it_consec);
                        // diff core , not consecutive
                        if (curr_input_page.value().first != next_input_page.value().first) {
                            break;
                        }
                        // not consecutive
                        else if ((curr_input_page.value().second + 1) != next_input_page.value().second) {
                            break;
                        }
                        // next page is padding
                        consecutive_it = consecutive_it + 1;
                        last_it_consec = consecutive_it;
                    }
                    uint32_t stride_size = std::distance(it, last_it_consec);
                    if (last_it_consec == it) {
                        stride_size = 1;
                    }
                    auto stride_it = it + stride_size;
                    auto last_it_stride = it;

                    // TT_ASSERT((stride_it == end) or stride_it->has_value());
                    TT_ASSERT(last_it_stride->has_value());
                    // if stride_range is within same core
                    // the jump in data is end of curr - end last stride
                    // if stride range is in diff core
                    // jump in data is curr - beginning of last stride
                    uint32_t data_stride;
                    if ((stride_it != end) and (stride_it != it) and stride_it->has_value()) {
                        // data stride within core
                        if (stride_it->has_value() and stride_it->value().first == last_it_stride->value().first and
                            (stride_it->value().second > last_it_stride->value().second)) {
                            auto next_input_page = *(stride_it);
                            auto prev_input_page = *(last_it_stride);
                            TT_ASSERT(prev_input_page.has_value());
                            TT_ASSERT(next_input_page.has_value());
                            data_stride = next_input_page.value().second - prev_input_page.value().second - stride_size;
                            stride = Stride{.core = {0, 0}, .data = data_stride};
                        }
                        // strided core but same data
                        // currently only handling increasing cores within same stride
                        // TODO : negative strides for cores
                        else if (
                            stride_it->has_value() and (stride_it->value().first != last_it_stride->value().first) and
                            (stride_it->value().first.x >= it->value().first.x and
                             stride_it->value().first.y >= it->value().first.y) and
                            (stride_it->value().second == it->value().second)) {
                            auto next_input_page = *(stride_it);
                            auto prev_input_page = *it;
                            TT_ASSERT(prev_input_page.has_value());
                            TT_ASSERT(next_input_page.has_value());
                            data_stride = 0;
                            stride = Stride{
                                .core =
                                    {next_input_page.value().first.x - prev_input_page.value().first.x,
                                     next_input_page.value().first.y - prev_input_page.value().first.y},
                                .data = data_stride};
                        }
                        // diff data and diff core, not handled yet
                        else {
                            TT_ASSERT(it->has_value());
                            ret_map[output_core].push_back(PageStride{
                                .start_core = start_core,
                                .start_data = it->value().second,
                                .stride_size = stride_size,
                                .stride = stride,
                                .num_strides = 1,
                                .skip = false});
                            it = stride_it;
                            continue;
                        }
                        // TODO add stride of data and core
                    }
                    // only single stride
                    else {
                        data_stride = 0;
                    }

                    TT_ASSERT(stride.core.x < full_grid.x and stride.core.y < full_grid.y);
                    TT_ASSERT(data_stride < output_buffer->num_pages());
                    auto stride_start = stride_it;
                    uint32_t num_strides = 1;
                    while (stride_it != end and stride_it->has_value()) {
                        bool stride_not_complete = false;
                        auto stride_it_inner = stride_it + 1;
                        auto last_it_stride_inner = stride_it;
                        for (uint32_t i = 0; i < stride_size - 1; i++) {
                            auto next_input_page = *(stride_it_inner);
                            auto curr_input_page = *(last_it_stride_inner);
                            TT_ASSERT(curr_input_page.has_value());
                            int increment = 1;
                            if (!(next_input_page.has_value()) or
                                (next_input_page.value().first != curr_input_page.value().first) or
                                ((int)next_input_page.value().second !=
                                 (int)(curr_input_page.value().second) + (int)increment)) {
                                stride_not_complete = true;
                                break;
                            }
                            last_it_stride_inner = stride_it_inner;
                            stride_it_inner = stride_it_inner + 1;
                        }
                        if (stride_not_complete) {
                            break;
                        }
                        num_strides++;
                        last_it_stride = stride_it_inner - 1;
                        stride_it = stride_it_inner;
                        if (stride_it == end or !stride_it->has_value()) {
                            break;
                        }
                        auto next_input_page = *(stride_it);
                        auto curr_input_page = *(last_it_stride);
                        bool core_stride = ((stride.core.x != 0) or (stride.core.y != 0));
                        // TT_ASSERT(curr_input_page.has_value());
                        if (!curr_input_page.has_value() or !next_input_page.has_value() or
                            (next_input_page.value().first.x - curr_input_page.value().first.x != stride.core.x) or
                            (next_input_page.value().first.y - curr_input_page.value().first.y != stride.core.y) or
                            (abs((int)next_input_page.value().second - (int)curr_input_page.value().second) !=
                             (int)stride.data)) {
                            break;
                        }
                    }
                    TT_ASSERT(it->has_value());
                    ret_map[output_core].push_back(PageStride{
                        .start_core = start_core,
                        .start_data = it->value().second,
                        .stride_size = stride_size,
                        .stride = stride,
                        .num_strides = num_strides,
                        .skip = false});
                    it = stride_it;
                }
            }
        }
        output_core_id++;
    }

    return ret_map;
}

enum class ReshardStridesInRange { ALL_STRIDES, FIRST_HALF, SECOND_HALF };

std::vector<uint32_t> get_runtime_args_for_given_ranges(
    const std::vector<uint32_t>& physical_core_coords,
    const std::vector<PageStride>& page_stride_vector,
    const uint32_t output_page_offset,
    const uint32_t& input_addr,
    const uint32_t starting_range,
    const uint32_t ending_range,
    const ReshardStridesInRange reshard_strides_in_range = ReshardStridesInRange::ALL_STRIDES) {
    std::vector<uint32_t> runtime_args = physical_core_coords;
    runtime_args.push_back(input_addr);
    runtime_args.push_back(0);
    runtime_args.push_back(ending_range - starting_range);
    runtime_args.push_back(output_page_offset);
    uint32_t num_output_pages = 0;

    for (uint32_t range_id = starting_range; range_id < ending_range; range_id++) {
        PageStride ps = page_stride_vector[range_id];
        uint32_t num_strides;
        uint32_t start_core_x;
        uint32_t start_core_y;
        uint32_t start_data;
        if (reshard_strides_in_range == ReshardStridesInRange::ALL_STRIDES) {
            num_strides = ps.num_strides;
            start_core_x = ps.start_core.x;
            start_core_y = ps.start_core.y;
            start_data = ps.start_data;
        } else {
            if (reshard_strides_in_range == ReshardStridesInRange::FIRST_HALF) {
                num_strides = ps.num_strides / 2;
                start_core_x = ps.start_core.x;
                start_core_y = ps.start_core.y;
                start_data = ps.start_data;
            } else {
                uint32_t strides_in_first_half = ps.num_strides / 2;
                num_strides = ps.num_strides - (strides_in_first_half);
                start_core_x = ps.start_core.x + (strides_in_first_half * ps.stride.core.x);
                start_core_y = ps.start_core.y + (strides_in_first_half * ps.stride.core.y);
                start_data = ps.start_data + (strides_in_first_half * ps.start_data);
            }
        }
        if (num_strides > 0) {
            uint32_t core_start_stride =
                (start_core_x << 24) | (start_core_y << 16) | (ps.stride.core.x << 8) | ps.stride.core.y;
            runtime_args.push_back((uint32_t)core_start_stride);  // start_x
            uint32_t stride_data_start = (ps.stride.data << 16) | (start_data);
            runtime_args.push_back((uint32_t)stride_data_start);  // stride_data
            uint32_t stride_size_num_strides = (ps.stride_size << 16) | (num_strides << 8) | ((uint32_t)ps.skip);
            runtime_args.push_back((uint32_t)stride_size_num_strides);  // stride_size
            num_output_pages += ps.stride_size * num_strides;
        }
    }
    runtime_args[physical_core_coords.size() + 1] = num_output_pages;
    return runtime_args;
}

operation::ProgramWithCallbacks reshard_multi_core_same_width(const Tensor& input, Tensor& output) {
    auto device = input.device();

    std::shared_ptr<tt_metal::Program> program = tt_metal::CreateProgram();

    const auto input_shard_spec = input.shard_spec().value();
    const auto output_shard_spec = output.shard_spec().value();
    const auto& all_cores = output_shard_spec.grid;
    auto grid = input.buffer()->buffer_type() == BufferType::DRAM ? device->dram_grid_size()
                                                                  : device->compute_with_storage_grid_size();
    auto input_core_type = input.buffer()->core_type();
    constexpr uint32_t dst_cb_index = CB::c_in0;
    auto input_cores = corerange_to_cores(
        input_shard_spec.grid, std::nullopt, input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto output_cores =
        corerange_to_cores(all_cores, std::nullopt, output_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    uint32_t total_size, unit_size, input_units_per_shard, output_units_per_shard;
    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    uint32_t num_output_units = input.buffer()->num_pages();
    if (input.get_layout() == Layout::TILE) {
        unit_size = tt_metal::detail::TileSize(data_format);
        input_units_per_shard = input_shard_spec.numel() / TILE_HW;
        output_units_per_shard = output_shard_spec.numel() / TILE_HW;
        total_size = output_units_per_shard * unit_size;
    } else {
        unit_size = output_shard_spec.shape[1] * output.element_size();
        input_units_per_shard = input_shard_spec.shape[0];
        output_units_per_shard = output_shard_spec.shape[0];
        total_size = output_units_per_shard * unit_size;
    }

    tt_metal::KernelHandle kernel_id_0 = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/sharded/kernels/dataflow/reshard_same_width_reader.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig({dst_cb_index}));

    tt_metal::KernelHandle kernel_id_1 = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/sharded/kernels/dataflow/reshard_same_width_reader.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig({dst_cb_index}));

    tt_metal::CircularBufferConfig cb_dst_config =
        tt_metal::CircularBufferConfig(total_size, {{dst_cb_index, data_format}})
            .set_page_size(dst_cb_index, unit_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);

    uint32_t input_core_idx = 0;
    uint32_t input_core_units_rem = input_units_per_shard;
    uint32_t input_address = input.buffer()->address();
    auto input_buffer_type = input.buffer()->buffer_type();
    auto bank_id = device->bank_ids_from_logical_core(input_buffer_type, input_cores[input_core_idx])[0];
    uint32_t bank_offset = device->bank_offset(input_buffer_type, bank_id);
    auto input_core = device->physical_core_from_logical_core(input_cores[input_core_idx], input_core_type);

    std::array<tt_metal::KernelHandle, 2> kernels = {kernel_id_0, kernel_id_1};
    uint32_t output_units_left = num_output_units;
    for (const auto& core : output_cores) {
        uint32_t output_units_per_core = std::min(output_units_left, output_units_per_shard);
        output_units_left -= output_units_per_core;
        uint32_t output_units_per_kernel = div_up(output_units_per_core, kernels.size());
        for (const auto& kernel_id : kernels) {
            std::vector<uint32_t> kernel_args = {input_address, 0, 0};
            uint32_t output_units_to_get = std::min(output_units_per_core, output_units_per_kernel);
            if (output_units_to_get != 0) {
                uint32_t num_reads = 0;
                kernel_args[1] = (output_units_per_shard - output_units_per_core) * unit_size;
                while (output_units_to_get > 0) {
                    if (input_core_units_rem == 0) {
                        input_core_idx++;
                        input_core_units_rem = input_units_per_shard;
                        bank_id = device->bank_ids_from_logical_core(input_buffer_type, input_cores[input_core_idx])[0];
                        bank_offset = device->bank_offset(input_buffer_type, bank_id);
                        input_core = device->physical_core_from_logical_core(input_cores[input_core_idx], input_core_type);
                    }
                    uint32_t units_to_read = std::min(input_core_units_rem, output_units_to_get);
                    auto input_core =
                        device->physical_core_from_logical_core(input_cores[input_core_idx], input_core_type);
                    kernel_args.insert(
                        kernel_args.end(),
                        {static_cast<uint32_t>(input_core.x),
                         static_cast<uint32_t>(input_core.y),
                         (input_units_per_shard - input_core_units_rem) * unit_size + bank_offset,
                         units_to_read * unit_size});
                    output_units_per_core -= units_to_read;
                    output_units_to_get -= units_to_read;
                    input_core_units_rem -= units_to_read;
                    num_reads++;
                }
                kernel_args[2] = num_reads;
            }
            SetRuntimeArgs(program, kernel_id, core, kernel_args);
        }
    }

    auto override_runtime_arguments_callback = [kernel_id_0, kernel_id_1, cb_dst0, grid, output_cores](
                                                   const void* operation,
                                                   std::shared_ptr<Program>  program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);
        uint32_t input_addr = input.buffer()->address();
        auto& runtime_args_0_by_core = GetRuntimeArgs(program, kernel_id_0);
        auto& runtime_args_1_by_core = GetRuntimeArgs(program, kernel_id_1);
        for (auto core : output_cores) {
            auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
            auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
            runtime_args_0[0] = input_addr;
            runtime_args_1[0] = input_addr;
        }
        UpdateDynamicCircularBufferAddress(program, cb_dst0, *output.buffer());
    };

    return {.program = program, .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks reshard_multi_core_generic(const Tensor& input, Tensor& output) {
    auto device = input.device();
    auto output_core_to_page_range_pair = get_core_page_ranges(input.buffer(), output.buffer());

    std::shared_ptr<tt_metal::Program> program = tt_metal::CreateProgram();

    auto input_shard_spec = input.shard_spec().value();
    auto output_shard_spec = output.shard_spec().value();
    auto all_cores = output_shard_spec.grid;
    auto grid = input.buffer()->buffer_type() == BufferType::DRAM ? device->dram_grid_size()
                                                                  : device->compute_with_storage_grid_size();
    auto input_core_type = input.buffer()->core_type();
    uint32_t dst_cb_index = 16;
    auto cores =
        corerange_to_cores(all_cores, std::nullopt, output_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    uint32_t total_size, page_size, unit_size;
    auto output_shard_shape = output_shard_spec.shape;
    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    if (input.get_layout() == Layout::TILE) {
        page_size = tt_metal::detail::TileSize(data_format);
        unit_size = page_size;
        total_size = output_shard_spec.numel() / TILE_HW * unit_size;
    } else {
        unit_size = output_shard_spec.shape[1] * output.element_size();
        page_size = output.get_legacy_shape()[-1] * output.element_size();
        total_size = output_shard_shape[0] * unit_size;
    }

    tt_metal::KernelHandle kernel_id_0 = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/sharded/kernels/dataflow/reshard_reader.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig({dst_cb_index, (uint32_t)grid.x, (uint32_t)grid.y, page_size}));

    tt_metal::KernelHandle kernel_id_1 = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/sharded/kernels/dataflow/reshard_reader.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig({dst_cb_index, (uint32_t)grid.x, (uint32_t)grid.y, page_size}));

    tt_metal::CircularBufferConfig cb_dst_config =
        tt_metal::CircularBufferConfig(total_size, {{dst_cb_index, data_format}})
            .set_page_size(dst_cb_index, unit_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);

    std::vector<uint32_t> physical_core_coords;
    physical_core_coords.reserve(grid.x * grid.y);
    for (uint32_t i = 0; i < grid.x; i++) {
        auto physical_input_core = device->physical_core_from_logical_core(CoreCoord(i, 0), input_core_type);
        physical_core_coords.push_back(physical_input_core.x);
    }
    for (uint32_t i = 0; i < grid.y; i++) {
        auto physical_input_core = device->physical_core_from_logical_core(CoreCoord(0, i), input_core_type);
        physical_core_coords.push_back(physical_input_core.y);
    }

    for (const auto& core : cores) {
        auto page_stride_vector = output_core_to_page_range_pair.at(core);
        uint32_t num_ranges = page_stride_vector.size();
        std::vector<uint32_t> runtime_args = physical_core_coords;
        auto runtime_args_0 = get_runtime_args_for_given_ranges(
            physical_core_coords,
            page_stride_vector,
            0,
            input.buffer()->address(),
            0,
            div_up(page_stride_vector.size(), 2));
        auto output_page_offset =
            runtime_args_0[physical_core_coords.size() + 1];  // offset is equivalent to number of pages output in
                                                              // previous risc core
        tt_metal::SetRuntimeArgs(program, kernel_id_0, core, runtime_args_0);
        auto runtime_args_1 = get_runtime_args_for_given_ranges(
            physical_core_coords,
            page_stride_vector,
            output_page_offset,
            input.buffer()->address(),
            div_up(page_stride_vector.size(), 2),
            page_stride_vector.size());
        tt_metal::SetRuntimeArgs(program, kernel_id_1, core, runtime_args_1);
    }

    auto override_runtime_arguments_callback = [kernel_id_0, kernel_id_1, cb_dst0, grid, cores](
                                                   const void* operation,
                                                   std::shared_ptr<Program>  program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);
        uint32_t input_addr = input.buffer()->address();
        auto& runtime_args_0_by_core = GetRuntimeArgs(program, kernel_id_0);
        auto& runtime_args_1_by_core = GetRuntimeArgs(program, kernel_id_1);
        for (auto core : cores) {
            auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
            auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
            runtime_args_0[grid.x + grid.y] = input_addr;
            runtime_args_1[grid.x + grid.y] = input_addr;
        }
        UpdateDynamicCircularBufferAddress(program, cb_dst0, *output.buffer());
    };

    return {.program = program, .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks reshard_multi_core(const Tensor& input, Tensor& output) {
    if (input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
        output.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        return reshard_multi_core_same_width(input, output);
    } else {
        return reshard_multi_core_generic(input, output);
    }
}

}  // namespace tt_metal

}  // namespace tt
