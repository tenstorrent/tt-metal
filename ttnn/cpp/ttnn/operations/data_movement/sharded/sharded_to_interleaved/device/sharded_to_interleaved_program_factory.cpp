// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_op.hpp"

using namespace tt::constants;
using namespace tt;
namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks sharded_to_interleaved_multi_core(
    const Tensor& input, const Tensor& output, uint32_t num_slices, uint32_t slice_index) {
    tt_metal::Program program{};

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
        num_units_per_row = output.get_shape().with_tile_padding()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = output.volume() / output.get_shape().with_tile_padding()[-1] / TILE_HEIGHT / num_slices;
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
    } else {
        num_units = (output.volume() / output.get_shape().with_tile_padding()[-1] / shard_spec.shape[0]) *
                    (input.get_shape().with_tile_padding()[-1] / shard_spec.shape[1]);
        input_unit_size = shard_spec.shape[1] * input.element_size();
        output_unit_size = shard_spec.shape[1] * output.element_size();
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = output.get_shape().with_tile_padding()[-1] * output.element_size();
        num_units_offset = 1;
        uint32_t num_units_height = input.volume() / input.get_shape().with_tile_padding()[-1];
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
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp",
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
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
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
            Program& program,
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

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}




}
