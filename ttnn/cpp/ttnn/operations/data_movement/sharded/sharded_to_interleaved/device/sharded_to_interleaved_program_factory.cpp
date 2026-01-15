// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ShardedToInterleavedProgramFactory::cached_program_t ShardedToInterleavedProgramFactory::create(
    const ShardedToInterleavedParams& operation_attributes,
    const ShardedToInterleavedInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    const uint32_t num_slices = operation_attributes.num_slices;
    const uint32_t slice_index = operation_attributes.slice_index;
    const bool is_l1_aligned = true;

    tt_metal::Program program{};
    uint32_t num_units_per_shard, input_unit_size, output_unit_size, num_units_per_shard_width,
        num_units_per_shard_height, num_units_offset, num_units_per_row, num_units_height,
        num_units_per_shard_height_last, num_units_per_shard_width_last;

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout();

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t num_cores_unpadded = num_cores;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);

    CoreCoord end_core = cores[num_cores - 1];
    if (output.layout() == Layout::TILE) {
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = output.padded_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        num_units_height = output.physical_volume() / output.padded_shape()[-1] / TILE_HEIGHT / num_slices;
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
    } else {
        input_unit_size = shard_spec.shape[1] * input.element_size();
        output_unit_size = shard_spec.shape[1] * output.element_size();
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = output.padded_shape()[-1] * output.element_size();
        num_units_offset = 1;
        num_units_height = input.physical_volume() / input.padded_shape()[-1];
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            output_unit_size - (round_up(num_units_per_row, output_unit_size) - num_units_per_row);
    }

    // re-calculate end_core in the case shard grid is larger than used grid
    if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_cores_unpadded = div_up(num_units_height, num_units_per_shard_height);
    } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
        if (output.layout() == Layout::TILE) {
            num_cores_unpadded = div_up(num_units_per_row, num_units_per_shard_width);
        } else {
            num_cores_unpadded = div_up(num_units_per_row, output_unit_size);
        }
    }
    end_core = cores[num_cores_unpadded - 1];

    // Create CoreRangeSet for only the cores that will be used (fixes NOC error when grid > data)
    CoreRangeSet used_cores = num_cores_unpadded < num_cores
                                  ? select_from_corerangeset(all_cores, 0, num_cores_unpadded - 1, rm_orientation)
                                  : all_cores;

    bool convert_df = input_cb_data_format != output_cb_data_format;

    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t out_cb_index = src0_cb_index;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t input_page_size = align(input_unit_size, input.buffer()->alignment());
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, input_page_size)
            .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, used_cores, cb_src0_config);
    if (convert_df) {
        out_cb_index = CBIndex::c_16;
        uint32_t output_page_size = align(output_unit_size, output.buffer()->alignment());
        tt_metal::CircularBufferConfig output_cb_out_config =
            tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
                .set_page_size(out_cb_index, output_page_size);
        tt_metal::CreateCircularBuffer(program, used_cores, output_cb_out_config);
    }

    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index};

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        used_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);

    tt_metal::KernelHandle unary_writer_kernel_id;
    if (input.layout() == Layout::TILE) {
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)out_cb_index};
        TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

        unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "writer_unary_sharded_blocks_interleaved_start_id.cpp",
            used_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    } else {
        std::vector<uint32_t> writer_compile_time_args = {out_cb_index, num_units_per_row};
        TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

        unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
            used_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }
    if (convert_df) {
        std::vector<uint32_t> compute_kernel_args = {num_units_per_shard};

        tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/eltwise_copy.cpp",
            used_cores,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args});
    }

    tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, used_cores, {num_units_per_shard});

    uint32_t starting_idx_h = operations::data_movement::detail::calculate_starting_idx_h(output, num_slices, slice_index);
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;

    for (uint32_t core_idx = 0; core_idx < num_cores_unpadded; core_idx++) {
        const auto& core = cores[core_idx];
        uint32_t shard_height = num_units_per_shard_height;
        uint32_t shard_width = input.layout() == Layout::TILE ? num_units_per_shard_width : output_unit_size;
        if (input.layout() == Layout::TILE) {
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
            uint32_t l1_alignment = hal::get_l1_alignment();
            uint32_t padded_shard_width = align(output_unit_size, dst_buffer->alignment());
            if (is_blackhole or is_l1_aligned) {
                if (!dst_is_dram or is_l1_aligned) {
                    padded_shard_width = align(output_unit_size, l1_alignment);
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
                 (is_blackhole) ? shard_width : padded_shard_width,
                 curr_idx_w,
                 curr_idx_h});
            curr_idx_w += output_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .unary_reader_kernel_id = unary_reader_kernel_id,
            .unary_writer_kernel_id = unary_writer_kernel_id,
            .cb_src0 = cb_src0,
            .cores = cores,
            .num_slices = num_slices,
            .num_cores_unpadded = num_cores_unpadded,
        }};
}

void ShardedToInterleavedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ShardedToInterleavedParams& operation_attributes,
    const ShardedToInterleavedInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& output = output_tensor;
    auto& program = cached_program.program;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& cb_src0 = cached_program.shared_variables.cb_src0;
    const auto& cores = cached_program.shared_variables.cores;
    const uint32_t num_slices = cached_program.shared_variables.num_slices;
    const uint32_t num_cores_unpadded = cached_program.shared_variables.num_cores_unpadded;

    auto* src_buffer = tensor_args.input_tensor.buffer();
    auto* dst_buffer = output.buffer();

    // Calculate starting_idx_h if partial operation
    uint32_t starting_idx_h = operations::data_movement::detail::calculate_starting_idx_h(output, num_slices, operation_attributes.slice_index);

    auto& runtime_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);
    for (uint32_t core_idx = 0; core_idx < num_cores_unpadded; core_idx++) {
        const auto& core = cores[core_idx];
        auto& runtime_args = runtime_args_by_core[core.x][core.y];
        runtime_args[0] = dst_buffer->address();
        if (num_slices > 1) {
            runtime_args[8] = starting_idx_h;
        }
    }
    UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
}

}  // namespace ttnn::prim
