// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_partial_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

InterleavedToShardedPartialProgramFactory::cached_program_t InterleavedToShardedPartialProgramFactory::create(
    const InterleavedToShardedPartialParams& params, const Tensor& input, Tensor& output) {
    // Partial op behavior
    bool keep_l1_aligned = false;

    uint32_t num_slices = params.num_slices;
    uint32_t slice_index = params.slice_index;

    tt::tt_metal::Program program{};

    uint32_t num_units_per_shard, input_unit_size, output_unit_size, num_units_per_shard_width,
        num_units_per_shard_height, num_units_offset, num_units_per_row, num_units_per_shard_height_last,
        num_units_per_shard_width_last, padded_offset_bytes;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = output.shard_spec().value();
    auto shard_strategy = output.memory_config().memory_layout();

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    CoreCoord end_core = (*shard_spec.grid.ranges().rbegin()).end_coord;

    bool convert_df = input_cb_data_format != output_cb_data_format;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);

    if (input.layout() == Layout::TILE) {
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        TT_FATAL(
            shard_spec.shape[0] % TILE_HEIGHT == 0 && shard_spec.shape[1] % TILE_WIDTH == 0,
            "Shard shape {} must be tile {}x{} sized!",
            shard_spec.shape,
            TILE_HEIGHT,
            TILE_WIDTH);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.padded_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT / num_slices;
        num_units_per_shard_height_last =
            num_units_per_shard_height -
            (tt::round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width -
            (tt::round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
        padded_offset_bytes = (num_units_per_shard_width - num_units_per_shard_width_last) * input_unit_size;
    } else {
        input_unit_size = shard_spec.shape[1] * input.element_size();
        output_unit_size = shard_spec.shape[1] * output.element_size();
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.padded_shape()[-1] * input.element_size();
        num_units_offset = 1;
        uint32_t num_units_height = input.physical_volume() / input.padded_shape()[-1];
        num_units_per_shard_height_last =
            num_units_per_shard_height -
            (tt::round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        // TODO: Use a different variable name. Units refers to pages, but this is being used as size
        num_units_per_shard_width_last =
            input_unit_size - (tt::round_up(num_units_per_row, input_unit_size) - num_units_per_row);
        // Adjust accordingly to l1 alignment, do it for all archs
        if (keep_l1_aligned) {
            padded_offset_bytes = tt::align(input_unit_size, hal::get_l1_alignment());
        } else {
            padded_offset_bytes = tt::align(input_unit_size, input.buffer()->alignment());
        }
    }

    auto all_cores = shard_spec.grid;
    uint32_t input_cb_index = tt::CBIndex::c_0;
    uint32_t scratch_cb_index = tt::CBIndex::c_1;
    uint32_t out_cb_index = input_cb_index;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t output_page_size = tt::align(output_unit_size, dst_buffer->alignment());
    if (convert_df) {
        out_cb_index = tt::CBIndex::c_16;
        uint32_t input_page_size = tt::align(input_unit_size, src_buffer->alignment());
        tt::tt_metal::CircularBufferConfig input_cb_out_config =
            tt::tt_metal::CircularBufferConfig(
                num_input_units * input_page_size, {{input_cb_index, input_cb_data_format}})
                .set_page_size(input_cb_index, input_page_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, input_cb_out_config);
    }
    tt::tt_metal::CircularBufferConfig output_cb_out_config =
        tt::tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, output_page_size);
    if (!dst_is_dram) {
        output_cb_out_config = output_cb_out_config.set_globally_allocated_address(*output.buffer());
    }
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, output_cb_out_config);
    uint32_t dram_alignment = hal::get_dram_alignment();
    if ((src_is_dram && (input_unit_size % dram_alignment != 0)) || is_blackhole || keep_l1_aligned) {
        uint32_t scratch_cb_page_size;
        // scratchpad going to be used to align DRAM (64B) to L1 (16B)
        if (is_blackhole) {
            scratch_cb_page_size = tt::align(input_unit_size, hal::get_l1_alignment());
        } else {
            scratch_cb_page_size = tt::align(input_unit_size, dram_alignment);
        }
        tt::tt_metal::CircularBufferConfig scratch_cb_out_config =
            tt::tt_metal::CircularBufferConfig(4 * scratch_cb_page_size, {{scratch_cb_index, input_cb_data_format}})
                .set_page_size(scratch_cb_index, scratch_cb_page_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, scratch_cb_out_config);
    }

    tt::tt_metal::KernelHandle unary_reader_kernel_id;
    if (input.layout() == Layout::TILE) {
        std::vector<uint32_t> reader_compile_time_args = {input_cb_index, all_cores.num_cores()};
        tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

        unary_reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "reader_unary_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    } else {
        std::vector<uint32_t> reader_compile_time_args = {input_cb_index, scratch_cb_index, num_units_per_row};
        tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

        unary_reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    }

    std::string writer_kernel;
    std::vector<uint32_t> writer_compile_time_args = {out_cb_index};
    if (dst_is_dram) {
        if (input.layout() == Layout::TILE) {
            writer_kernel = std::string(
                "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                "writer_unary_sharded_blocks_start_id.cpp");
        } else {
            writer_kernel = std::string(
                "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                "writer_unary_sharded_stick_layout_start_id.cpp");
        }
        shard_builder::extend_sharding_compile_time_args(output, writer_compile_time_args);
    } else {
        writer_kernel = std::string(
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp");
    }
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel, all_cores, tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    if (convert_df) {
        compute_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{});
    }

    uint32_t starting_idx_h =
        operations::data_movement::detail::calculate_starting_idx_h(input, num_slices, slice_index);
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;

    const auto cores = corerange_to_cores(shard_spec.grid, std::nullopt, rm_orientation);
    for (const auto& core : cores) {
        uint32_t curr_num_units_per_shard = num_units_per_shard;
        if (input.layout() == Layout::TILE) {
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

            // Reader run-time args
            std::vector<uint32_t> reader_run_time_args = {
                src_buffer->address(),
                shard_height,
                shard_width,
                padded_offset,
                num_units_offset,
                curr_num_units_per_shard,
                curr_idx_h + curr_idx_w,
                starting_idx_h};
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_run_time_args);

            // Writer run-time args
            uint32_t pad_offset = (num_units_per_shard_width - shard_width) * output_unit_size;
            std::vector<uint32_t> writer_run_time_args;
            if (dst_is_dram) {
                writer_run_time_args = {
                    dst_buffer->address(),
                    shard_height,
                    shard_width,
                    pad_offset,
                    curr_num_units_per_shard,
                    num_units_offset,
                    curr_idx_h + curr_idx_w,
                    starting_idx_h};
                shard_builder::extend_sharding_run_time_args(output, writer_run_time_args);
            } else {
                writer_run_time_args = {curr_num_units_per_shard};
            }
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_run_time_args);

            // Update indexing
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

            uint32_t dram_alignment = hal::get_dram_alignment();
            uint32_t l1_alignment = hal::get_l1_alignment();
            bool aligned =
                (src_is_dram ? (curr_idx_w % dram_alignment == 0) && (padded_offset_bytes % dram_alignment == 0)
                             : true);
            // for blackhole and keep_l1_aligned cases, always enforce unaligned kernel call
            aligned = aligned and !(is_blackhole);
            uint32_t aligned_width_offset, aligned_shard_width, aligned_offset;
            if (!aligned) {
                // TODO: is this right, leaving non BH case the same for now, should investigate
                if (!is_blackhole) {
                    aligned_width_offset = tt::round_down(curr_idx_w, dram_alignment);
                } else {
                    if (src_is_dram) {
                        aligned_width_offset = tt::round_down(curr_idx_w, dram_alignment);
                    } else {
                        aligned_width_offset = tt::round_down(curr_idx_w, l1_alignment);
                    }
                }
                aligned_offset = curr_idx_w - aligned_width_offset;
                aligned_shard_width = aligned_offset + shard_width;
            } else {
                aligned_width_offset = curr_idx_w;
                aligned_shard_width = shard_width;
                aligned_offset = 0;
            }

            // Reader run-time args
            std::vector<uint32_t> reader_run_time_args = {
                src_buffer->address(),
                num_units_per_row,
                shard_height,
                shard_width,
                padded_offset_bytes,
                static_cast<uint32_t>(aligned),
                aligned_width_offset,
                aligned_shard_width,
                aligned_offset,
                curr_idx_h};
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_run_time_args);

            // Writer run-time args
            std::vector<uint32_t> writer_run_time_args;
            if (dst_is_dram) {
                uint32_t page_id_within_row = curr_idx_w / input_unit_size;
                uint32_t output_width_in_pages = tt::div_up(num_units_per_row, input_unit_size);
                uint32_t start_id = (curr_idx_h * output_width_in_pages) + page_id_within_row;
                writer_run_time_args = {
                    dst_buffer->address(),
                    shard_height,
                    shard_width,
                    padded_offset_bytes,
                    start_id,
                    output_width_in_pages};
                shard_builder::extend_sharding_run_time_args(output, writer_run_time_args);
            } else {
                writer_run_time_args = {curr_num_units_per_shard};
            }
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_run_time_args);

            // Update indexing
            curr_idx_w += input_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
        if (convert_df) {
            tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {curr_num_units_per_shard});
        }
    }

    return {std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, cb_output, cores, num_slices}};
}

void InterleavedToShardedPartialProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const InterleavedToShardedPartialParams& params,
    const Tensor& input_tensor,
    Tensor& output_tensor) {
    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    uint32_t num_slices = params.num_slices;
    uint32_t slice_index = params.slice_index;
    bool partial_op = num_slices > 1;
    uint32_t starting_idx_h = 0;
    if (partial_op) {
        uint32_t runtime_slice_index = slice_index;
        starting_idx_h =
            operations::data_movement::detail::calculate_starting_idx_h(input_tensor, num_slices, runtime_slice_index);
    }

    auto& shared_variables = cached_program.shared_variables;
    auto& program = cached_program.program;
    auto unary_reader_kernel_id = shared_variables.unary_reader_kernel_id;
    auto unary_writer_kernel_id = shared_variables.unary_writer_kernel_id;
    auto cb_output = shared_variables.cb_output;
    const auto& cores = shared_variables.cores;

    auto& runtime_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
    for (const auto& core : cores) {
        auto& runtime_args = runtime_args_by_core[core.x][core.y];
        runtime_args[0] = src_buffer->address();
        if (partial_op) {
            runtime_args[7] = starting_idx_h;
        }
    }

    if (dst_is_dram) {
        auto& runtime_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);
        for (const auto& core : cores) {
            auto& runtime_args = runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_buffer->address();
        }
    } else {
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    }
}

}  // namespace ttnn::prim
