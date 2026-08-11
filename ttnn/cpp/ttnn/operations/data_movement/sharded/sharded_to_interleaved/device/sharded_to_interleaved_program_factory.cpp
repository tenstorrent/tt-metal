// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Anonymous-namespace helper unique to sharded_to_interleaved to avoid unity-build collisions.
void push_s2i_cb_pair(
    ProgramDescriptor& desc,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t total_size,
    uint32_t page_size,
    const CoreRangeSet& core_ranges,
    Buffer* bound_buffer) {
    CBDescriptor cb;
    cb.total_size = total_size;
    cb.core_ranges = core_ranges;
    cb.format_descriptors.push_back(CBFormatDescriptor{
        .buffer_index = static_cast<uint8_t>(cb_index),
        .data_format = data_format,
        .page_size = page_size,
    });
    cb.buffer = bound_buffer;
    desc.cbs.push_back(std::move(cb));
}

}  // namespace

ProgramDescriptor ShardedToInterleavedProgramFactory::create_descriptor(
    const ShardedToInterleavedParams& operation_attributes,
    const ShardedToInterleavedInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    const uint32_t num_slices = operation_attributes.num_slices;
    const uint32_t slice_index = operation_attributes.slice_index;
    const bool is_l1_aligned = true;

    uint32_t num_units_per_shard = 0;
    uint32_t input_unit_size = 0;
    uint32_t output_unit_size = 0;
    uint32_t num_units_per_shard_width = 0;
    uint32_t num_units_per_shard_height = 0;
    uint32_t num_units_offset = 0;
    uint32_t num_units_per_row = 0;
    uint32_t num_units_height = 0;
    uint32_t num_units_per_shard_height_last = 0;
    uint32_t num_units_per_shard_width_last = 0;

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout();

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t num_cores_unpadded = num_cores;
    CoreCoord end_core;
    if (output.layout() == Layout::TILE) {
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.padded_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        num_units_height = (input.physical_volume() / input.padded_shape()[-1]) / TILE_HEIGHT;
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
    } else {
        input_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * input.element_size());
        output_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * output.element_size());
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = static_cast<uint32_t>(input.logical_shape()[-1] * input.element_size());
        num_units_offset = 1;
        num_units_height = static_cast<uint32_t>(input.logical_volume() / input.logical_shape()[-1]);
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
    }

    const uint32_t width_shards = output.layout() == Layout::TILE
                                      ? div_up(num_units_per_row, num_units_per_shard_width)
                                      : div_up(num_units_per_row, output_unit_size);
    const uint32_t height_shards = div_up(num_units_height, num_units_per_shard_height);

    // Restrict to the cores that actually hold data; a larger grid otherwise causes a NOC error.
    CoreRangeSet used_cores = all_cores;
    if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_cores_unpadded = height_shards;
    } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
        num_cores_unpadded = width_shards;
    } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED && all_cores.ranges().size() == 1) {
        // Block sharding needs a sub-rectangle, not a leading run of cores: a grid with more columns
        // than the data needs leaves whole columns holding only padding, and walking them wraps the
        // row cursor early so later cores overwrite the next band of rows.
        const auto& bbox = all_cores.ranges()[0];
        const uint32_t grid_x = rm_orientation ? width_shards : height_shards;
        const uint32_t grid_y = rm_orientation ? height_shards : width_shards;
        used_cores = CoreRangeSet(CoreRange(
            bbox.start_coord, CoreCoord(bbox.start_coord.x + grid_x - 1, bbox.start_coord.y + grid_y - 1)));
        num_cores_unpadded = grid_x * grid_y;
    }
    if (used_cores == all_cores && num_cores_unpadded < num_cores) {
        used_cores = select_from_corerangeset(all_cores, 0, num_cores_unpadded - 1, rm_orientation);
    }

    const auto active_cores = corerange_to_cores(used_cores, std::nullopt, rm_orientation);
    num_cores_unpadded = static_cast<uint32_t>(active_cores.size());
    end_core = active_cores.back();

    bool convert_df = input_cb_data_format != output_cb_data_format;

    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t out_cb_index = src0_cb_index;
    uint32_t num_input_units = num_units_per_shard;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();
    uint32_t input_page_size = tt::align(input_unit_size, src_buffer->alignment());
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);

    ProgramDescriptor desc;

    // Sharded input CB. Bind to src buffer for dynamic-CB rebinding on cache hits via cb.buffer.
    push_s2i_cb_pair(
        desc,
        src0_cb_index,
        input_cb_data_format,
        num_input_units * input_page_size,
        input_page_size,
        used_cores,
        /*bound_buffer=*/src_buffer);

    if (convert_df) {
        out_cb_index = CBIndex::c_16;
        uint32_t output_page_size = tt::align(output_unit_size, dst_buffer->alignment());
        push_s2i_cb_pair(
            desc,
            out_cb_index,
            output_cb_data_format,
            num_input_units * output_page_size,
            output_page_size,
            used_cores,
            /*bound_buffer=*/nullptr);
    }

    // Reader kernel (sharded input streamed in via globally-allocated CB).
    KernelDescriptor reader_desc;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = used_cores;
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    reader_desc.compile_time_args = {src0_cb_index};

    // Writer kernel (writes interleaved output to DRAM).
    KernelDescriptor writer_desc;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = used_cores;
    writer_desc.config = WriterConfigDescriptor{};
    std::vector<uint32_t> writer_compile_time_args = {out_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    if (input.layout() == Layout::TILE) {
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "writer_unary_sharded_blocks_interleaved_start_id.cpp";
    } else {
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp";
    }
    writer_desc.compile_time_args = std::move(writer_compile_time_args);

    // Optional compute kernel for data-format conversion.
    KernelDescriptor compute_desc;
    if (convert_df) {
        compute_desc.kernel_source = "ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = used_cores;
        compute_desc.config = ComputeConfigDescriptor{};
        compute_desc.compile_time_args = {num_units_per_shard};
    }

    // Reader runtime args: identical on every used core.
    for (const auto& core_range : used_cores.ranges()) {
        for (const auto& core : core_range) {
            reader_desc.emplace_runtime_args(core, {num_units_per_shard});
        }
    }

    uint32_t starting_idx_h =
        operations::data_movement::detail::calculate_starting_idx_h(output, num_slices, slice_index);
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;

    for (uint32_t core_idx = 0; core_idx < num_cores_unpadded; core_idx++) {
        const auto& core = active_cores[core_idx];
        if (input.layout() == Layout::TILE) {
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
            // Writer run-time args: arg 0 is the destination-buffer base address (binding via Buffer*).
            KernelDescriptor::RTArgList writer_rt;
            writer_rt.push_back(dst_buffer);
            writer_rt.push_back(num_units_per_shard_height);
            writer_rt.push_back(num_units_per_shard_width);
            writer_rt.push_back(shard_height);
            writer_rt.push_back(shard_width);
            writer_rt.push_back(num_units_offset);
            writer_rt.push_back(num_units_per_shard);
            writer_rt.push_back(curr_idx_h + curr_idx_w);
            writer_rt.push_back(starting_idx_h);
            writer_desc.emplace_runtime_args(core, writer_rt);

            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            uint32_t shard_height = num_units_per_shard_height;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            // A shard can be wider than the logical row (e.g. pool pads channels up to a tile).
            const uint32_t shard_width = std::min(output_unit_size, num_units_per_row - curr_idx_w);

            uint32_t l1_alignment = hal::get_l1_alignment();
            uint32_t padded_shard_width = tt::align(output_unit_size, dst_buffer->alignment());
            if (is_blackhole or is_l1_aligned) {
                if (!dst_is_dram or is_l1_aligned) {
                    padded_shard_width = tt::align(output_unit_size, l1_alignment);
                }
            }
            // Writer run-time args: arg 0 is the destination-buffer base address (binding via Buffer*).
            KernelDescriptor::RTArgList writer_rt;
            writer_rt.push_back(dst_buffer);
            writer_rt.push_back(num_units_per_row);
            writer_rt.push_back(shard_height);
            writer_rt.push_back(shard_width);
            writer_rt.push_back(padded_shard_width);
            writer_rt.push_back(curr_idx_w);
            writer_rt.push_back(curr_idx_h);
            writer_desc.emplace_runtime_args(core, writer_rt);

            curr_idx_w += output_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    if (convert_df) {
        desc.kernels.push_back(std::move(compute_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
