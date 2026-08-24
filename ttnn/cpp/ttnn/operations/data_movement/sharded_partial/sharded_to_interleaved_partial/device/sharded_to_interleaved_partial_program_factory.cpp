// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_program_factory.hpp"

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

// Anonymous-namespace helper unique to sharded_to_interleaved_partial to avoid unity-build collisions.
void push_s2i_partial_cb_pair(
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

ProgramDescriptor ShardedToInterleavedPartialProgramFactory::create_descriptor(
    const ShardedToInterleavedPartialParams& operation_attributes,
    const ShardedToInterleavedPartialInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    const uint32_t num_slices = operation_attributes.num_slices;
    const uint32_t slice_index = operation_attributes.slice_index;
    const bool is_l1_aligned = true;

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout();
    const bool is_tile = input.layout() == Layout::TILE;
    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto& all_cores = shard_spec.grid;

    // Extents in destination units: tiles for tile layout, sticks x bytes for row-major. A row-major
    // page is one logical row, so its extent comes from the logical shape, not the padded one.
    uint32_t input_unit_size = 0;
    uint32_t output_unit_size = 0;
    uint32_t tensor_h = 0;
    uint32_t tensor_w = 0;
    uint32_t shard_h = 0;
    uint32_t shard_w = 0;
    uint32_t num_units_per_shard = 0;
    if (is_tile) {
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        tensor_h = (input.physical_volume() / input.padded_shape()[-1]) / TILE_HEIGHT;
        tensor_w = input.padded_shape()[-1] / TILE_WIDTH;
        shard_h = shard_spec.shape[0] / TILE_HEIGHT;
        shard_w = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = shard_h * shard_w;
    } else {
        input_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * input.element_size());
        output_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * output.element_size());
        tensor_h = static_cast<uint32_t>(input.logical_volume() / input.logical_shape()[-1]);
        tensor_w = static_cast<uint32_t>(input.logical_shape()[-1] * input.element_size());
        shard_h = shard_spec.shape[0];
        shard_w = output_unit_size;
        num_units_per_shard = shard_h;
    }

    const uint32_t height_shards = div_up(tensor_h, shard_h);
    const uint32_t width_shards = div_up(tensor_w, shard_w);
    const uint32_t num_active_cores = height_shards * width_shards;

    // A grid provisioned wider than the data leaves cores holding only padding; leave them out.
    const CoreCoord grid_origin = all_cores.bounding_box().start_coord;
    CoreRangeSet used_cores;
    if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
        const uint32_t grid_x = rm_orientation ? width_shards : height_shards;
        const uint32_t grid_y = rm_orientation ? height_shards : width_shards;
        used_cores =
            CoreRangeSet(CoreRange(grid_origin, CoreCoord{grid_origin.x + grid_x - 1, grid_origin.y + grid_y - 1}));
    } else {
        used_cores = num_active_cores < all_cores.num_cores()
                         ? select_from_corerangeset(all_cores, 0, num_active_cores - 1, rm_orientation)
                         : all_cores;
    }
    const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);

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
    push_s2i_partial_cb_pair(
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
        push_s2i_partial_cb_pair(
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
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    reader_desc.compile_time_args = {src0_cb_index};

    // Writer kernel (writes interleaved output to DRAM).
    KernelDescriptor writer_desc;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = used_cores;
    writer_desc.config = WriterConfigDescriptor{};
    std::vector<uint32_t> writer_compile_time_args = {out_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    if (is_tile) {
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

    // Source stride: the shard's row pitch in L1, which stays the full shard width.
    uint32_t padded_shard_width = tt::align(output_unit_size, dst_buffer->alignment());
    if (is_blackhole or is_l1_aligned) {
        if (!dst_is_dram or is_l1_aligned) {
            padded_shard_width = tt::align(output_unit_size, hal::get_l1_alignment());
        }
    }

    for (uint32_t sh = 0; sh < height_shards; sh++) {
        for (uint32_t sw = 0; sw < width_shards; sw++) {
            // Height sharding has a single column and width sharding a single row, so one index is 0.
            const CoreCoord core =
                shard_strategy == TensorMemoryLayout::BLOCK_SHARDED
                    ? CoreCoord{grid_origin.x + (rm_orientation ? sw : sh), grid_origin.y + (rm_orientation ? sh : sw)}
                    : cores[sh + sw];
            const uint32_t h0 = sh * shard_h;
            const uint32_t w0 = sw * shard_w;
            // Clipping to the tensor is what keeps the write inside the destination page.
            const uint32_t shard_height = std::min(shard_h, tensor_h - h0);
            const uint32_t shard_width = std::min(shard_w, tensor_w - w0);

            // Arg 0 is the destination-buffer base address (binding via Buffer*).
            KernelDescriptor::RTArgList writer_rt;
            writer_rt.push_back(dst_buffer);
            if (is_tile) {
                writer_rt.push_back(shard_h);
                writer_rt.push_back(shard_w);
                writer_rt.push_back(shard_height);
                writer_rt.push_back(shard_width);
                writer_rt.push_back(tensor_w);
                writer_rt.push_back(num_units_per_shard);
                writer_rt.push_back(h0 * tensor_w + w0);
                writer_rt.push_back(starting_idx_h);
            } else {
                writer_rt.push_back(tensor_w);
                writer_rt.push_back(shard_height);
                writer_rt.push_back(shard_width);
                writer_rt.push_back(padded_shard_width);
                writer_rt.push_back(w0);
                writer_rt.push_back(h0);
            }
            writer_desc.emplace_runtime_args(core, writer_rt);
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
