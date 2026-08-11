// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Anonymous-namespace helper unique to interleaved_to_sharded to avoid unity-build collisions.
void push_i2s_cb_pair(
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

// Hardcoded for non-partial interleaved_to_sharded operation
// to keep backward compatibility after migration to new infra
// https://github.com/tenstorrent/tt-metal/issues/32752
constexpr uint32_t num_slices = 1;
constexpr uint32_t slice_index = 0;

ProgramDescriptor InterleavedToShardedProgramFactory::create_descriptor(
    const InterleavedToShardedParams& /*operation_attributes*/,
    const InterleavedToShardedInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    // Keep explicit bool init to match legacy behavior which forced it true
    bool keep_l1_aligned = true;  // operation_attributes.keep_l1_aligned;

    uint32_t num_units_per_shard = 0;
    uint32_t input_unit_size = 0;
    uint32_t output_unit_size = 0;
    uint32_t num_units_per_shard_width = 0;
    uint32_t num_units_per_shard_height = 0;
    uint32_t num_units_offset = 0;
    uint32_t num_units_per_row = 0;
    uint32_t num_units_per_shard_height_last = 0;
    uint32_t num_units_per_shard_width_last = 0;
    uint32_t padded_offset_bytes = 0;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = output.shard_spec().value();
    auto shard_strategy = output.memory_config().memory_layout();

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    auto cores = get_optimal_worker_cores_for_sharded_tensor(output);
    auto all_cores = CoreRangeSet(ttsl::Span<const CoreCoord>(cores));
    CoreCoord end_core = cores.back();

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
        uint32_t num_units_height = (input.physical_volume() / input.padded_shape()[-1]) / TILE_HEIGHT;
        num_units_per_shard_height_last =
            num_units_per_shard_height -
            (tt::round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width -
            (tt::round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
        padded_offset_bytes = (num_units_per_shard_width - num_units_per_shard_width_last) * input_unit_size;
    } else {
        input_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * input.element_size());
        output_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * output.element_size());
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = static_cast<uint32_t>(input.logical_shape()[-1] * input.element_size());
        num_units_offset = 1;
        uint32_t num_units_height = static_cast<uint32_t>(input.logical_volume() / input.logical_shape()[-1]);
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

    uint32_t input_cb_index = tt::CBIndex::c_0;
    uint32_t scratch_cb_index = tt::CBIndex::c_1;
    uint32_t out_cb_index = input_cb_index;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t output_page_size = tt::align(output_unit_size, dst_buffer->alignment());

    ProgramDescriptor desc;

    if (convert_df) {
        out_cb_index = tt::CBIndex::c_16;
        uint32_t input_page_size = tt::align(input_unit_size, src_buffer->alignment());
        // Non-globally-allocated input CB (interleaved input streamed via reader).
        push_i2s_cb_pair(
            desc,
            input_cb_index,
            input_cb_data_format,
            num_input_units * input_page_size,
            input_page_size,
            all_cores,
            /*bound_buffer=*/nullptr);
    }

    // Output CB. When destination is sharded (non-DRAM) we bind it to the output buffer
    // for dynamic-CB rebinding on cache hits via cb.buffer. When dst is DRAM, no binding.
    push_i2s_cb_pair(
        desc,
        out_cb_index,
        output_cb_data_format,
        num_input_units * output_page_size,
        output_page_size,
        all_cores,
        /*bound_buffer=*/dst_is_dram ? nullptr : dst_buffer);

    uint32_t dram_alignment = hal::get_dram_alignment();
    uint32_t l1_alignment = hal::get_l1_alignment();
    uint32_t num_trids = 4;
    if ((src_is_dram && (input_unit_size % dram_alignment != 0)) || is_blackhole || keep_l1_aligned) {
        // scratchpad going to be used to align DRAM (64B) to L1 (16B)
        // This is done to mitigate the alignment issues.
        // See issue #34414.
        uint32_t scratch_cb_page_size = tt::align(input_unit_size + dram_alignment, dram_alignment);
        push_i2s_cb_pair(
            desc,
            scratch_cb_index,
            input_cb_data_format,
            num_trids * scratch_cb_page_size,
            scratch_cb_page_size,
            all_cores,
            /*bound_buffer=*/nullptr);
    }

    // Reader kernel.
    KernelDescriptor reader_desc;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.config = ReaderConfigDescriptor{};
    if (input.layout() == Layout::TILE) {
        std::vector<uint32_t> reader_compile_time_args = {input_cb_index, all_cores.num_cores()};
        tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "reader_unary_sharded_blocks_interleaved_start_id.cpp";
        reader_desc.compile_time_args = std::move(reader_compile_time_args);
    } else {
        std::vector<uint32_t> reader_compile_time_args = {input_cb_index, scratch_cb_index, num_trids};
        tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp";
        reader_desc.compile_time_args = std::move(reader_compile_time_args);
    }

    // Writer kernel.
    KernelDescriptor writer_desc;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.config = WriterConfigDescriptor{};
    std::vector<uint32_t> writer_compile_time_args = {out_cb_index};
    if (dst_is_dram) {
        if (input.layout() == Layout::TILE) {
            writer_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                "writer_unary_sharded_blocks_start_id.cpp";
        } else {
            writer_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                "writer_unary_sharded_stick_layout_start_id.cpp";
        }
        TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    } else {
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp";
    }
    writer_desc.compile_time_args = std::move(writer_compile_time_args);

    // Optional compute kernel for data-format conversion.
    KernelDescriptor compute_desc;
    if (convert_df) {
        compute_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = all_cores;
        compute_desc.config = ComputeConfigDescriptor{};
    }

    uint32_t starting_idx_h =
        operations::data_movement::detail::calculate_starting_idx_h(input, num_slices, slice_index);
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;

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

            // Reader run-time args: arg 0 is the source-buffer base address (binding).
            KernelDescriptor::RTArgList reader_rt;
            reader_rt.push_back(src_buffer);
            reader_rt.push_back(shard_height);
            reader_rt.push_back(shard_width);
            reader_rt.push_back(padded_offset);
            reader_rt.push_back(num_units_offset);
            reader_rt.push_back(curr_num_units_per_shard);
            reader_rt.push_back(curr_idx_h + curr_idx_w);
            reader_rt.push_back(starting_idx_h);
            reader_desc.emplace_runtime_args(core, reader_rt);

            // Writer run-time args
            uint32_t pad_offset = (num_units_per_shard_width - shard_width) * output_unit_size;
            if (dst_is_dram) {
                KernelDescriptor::RTArgList writer_rt;
                writer_rt.push_back(dst_buffer);
                writer_rt.push_back(shard_height);
                writer_rt.push_back(shard_width);
                writer_rt.push_back(pad_offset);
                writer_rt.push_back(curr_num_units_per_shard);
                writer_rt.push_back(num_units_offset);
                writer_rt.push_back(curr_idx_h + curr_idx_w);
                writer_rt.push_back(starting_idx_h);
                writer_desc.emplace_runtime_args(core, writer_rt);
            } else {
                writer_desc.emplace_runtime_args(core, {curr_num_units_per_shard});
            }

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

            bool aligned = false;
            if (src_is_dram) {
                aligned = (curr_idx_w % dram_alignment == 0) && (padded_offset_bytes % dram_alignment == 0);
            } else if (is_blackhole) {
                aligned = (curr_idx_w % l1_alignment == 0) && (padded_offset_bytes % l1_alignment == 0);
            } else {
                aligned = true;
            }
            uint32_t aligned_width_offset = 0;
            uint32_t aligned_shard_width = 0;
            uint32_t aligned_offset = 0;
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

            // Reader run-time args: arg 0 is the source-buffer base address (binding).
            KernelDescriptor::RTArgList reader_rt;
            reader_rt.push_back(src_buffer);
            reader_rt.push_back(num_units_per_row);
            reader_rt.push_back(shard_height);
            reader_rt.push_back(shard_width);
            reader_rt.push_back(padded_offset_bytes);
            reader_rt.push_back(static_cast<uint32_t>(aligned));
            reader_rt.push_back(aligned_width_offset);
            reader_rt.push_back(aligned_shard_width);
            reader_rt.push_back(aligned_offset);
            reader_rt.push_back(curr_idx_h);
            reader_desc.emplace_runtime_args(core, reader_rt);

            // Writer run-time args
            if (dst_is_dram) {
                uint32_t page_id_within_row = curr_idx_w / input_unit_size;
                uint32_t output_width_in_pages = tt::div_up(num_units_per_row, input_unit_size);
                uint32_t start_id = (curr_idx_h * output_width_in_pages) + page_id_within_row;
                KernelDescriptor::RTArgList writer_rt;
                writer_rt.push_back(dst_buffer);
                writer_rt.push_back(shard_height);
                writer_rt.push_back(shard_width);
                writer_rt.push_back(padded_offset_bytes);
                writer_rt.push_back(start_id);
                writer_rt.push_back(output_width_in_pages);
                writer_desc.emplace_runtime_args(core, writer_rt);
            } else {
                writer_desc.emplace_runtime_args(core, {curr_num_units_per_shard});
            }

            // Update indexing
            curr_idx_w += input_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
        if (convert_df) {
            compute_desc.emplace_runtime_args(core, {curr_num_units_per_shard});
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
