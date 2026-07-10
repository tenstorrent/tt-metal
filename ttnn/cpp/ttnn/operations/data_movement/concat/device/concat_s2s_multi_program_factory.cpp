// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2s_multi_program_factory.hpp"

#include <algorithm>
#include <numeric>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

namespace {

uint32_t find_greatest_common_page_size(std::vector<uint32_t>& stick_sizes, uint32_t alignment) {
    TT_FATAL(!stick_sizes.empty(), "Need at least one stick size to find page size");
    uint32_t page_size = tt::align(stick_sizes[0], alignment);
    for (size_t idx = 1; idx < stick_sizes.size(); idx++) {
        const uint32_t padded_stick_size = tt::align(stick_sizes[idx], alignment);
        page_size = std::gcd(page_size, padded_stick_size);
    }
    return page_size;
}

}  // namespace

tt::tt_metal::ProgramDescriptor ConcatS2SMultiProgramFactory::create_descriptor(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensors = tensor_args.input_tensors;
    Tensor& output = tensor_return_value;
    const bool is_height_concat = 2 == operation_attributes.dim;
    ProgramDescriptor desc;

    const uint32_t num_input_tensors = input_tensors.size();
    const uint32_t cb_dst_id = 16;
    TT_FATAL(num_input_tensors <= cb_dst_id, "Not enough circular buffer for {} inputs.", num_input_tensors);
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;

    // Assume inputs and output have the same element size and alignment.
    const uint32_t element_size = input_tensors[0].element_size();
    const uint32_t alignment = input_tensors[0].buffer()->alignment();

    uint32_t page_size;
    uint32_t elements_per_page_width;
    uint32_t elements_per_page_height;
    std::optional<Tile> tile = std::nullopt;
    if (rm_layout) {
        std::vector<uint32_t> all_stick_sizes;
        all_stick_sizes.reserve(input_tensors.size() + 1);
        all_stick_sizes.push_back(output.shard_spec().value().shape[1]);
        std::transform(
            input_tensors.begin(), input_tensors.end(), std::back_inserter(all_stick_sizes), [](const Tensor& tensor) {
                return tensor.element_size() * tensor.shard_spec().value().shape[1];
            });
        page_size = find_greatest_common_page_size(all_stick_sizes, alignment);
        elements_per_page_width = page_size / element_size;
        elements_per_page_height = 1;
    } else {
        tile = output.tensor_spec().tile();
        page_size = tile->get_tile_size(cb_data_format);
        elements_per_page_width = tile->get_width();
        elements_per_page_height = tile->get_height();
    }

    const std::optional<TileDescriptor> tile_descriptor =
        tile.has_value() ? std::optional<TileDescriptor>(TileDescriptor(tile.value())) : std::nullopt;

    std::vector<uint32_t> input_num_pages_per_stick;
    std::vector<uint32_t> input_num_sticks;
    std::vector<uint32_t> input_write_offsets;
    input_num_pages_per_stick.reserve(num_input_tensors);
    input_num_sticks.reserve(num_input_tensors);
    input_write_offsets.reserve(num_input_tensors);

    // Assume inputs and output have the same sharding grid.
    const auto all_cores = input_tensors[0].shard_spec().value().grid;

    // Input CBs
    uint32_t curr_input_write_offset = 0;
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        const auto shard_spec = input_tensors[input_id].shard_spec().value();
        input_num_pages_per_stick.push_back(tt::div_up(shard_spec.shape[1], elements_per_page_width));
        input_num_sticks.push_back(tt::div_up(shard_spec.shape[0], elements_per_page_height));
        input_write_offsets.push_back(curr_input_write_offset);

        const uint32_t input_num_pages = input_num_pages_per_stick[input_id] * input_num_sticks[input_id];
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_size * input_num_pages,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(input_id),
                .data_format = cb_data_format,
                .page_size = page_size,
                .tile = tile_descriptor,
            }}},
            .buffer = input_tensors[input_id].buffer(),
        });

        curr_input_write_offset +=
            page_size * (is_height_concat ? input_num_pages : input_num_pages_per_stick[input_id]);
    }

    // Output CB
    const auto output_shard_spec = output.shard_spec().value();
    const uint32_t output_num_pages_per_stick = tt::div_up(output_shard_spec.shape[1], elements_per_page_width);
    const uint32_t output_num_sticks = tt::div_up(output_shard_spec.shape[0], elements_per_page_height);
    desc.cbs.push_back(CBDescriptor{
        .total_size = page_size * output_num_sticks * output_num_pages_per_stick,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_dst_id),
            .data_format = cb_data_format,
            .page_size = page_size,
            .tile = tile_descriptor,
        }}},
        .buffer = output.buffer(),
    });

    const uint32_t output_stride = page_size * output_num_pages_per_stick;
    const KernelDescriptor::CompileTimeArgs compile_time_args = {
        cb_dst_id, page_size, output_stride, num_input_tensors};

    std::vector<uint32_t> runtime_args_0;
    std::vector<uint32_t> runtime_args_1;
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        const auto input_num_sticks_per_risc = tt::div_up(input_num_sticks[input_id], 2);
        runtime_args_0.push_back(input_num_pages_per_stick[input_id]);
        runtime_args_0.push_back(input_num_sticks_per_risc);
        runtime_args_0.push_back(input_write_offsets[input_id]);
        runtime_args_0.push_back(0);
        runtime_args_1.push_back(input_num_pages_per_stick[input_id]);
        runtime_args_1.push_back(input_num_sticks[input_id] - input_num_sticks_per_risc);
        runtime_args_1.push_back(input_write_offsets[input_id] + (output_stride * input_num_sticks_per_risc));
        runtime_args_1.push_back(page_size * input_num_pages_per_stick[input_id] * input_num_sticks_per_risc);
    }

    // Match the legacy CachedProgram path: SetRuntimeArgs(..., all_cores, args).
    // These values must live in per-core runtime args, not common_runtime_args.
    // BRISC (writer) and NCRISC (reader) each have their own RTA region; using
    // common_runtime_args for both kernels made both RISCs observe the same
    // offsets (precision failures in sharded concat).
    KernelDescriptor::CoreRuntimeArgs reader_rt_args(runtime_args_0.begin(), runtime_args_0.end());
    KernelDescriptor::CoreRuntimeArgs writer_rt_args(runtime_args_1.begin(), runtime_args_1.end());

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2s_tensor_concat.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = compile_time_args;
    reader_desc.runtime_args.reserve(all_cores.num_cores());
    for (const auto& range : all_cores.ranges()) {
        for (const CoreCoord& core : range) {
            reader_desc.runtime_args.emplace_back(core, reader_rt_args);
        }
    }
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2s_tensor_concat.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = compile_time_args;
    writer_desc.runtime_args.reserve(all_cores.num_cores());
    for (const auto& range : all_cores.ranges()) {
        for (const CoreCoord& core : range) {
            writer_desc.runtime_args.emplace_back(core, writer_rt_args);
        }
    }
    writer_desc.config = WriterConfigDescriptor{};

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim
