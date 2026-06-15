// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "full_program_factory_common.hpp"
#include "full_program_factory_sharded.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::full {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

ProgramDescriptor FullShardedProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    auto fill_value = operation_attributes.fill_value;
    DataType dtype{operation_attributes.dtype};

    auto data_format = datatype_to_dataformat_converter(dtype);

    uint32_t tensor_width_in_pages = output.buffer()->shard_spec().tensor2d_shape_in_pages[1];

    std::vector<CoreCoord> runtime_cores = get_optimal_worker_cores_for_sharded_tensor(output);
    const auto& compute_core_range = CoreRangeSet(ttsl::Span<const CoreCoord>(runtime_cores));

    const auto& aligned_page_size = output.buffer()->aligned_page_size();
    const auto& page_size = output.buffer()->page_size();

    constexpr CBIndex cb_fill_value_id = CBIndex::c_24;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = page_size,
        .core_ranges = compute_core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_fill_value_id),
            .data_format = data_format,
            .page_size = page_size,
        }}},
    });

    auto writer_defines = defines_from_map(get_writer_defines(dtype));
    auto u = encode_fill_value(fill_value, dtype);

    uint32_t elems_per_page = page_size / datum_size(data_format);
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)cb_fill_value_id, elems_per_page, page_size, aligned_page_size, tensor_width_in_pages};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = compute_core_range;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t shard_height_in_pages = output.buffer()->shard_spec().shape_in_pages()[0];
    uint32_t shard_width_in_pages = output.buffer()->shard_spec().shape_in_pages()[1];
    uint32_t tensor_height_in_pages = output.buffer()->shard_spec().tensor2d_shape_in_pages[0];
    uint32_t num_shards_across_width = tt::div_up(tensor_width_in_pages, shard_width_in_pages);
    uint32_t num_shards_across_height = tt::div_up(tensor_height_in_pages, shard_height_in_pages);

    for (uint32_t i = 0; i < runtime_cores.size(); i++) {
        const auto& core = runtime_cores[i];

        uint32_t shard_row_idx = i / num_shards_across_width;
        uint32_t shard_col_idx = i % num_shards_across_width;

        uint32_t first_page_id =
            (shard_row_idx * shard_height_in_pages * tensor_width_in_pages) + (shard_col_idx * shard_width_in_pages);

        uint32_t valid_pages_width = (shard_col_idx == num_shards_across_width - 1)
                                         ? (tensor_width_in_pages - (shard_col_idx * shard_width_in_pages))
                                         : shard_width_in_pages;

        uint32_t valid_pages_height = (shard_row_idx == num_shards_across_height - 1)
                                          ? (tensor_height_in_pages - (shard_row_idx * shard_height_in_pages))
                                          : shard_height_in_pages;
        writer_desc.emplace_runtime_args(
            core, {output.buffer(), u.u32, first_page_id, valid_pages_width, valid_pages_height});
    }

    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::full
