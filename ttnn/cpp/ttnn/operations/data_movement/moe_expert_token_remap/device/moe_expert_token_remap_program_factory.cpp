// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "moe_expert_token_remap_device_operation.hpp"

namespace ttnn::operations::data_movement {

using namespace tt::tt_metal;

ProgramDescriptor MoeExpertTokenRemapDeviceOperation::Multicore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& metadata_tensor = tensor_args.metadata_tensor;
    const auto& mapping_tensor = tensor_args.mapping_tensor;
    const auto& topk_tensor = tensor_args.topk_tensor;

    const auto& metadata_tensor_shape = metadata_tensor.logical_shape();
    const auto batch_size = metadata_tensor_shape[1];
    const auto seq_size = metadata_tensor_shape[2];
    const auto selected_experts_k = metadata_tensor_shape[3];
    const auto experts = mapping_tensor.logical_shape()[-2];

    const auto& output_mapping_tensor = tensor_return_value.at(0);
    const auto& output_reduced_tensor = tensor_return_value.at(1);

    const auto experts_per_device = output_mapping_tensor.logical_shape()[-1];

    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    const auto mapping_page_size_bytes = mapping_tensor.tensor_spec().compute_page_size_bytes();
    const auto aligned_mapping_page_size_bytes = tt::align(mapping_page_size_bytes, l1_alignment);

    const auto metadata_page_size_bytes = metadata_tensor.tensor_spec().compute_page_size_bytes();
    const auto aligned_metadata_page_size_bytes = tt::align(metadata_page_size_bytes, l1_alignment);

    const auto topk_page_size_bytes = topk_tensor.tensor_spec().compute_page_size_bytes();
    const auto aligned_topk_page_size_bytes = tt::align(topk_page_size_bytes, l1_alignment);

    const auto output_mapping_page_size_bytes = output_mapping_tensor.tensor_spec().compute_page_size_bytes();
    const auto output_reduced_page_size_bytes = output_reduced_tensor.tensor_spec().compute_page_size_bytes();

    ProgramDescriptor desc;

    // todo maybe, subdevice
    auto* mesh_device = topk_tensor.device();
    const auto grid = mesh_device->compute_with_storage_grid_size();
    // CoreCoord grid = {1,1};

    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // full mapping buffer
    const auto mapping_data_format = datatype_to_dataformat_converter(mapping_tensor.dtype());
    const auto mapping_tensor_cb_id = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_mapping_page_size_bytes,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(mapping_tensor_cb_id),
            .data_format = mapping_data_format,
            .page_size = aligned_mapping_page_size_bytes,
        }}},
    });

    // scratch space to store and share indices of per device experts
    const auto local_experts_cb_id = tt::CBIndex::c_1;
    using local_experts_t = uint16_t;
    const auto aligned_local_expert_page_size_bytes =
        tt::align(experts_per_device * sizeof(local_experts_t), l1_alignment);
    const auto local_experts_dataformat =
        datatype_to_dataformat_converter(tt::tt_metal::convert_to_data_type<local_experts_t>());
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_local_expert_page_size_bytes,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(local_experts_cb_id),
            .data_format = local_experts_dataformat,
            .page_size = aligned_local_expert_page_size_bytes,
        }}},
    });

    // metadata page buffer
    constexpr uint32_t metadata_buffer_factor = 1;
    const auto metadata_data_format = datatype_to_dataformat_converter(metadata_tensor.dtype());
    const auto metadata_cb_id = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = metadata_buffer_factor * aligned_metadata_page_size_bytes,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(metadata_cb_id),
            .data_format = metadata_data_format,
            .page_size = aligned_metadata_page_size_bytes,
        }}},
    });

    // topk page buffer
    constexpr uint32_t topk_buffer_factor = 2;
    const auto topk_data_format = datatype_to_dataformat_converter(topk_tensor.dtype());
    const auto topk_cb_id = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = topk_buffer_factor * aligned_topk_page_size_bytes,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(topk_cb_id),
            .data_format = topk_data_format,
            .page_size = aligned_topk_page_size_bytes,
        }}},
    });

    // output mapping staging buffer
    const auto output_mapping_data_format = datatype_to_dataformat_converter(output_mapping_tensor.dtype());
    const auto output_mapping_cb_id = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_mapping_page_size_bytes,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_mapping_cb_id),
            .data_format = output_mapping_data_format,
            .page_size = output_mapping_page_size_bytes,
        }}},
    });

    // output reduced staging buffer
    const auto output_reduced_data_format = datatype_to_dataformat_converter(output_reduced_tensor.dtype());
    const auto output_reduced_cb_id = tt::CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_reduced_page_size_bytes,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_reduced_cb_id),
            .data_format = output_reduced_data_format,
            .page_size = output_reduced_page_size_bytes,
        }}},
    });

    const auto& mesh_view = mesh_device->get_view();
    // Mesh dispatch coordinate is supplied by the framework via the per-coord
    // descriptor adapter path. Fall back to {0,0} if absent (single-device case).
    const auto coord = mesh_dispatch_coordinate.value_or(ttnn::MeshCoordinate{0, 0});
    const uint32_t flat_mesh_idx = (coord[0] * mesh_view.num_cols()) + coord[1];

    // slightly abusing this functionality since here we also have a single page if any experts are activated
    constexpr bool local_reduce = true;
    std::vector<uint32_t> reader_ct_args = {
        mapping_tensor_cb_id,
        local_experts_cb_id,
        metadata_cb_id,
        topk_cb_id,
        experts_per_device,
        batch_size,
        seq_size,
        experts,
        flat_mesh_idx,
        topk_page_size_bytes,
        selected_experts_k,
        mapping_page_size_bytes,
        metadata_page_size_bytes,
        local_reduce};
    TensorAccessorArgs(topk_tensor.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(mapping_tensor.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(metadata_tensor.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = total_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    const auto output_datum_size_bytes = tt::datum_size(output_mapping_data_format);
    const auto reduction_size = operation_attributes.reduction_size;
    std::vector<uint32_t> writer_ct_args = {
        local_experts_cb_id,
        metadata_cb_id,
        topk_cb_id,
        output_mapping_cb_id,
        output_reduced_cb_id,
        selected_experts_k,
        experts_per_device,
        output_mapping_page_size_bytes,
        output_datum_size_bytes,
        output_reduced_page_size_bytes,
        reduction_size,
    };
    TensorAccessorArgs(*output_mapping_tensor.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*output_reduced_tensor.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/moe_expert_token_remap/device/kernels/dataflow/"
        "writer_moe_expert_token_remap.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = total_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    // split work over metadata pages (batch*seq)
    const auto num_metadata_pages = metadata_tensor.buffer()->num_pages();

    const auto [core_page_increments, all_cores] =
        split_work_to_cores_even_multiples(grid, num_metadata_pages, reduction_size);

    Buffer* mapping_buffer = mapping_tensor.buffer();
    Buffer* metadata_buffer = metadata_tensor.buffer();
    Buffer* topk_buffer = topk_tensor.buffer();
    Buffer* output_mapping_buffer = output_mapping_tensor.buffer();
    Buffer* output_reduced_buffer = output_reduced_tensor.buffer();

    uint32_t page_idx_start = 0, page_idx_end = 0;
    std::vector<CoreCoord> utilized_cores = corerange_to_cores(all_cores, std::nullopt);
    TT_FATAL(utilized_cores.size() == core_page_increments.size(), "Internal error");

    auto cit = utilized_cores.begin();
    for (auto increment : core_page_increments) {
        page_idx_end += increment;
        // Buffer* entries (slots 0..2) trigger BufferBinding so the framework can
        // patch addresses on cache hits without rebuilding the descriptor.
        reader_desc.emplace_runtime_args(
            *cit, {mapping_buffer, metadata_buffer, topk_buffer, page_idx_start, page_idx_end});

        const uint32_t reduction_idx_start = page_idx_start / reduction_size;

        // Buffer* entries at slots 0 and 3 use BufferBinding for cache-hit address patching.
        writer_desc.emplace_runtime_args(
            *cit, {output_mapping_buffer, page_idx_start, page_idx_end, output_reduced_buffer, reduction_idx_start});

        page_idx_start += increment;
        ++cit;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::data_movement
