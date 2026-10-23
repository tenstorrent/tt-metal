// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>

#include "moe_routing_remap_device_operation.hpp"

namespace {

uint32_t compute_weight_count_offset(
    const ttnn::MeshCoordinate& mesh_coordinate, uint32_t cluster_axis, uint32_t non_zero_per_device) {
    if (cluster_axis == 0) {
        return mesh_coordinate[0] * non_zero_per_device;
    }
    if (cluster_axis == 1) {
        return mesh_coordinate[1] * non_zero_per_device;
    }
    TT_THROW("Unsupported cluster axis");
    return 0;
}
}  // unnamed namespace

namespace ttnn::operations::data_movement {

using namespace tt::tt_metal;

ProgramDescriptor MoeRoutingRemapDeviceOperation::SingleCore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& routing_weights = tensor_args.input_routing_weights;
    const auto non_zero_weight_size = operation_attributes.non_zero_weight_size;
    const auto expert_parallel_size = operation_attributes.expert_parallel_size;
    const auto cluster_axis = operation_attributes.cluster_axis;
    const auto num_cluster_experts = routing_weights.logical_shape()[-1];
    const uint32_t non_zero_per_device = non_zero_weight_size / expert_parallel_size;

    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    const auto routing_weight_page_size_bytes = routing_weights.tensor_spec().compute_page_size_bytes();
    const auto aligned_routing_weight_page_size_bytes = tt::align(routing_weight_page_size_bytes, l1_alignment);

    // single core is fine
    const CoreRangeSet total_cores{CoreRange{{0, 0}, {0, 0}}};
    const CoreCoord utilized_core{0, 0};

    ProgramDescriptor desc;

    // input routing weight buffer
    const auto routing_weights_cb_id = tt::CBIndex::c_0;
    const auto routing_weights_format = datatype_to_dataformat_converter(routing_weights.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_routing_weight_page_size_bytes,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(routing_weights_cb_id),
            .data_format = routing_weights_format,
            .page_size = aligned_routing_weight_page_size_bytes,
        }}},
    });

    // store indices of per device non-zero weights
    const auto local_weights_idxs_cb_id = tt::CBIndex::c_1;
    using local_weights_idxs_t = uint16_t;
    const auto aligned_local_weights_idxs_page_size_bytes =
        tt::align(non_zero_per_device * sizeof(local_weights_idxs_t), l1_alignment);
    const auto local_weights_idxs_dataformat =
        datatype_to_dataformat_converter(tt::tt_metal::convert_to_data_type<local_weights_idxs_t>());
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_local_weights_idxs_page_size_bytes,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(local_weights_idxs_cb_id),
            .data_format = local_weights_idxs_dataformat,
            .page_size = aligned_local_weights_idxs_page_size_bytes,
        }}},
    });

    // output routing weight buffer
    const auto local_weights_cb_id = tt::CBIndex::c_2;
    const auto local_weights_page_size_bytes = tensor_return_value.tensor_spec().compute_page_size_bytes();
    const auto aligned_local_weights_page_size_bytes = tt::align(local_weights_page_size_bytes, l1_alignment);
    // this actually needs to be the same datatype as the input. Also checked `validate`
    const auto local_weights_format = datatype_to_dataformat_converter(tensor_return_value.dtype());

    TT_FATAL(local_weights_format == routing_weights_format, "Input and output datatypes need to be the same");

    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_local_weights_page_size_bytes,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(local_weights_cb_id),
            .data_format = local_weights_format,
            .page_size = aligned_local_weights_page_size_bytes,
        }}},
    });

    auto* const routing_weights_buffer = routing_weights.buffer();
    auto* const local_weights_buffer = tensor_return_value.buffer();

    const auto input_datum_size_bytes = tt::datum_size(local_weights_format);
    std::vector<uint32_t> reader_ct_args = {
        routing_weights_cb_id,
        local_weights_idxs_cb_id,
        num_cluster_experts,
        non_zero_per_device,
        input_datum_size_bytes};
    TensorAccessorArgs(*routing_weights_buffer).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/moe_routing_remap/device/kernels/dataflow/reader_moe_routing_remap.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = total_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct_args = {
        routing_weights_cb_id,
        local_weights_idxs_cb_id,
        local_weights_cb_id,
        num_cluster_experts,
        non_zero_per_device,
        input_datum_size_bytes};
    TensorAccessorArgs(*local_weights_buffer).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/moe_routing_remap/device/kernels/dataflow/"
        "writer_moe_routing_remap.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = total_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    // device_weights_count_offset is per-coord; baked into each program in the
    // mesh workload (one descriptor per coord). Cache hits patch buffer addresses
    // via BufferBinding and keep the per-coord offset intact.
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "MoeRoutingRemap requires a mesh dispatch coordinate");
    const auto device_weights_count_offset =
        compute_weight_count_offset(*mesh_dispatch_coordinate, cluster_axis, non_zero_per_device);

    // Buffer* triggers a binding entry; the framework patches the address on
    // cache hits without rebuilding the descriptor.
    reader_desc.emplace_runtime_args(utilized_core, {routing_weights_buffer, device_weights_count_offset});
    writer_desc.emplace_runtime_args(utilized_core, {local_weights_buffer});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::data_movement
