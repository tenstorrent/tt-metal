// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
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

MoeRoutingRemapDeviceOperation::SingleCore::cached_mesh_workload_t
MoeRoutingRemapDeviceOperation::SingleCore::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, cached_program.shared_variables);
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<MoeRoutingRemapDeviceOperation::SingleCore::shared_variables_t>
MoeRoutingRemapDeviceOperation::SingleCore::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto routing_weights = tensor_args.input_routing_weights;
    const auto non_zero_weight_size = operation_attributes.non_zero_weight_size;
    const auto expert_parallel_size = operation_attributes.expert_parallel_size;
    const auto cluster_axis = operation_attributes.cluster_axis;
    const auto num_cluster_experts = routing_weights.logical_shape()[-1];
    const uint32_t non_zero_per_device = non_zero_weight_size / expert_parallel_size;

    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    const auto routing_weight_page_size_bytes = routing_weights.tensor_spec().compute_page_size_bytes();
    const auto aligned_routing_weight_page_size_bytes = tt::align(routing_weight_page_size_bytes, l1_alignment);

    // single core is fine
    CoreRange total_cores({0, 0}, {0, 0});
    Program program{};

    using tt::tt_metal::CircularBufferConfig;

    // input routing weight buffer
    const auto routing_weights_cb_id = tt::CBIndex::c_0;
    const auto routing_weights_format = datatype_to_dataformat_converter(routing_weights.dtype());
    CircularBufferConfig cb_routing_weights_config =
        CircularBufferConfig(aligned_routing_weight_page_size_bytes, {{routing_weights_cb_id, routing_weights_format}})
            .set_page_size(routing_weights_cb_id, aligned_routing_weight_page_size_bytes);
    CreateCircularBuffer(program, total_cores, cb_routing_weights_config);

    // store indices of per device non-zero weights
    const auto local_weights_idxs_cb_id = tt::CBIndex::c_1;
    using local_weights_idxs_t = uint16_t;
    const auto aligned_local_weights_idxs_page_size_bytes =
        tt::align(non_zero_per_device * sizeof(local_weights_idxs_t), l1_alignment);
    const auto local_weights_idxs_dataformat =
        datatype_to_dataformat_converter(tt::tt_metal::convert_to_data_type<local_weights_idxs_t>());
    CircularBufferConfig cb_local_weights_idxs_config =
        CircularBufferConfig(
            aligned_local_weights_idxs_page_size_bytes, {{local_weights_idxs_cb_id, local_weights_idxs_dataformat}})
            .set_page_size(local_weights_idxs_cb_id, aligned_local_weights_idxs_page_size_bytes);
    CreateCircularBuffer(program, total_cores, cb_local_weights_idxs_config);

    // output routing weight buffer
    const auto local_weights_cb_id = tt::CBIndex::c_2;
    const auto local_weights_page_size_bytes = tensor_return_value.tensor_spec().compute_page_size_bytes();
    const auto aligned_local_weights_page_size_bytes = tt::align(local_weights_page_size_bytes, l1_alignment);
    // this actually needs to be the same datatype as the input. Also checked `validate`
    const auto local_weights_format = datatype_to_dataformat_converter(tensor_return_value.dtype());

    TT_FATAL(local_weights_format == routing_weights_format, "Input and output datatypes need to be the same");

    CircularBufferConfig cb_local_weights_config =
        CircularBufferConfig(aligned_local_weights_page_size_bytes, {{local_weights_cb_id, local_weights_format}})
            .set_page_size(local_weights_cb_id, aligned_local_weights_page_size_bytes);
    CreateCircularBuffer(program, total_cores, cb_local_weights_config);

    const auto input_datum_size_bytes = tt::datum_size(local_weights_format);
    std::vector<uint32_t> reader_ct_args = {
        routing_weights_cb_id,
        local_weights_idxs_cb_id,
        num_cluster_experts,
        non_zero_per_device,
        input_datum_size_bytes};
    tt::tt_metal::TensorAccessorArgs(*routing_weights.buffer()).append_to(reader_ct_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/moe_routing_remap/device/kernels/dataflow/reader_moe_routing_remap.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> writer_ct_args = {
        routing_weights_cb_id,
        local_weights_idxs_cb_id,
        local_weights_cb_id,
        num_cluster_experts,
        non_zero_per_device,
        input_datum_size_bytes};
    tt::tt_metal::TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_ct_args);
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/moe_routing_remap/device/kernels/dataflow/"
        "writer_moe_routing_remap.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    const auto routing_weights_addr = routing_weights.buffer()->address();
    const auto local_weights_addr = tensor_return_value.buffer()->address();

    const auto device_weights_count_offset =
        compute_weight_count_offset(mesh_coordinate, cluster_axis, non_zero_per_device);

    constexpr auto num_reader_rt_args = 2, num_writer_rt_args = 1;
    const std::array<uint32_t, num_reader_rt_args> reader_runtime_args = {
        routing_weights_addr, device_weights_count_offset};
    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, *total_cores.begin(), reader_runtime_args);

    const std::array<uint32_t, num_writer_rt_args> writer_runtime_args = {local_weights_addr};
    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, *total_cores.begin(), writer_runtime_args);

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .utilized_core = *total_cores.begin()}};
}

void MoeRoutingRemapDeviceOperation::SingleCore::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& coord = range.start_coord();
        TT_FATAL(
            coord == range.end_coord(),
            "Expected single coordinate per program but got range of {} to {}",
            coord,
            range.end_coord());

        const auto& shared_variables = cached_workload.shared_variables.at(range);

        const auto& unary_reader_kernel_id = shared_variables.unary_reader_kernel_id;
        const auto& unary_writer_kernel_id = shared_variables.unary_writer_kernel_id;
        const auto& utilized_core = shared_variables.utilized_core;

        auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, utilized_core);
        auto& writer_runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, utilized_core);

        reader_runtime_args.at(0) = tensor_args.input_routing_weights.buffer()->address();
        writer_runtime_args.at(0) = tensor_return_value.buffer()->address();
    }
}
}  // namespace ttnn::operations::data_movement
