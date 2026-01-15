// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_async_device_operation.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ccl::neighbor_pad {

NeighborPadAsyncDeviceOperation::program_factory_t NeighborPadAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return NeighborPadAsyncMeshWorkloadFactory{};
}

void NeighborPadAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void NeighborPadAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(args.dim < 3, "Error, neighbor pad currently only supports padding non last dim, provided {}", args.dim);

    TT_FATAL(
        tensor_args.input_tensor.layout() == Layout::ROW_MAJOR,
        "Unsupported input tensor layout {}.",
        tensor_args.input_tensor.layout());

    TT_FATAL(!tensor_args.input_tensor.is_sharded(), "Neighbor pad does not support sharded input tensors.");

    TT_FATAL(
        args.padding_mode == "zeros" || args.padding_mode == "replicate",
        "Unsupported padding mode {}.",
        args.padding_mode);

    const auto& input_tensor_shape = tensor_args.input_tensor.padded_shape();
    TT_FATAL(
        args.padding_left <= input_tensor_shape[args.dim] && args.padding_right <= input_tensor_shape[args.dim],
        "One of the padding values {} or {} exceeds the shape of the input tensor in that dim {}.",
        args.padding_left,
        args.padding_right,
        input_tensor_shape[args.dim]);

    TT_FATAL(args.cluster_axis == 0 || args.cluster_axis == 1, "Unsupported cluster axis {}.", args.cluster_axis);

    TT_FATAL(args.num_links > 0, "Error, num_links should be more than 0 but has {}", args.num_links);
    if (args.dim > 0) {
        uint32_t outer_dim_size = 1;
        for (uint32_t d = 0; d < args.dim; d++) {
            outer_dim_size *= input_tensor_shape[d];
        }
        TT_FATAL(outer_dim_size >= args.num_links, "Not enough work to split among links, reduce num links");
    } else {
        uint32_t num_sticks_per_halo_dim = 1;
        for (uint32_t d = args.dim + 1; d < input_tensor_shape.size() - 1; d++) {
            num_sticks_per_halo_dim *= input_tensor_shape[d];
        }
        TT_FATAL(num_sticks_per_halo_dim >= args.num_links, "Not enough work to split among links, reduce num links");
    }

    if (args.secondary_cluster_axis.has_value()) {
        const auto& mesh_view = tensor_args.input_tensor.device()->get_view();
        uint32_t target_ring_size = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
        TT_FATAL(
            args.secondary_cluster_axis.value() == 0 || args.secondary_cluster_axis.value() == 1,
            "Unsupported secondary cluster axis {}.",
            args.secondary_cluster_axis.value());
        TT_FATAL(
            args.secondary_mesh_shape.has_value(),
            "If secondary cluster axis is specified, need to have a secondary mesh shape");
        TT_FATAL(
            !(target_ring_size % args.secondary_mesh_shape.value().at(0)) &&
                !(target_ring_size % args.secondary_mesh_shape.value().at(1)),
            "Secondary mesh shape ({},{}) is not valid given main cluster axis device count {}",
            args.secondary_mesh_shape.value().at(0),
            args.secondary_mesh_shape.value().at(1),
            target_ring_size);
    }
}

TensorSpec NeighborPadAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto shape = input_tensor.logical_shape();
    shape[args.dim] += (args.padding_left + args.padding_right);
    return TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), args.output_mem_config));
}

Tensor NeighborPadAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t NeighborPadAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto input_shape = tensor_args.input_tensor.padded_shape();
    auto input_memory_layout = tensor_args.input_tensor.layout();
    auto input_dtype = tensor_args.input_tensor.dtype();
    auto input_memory_config = tensor_args.input_tensor.memory_config();

    auto program_factory = select_program_factory(args, tensor_args);

    return operation::hash_operation<NeighborPadAsyncDeviceOperation>(
        args.dim,
        args.padding_left,
        args.padding_right,
        args.padding_mode,
        args.num_links,
        args.output_mem_config,
        args.topology,
        args.cluster_axis,
        args.ring_size,
        args.secondary_cluster_axis,
        args.secondary_mesh_shape,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config,
        program_factory.index());
}

}  // namespace ttnn::operations::experimental::ccl::neighbor_pad

namespace ttnn::prim {

ttnn::operations::experimental::ccl::neighbor_pad::NeighborPadAsyncDeviceOperation::tensor_return_value_t
neighbor_pad_async(
    const Tensor& input_tensor,
    int32_t dim,
    uint32_t padding_left,
    uint32_t padding_right,
    const std::string& padding_mode,
    uint32_t cluster_axis,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    std::optional<size_t> num_preferred_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::ccl::Topology> topology,
    std::optional<uint32_t> secondary_cluster_axis,
    const std::optional<std::vector<uint32_t>>& secondary_mesh_shape) {
    using OperationType = ttnn::operations::experimental::ccl::neighbor_pad::NeighborPadAsyncDeviceOperation;

    auto* mesh_device = input_tensor.device();
    uint32_t num_devices;
    const auto& mesh_view = mesh_device->get_view();
    num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    TT_FATAL(num_devices > 1, "neighbor_pad_async op will only work for num_devices > 1, but has {}", num_devices);

    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);

    auto operation_attributes = OperationType::operation_attributes_t(
        dim,
        padding_left,
        padding_right,
        padding_mode,
        cluster_axis,
        final_semaphore,
        barrier_semaphore,
        num_preferred_links.value_or(1),
        memory_config.value_or(input_tensor.memory_config()),
        topology_,
        num_devices,
        secondary_cluster_axis,
        secondary_mesh_shape);

    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor, .preallocated_output = std::nullopt};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
