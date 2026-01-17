// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_device_operation.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_program_factory.hpp"

namespace ttnn::operations::experimental::ccl::slice_reshard_async {

SliceReshardAsyncDeviceOperation::program_factory_t SliceReshardAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::SliceReshardAsyncProgramFactory{};
}

void SliceReshardAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void SliceReshardAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_FATAL(args.dim == 0, "Error, neighbor pad currently only supports sharding dim 0, provided {}", args.dim);
    TT_FATAL(
        tensor_args.input.layout() == Layout::ROW_MAJOR,
        "Unsupported input tensor layout {}.",
        tensor_args.input.layout());

    TT_FATAL(!tensor_args.input.is_sharded(), "Slice reshard does not support sharded input tensors.");

    TT_FATAL(
        !(args.output_dim_shape % args.ring_size), "Output dim shape must be divisible by num devices on cluster axis");

    TT_FATAL(args.cluster_axis == 0 || args.cluster_axis == 1, "Unsupported cluster axis {}.", args.cluster_axis);

    TT_FATAL(args.num_links > 0, "Error, num_links should be more than 0 but has {}", args.num_links);
}

TensorSpec SliceReshardAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto shape = input_tensor.logical_shape();
    shape[args.dim] = args.output_dim_shape / args.ring_size;
    return TensorSpec(
        shape, TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), args.output_mem_config));
}

Tensor SliceReshardAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t SliceReshardAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "SliceReshardAsyncDeviceOperation::compute_program_hash is called");

    auto program_factory = select_program_factory(args, tensor_args);

    return tt::tt_metal::operation::hash_operation<SliceReshardAsyncDeviceOperation>(
        args.dim,
        args.output_dim_offset,
        args.output_dim_shape,
        args.cluster_axis,
        args.num_links,
        args.output_mem_config,
        args.topology,
        args.ring_size,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::operations::experimental::ccl::slice_reshard_async

namespace ttnn::prim {

ttnn::operations::experimental::ccl::slice_reshard_async::SliceReshardAsyncDeviceOperation::tensor_return_value_t
slice_reshard_async(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    uint32_t output_dim_offset,
    uint32_t output_dim_shape,
    uint32_t cluster_axis,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    size_t num_links,
    const MemoryConfig& memory_config,
    ttnn::ccl::Topology topology) {
    using OperationType = ttnn::operations::experimental::ccl::slice_reshard_async::SliceReshardAsyncDeviceOperation;
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);
    uint32_t num_devices;
    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    // Use the mesh dimensions to determine the ring size
    num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    TT_FATAL(num_devices > 1, "slice_reshard_async op will only work for num_devices > 1, but has {}", num_devices);

    auto operation_attributes = OperationType::operation_attributes_t(
        devices,
        dim,
        output_dim_offset,
        output_dim_shape,
        cluster_axis,
        final_semaphore,
        barrier_semaphore,
        num_links,
        memory_config,
        topology,
        num_devices);
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
