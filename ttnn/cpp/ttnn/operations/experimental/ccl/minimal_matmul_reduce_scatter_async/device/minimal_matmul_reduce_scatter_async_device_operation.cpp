// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/matmul.hpp"  // import is_input_batched

#include "ttnn/operations/experimental/ccl/minimal_matmul_reduce_scatter_async/device/minimal_matmul_reduce_scatter_async_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/minimal_matmul_reduce_scatter_async/device/minimal_matmul_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.hpp"

namespace ttnn::operations::experimental::ccl::minimal_matmul_reduce_scatter_async {

MinimalMatmulReduceScatterAsyncDeviceOperation::program_factory_t
MinimalMatmulReduceScatterAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::MinimalMatmulReduceScatterAsyncProgramFactory{};
}

void MinimalMatmulReduceScatterAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MinimalMatmulReduceScatterAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Matmul validate
    ttnn::experimental::prim::MinimalMatmulDeviceOperation::validate_on_program_cache_miss(
        args.matmul_struct,
        {
            .input_tensor = tensor_args.input, .weight_tensor = tensor_args.weight, .bias_tensor = {tensor_args.bias},
            // .optional_output_tensors = {output_tensors.mm}
        });

    // Matmul Reduce Scatter validate
    TT_FATAL(
        args.reduce_scatter_params.dim == 3,
        "MatmulReduceScatterAsync requires dim=3 for the ReduceScatter operations.");
}

spec_return_value_t MinimalMatmulReduceScatterAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    ttnn::experimental::prim::MinimalMatmulInputs matmul_inputs{
        .input_tensor = tensor_args.input, .weight_tensor = tensor_args.weight, .bias_tensor = {tensor_args.bias},
        // .optional_output_tensors = {output_tensors.mm}
    };
    // Matmul shape
    ttnn::TensorSpec matmul_output_specs =
        ttnn::experimental::prim::MinimalMatmulDeviceOperation::compute_output_specs(args.matmul_struct, matmul_inputs);

    // Reduce Scatter shape - use the device operation's compute_output_specs
    using ReduceScatterOp = ttnn::experimental::prim::ReduceScatterMinimalAsyncDeviceOperation;

    ttnn::experimental::prim::ReduceScatterMinimalAsyncInputs reduce_scatter_tensor_args{
        tensor_args.input, std::nullopt, std::nullopt, matmul_output_specs};

    auto reduce_scatter_output_specs =
        ReduceScatterOp::compute_output_specs(args.reduce_scatter_params, reduce_scatter_tensor_args);

    return {.mm = matmul_output_specs, .reduce_scatter = reduce_scatter_output_specs[1]};
}

tensor_return_value_t MinimalMatmulReduceScatterAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor = ttnn::experimental::prim::MinimalMatmulDeviceOperation::create_output_tensors(
        args.matmul_struct,
        {
            .input_tensor = tensor_args.input, .weight_tensor = tensor_args.weight, .bias_tensor = {tensor_args.bias},
            // .optional_output_tensors = {output_tensors.mm}
        });
    if (tensor_args.persistent_output.has_value()) {
        return {.mm = matmul_output_tensor, .reduce_scatter = tensor_args.persistent_output.value()};
    }
    auto output_specs = compute_output_specs(args, tensor_args);
    auto output_tensor = tt::tt_metal::create_device_tensor(output_specs.reduce_scatter, tensor_args.input.device());
    return {.mm = matmul_output_tensor, .reduce_scatter = output_tensor};
}

tt::stl::hash::hash_t MinimalMatmulReduceScatterAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = tensor_args.input.padded_shape();
    auto input_memory_layout = tensor_args.input.layout();
    auto input_dtype = tensor_args.input.dtype();
    auto input_memory_config = tensor_args.input.memory_config();

    return tt::tt_metal::operation::hash_operation<MinimalMatmulReduceScatterAsyncDeviceOperation>(
        args.reduce_scatter_params.dim,
        args.reduce_scatter_params.num_links,
        args.reduce_scatter_params.ring_size,
        args.reduce_scatter_params.output_mem_config,
        args.reduce_scatter_params.optional_intermediate_mem_config.value(),
        args.reduce_scatter_params.topology,
        args.reduce_scatter_params.sub_device_id.has_value(),
        args.reduce_scatter_params.sub_device_id.has_value()
            ? tensor_args.input.device()->worker_cores(
                  tt::tt_metal::HalProgrammableCoreType::TENSIX, args.reduce_scatter_params.sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        args.reduce_scatter_params.cluster_axis,
        args.reduce_scatter_params.barrier_semaphore.has_value(),
        args.reduce_scatter_params.using_persistent_buffers,
        args.reduce_scatter_params.chunks_per_sync,
        args.reduce_scatter_params.num_workers_per_link,
        args.reduce_scatter_params.num_buffers_per_channel,
        args.reduce_scatter_core_grid_offset,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

}  // namespace ttnn::operations::experimental::ccl::minimal_matmul_reduce_scatter_async

namespace ttnn::prim {

ttnn::operations::experimental::ccl::minimal_matmul_reduce_scatter_async::
    MinimalMatmulReduceScatterAsyncDeviceOperation::tensor_return_value_t
    minimal_matmul_reduce_scatter_async(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        Tensor& persistent_intermediate_buffer,
        std::optional<ttnn::Tensor>& persistent_output_buffer,
        const uint32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        const CoreCoord reduce_scatter_core_grid_offset,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        const std::optional<const Tensor>& bias,
        const uint32_t num_links,
        const std::optional<ttnn::MemoryConfig>& memory_config_rs,
        const std::optional<ttnn::MemoryConfig>& intermediate_memory_config_rs,
        const ttnn::ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
        std::optional<uint32_t> cluster_axis,
        std::optional<uint32_t> num_workers_per_link,
        const std::optional<ttnn::MemoryConfig>& memory_config_mm,
        const std::optional<const DataType> dtype,
        const std::optional<const ::ttnn::experimental::prim::MinimalMatmulConfig>& program_config,
        const std::optional<const operations::unary::UnaryWithParam>& activation,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = ttnn::operations::experimental::ccl::minimal_matmul_reduce_scatter_async::
        MinimalMatmulReduceScatterAsyncDeviceOperation;
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);

    /* Matmul setup */
    // bool user_run_batched = ttnn::operations::matmul::detail::is_input_batched(weight_tensor.logical_shape());
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        MathFidelity::HiFi2,
        false /*approx_mode*/,
        true /*fp32_acc*/,
        true /*packer_acc*/);
    auto matmul_struct = ttnn::experimental::prim::MinimalMatmulDeviceOperation::operation_attributes_t{
        .config = program_config,
        .fused_activation = activation,
        .output_mem_config = memory_config_mm,
        .output_dtype = dtype,
        .compute_kernel_config = kernel_config_val,
    };

    // Not using persistent buffers not currently supported by the RSMM API
    bool using_persistent_buffers = persistent_output_buffer.has_value();

    /* ReduceScatter setup */
    ttnn::operations::experimental::ccl::minimal_matmul_reduce_scatter_async::ReduceScatterMinimalAsyncParams
        reduce_scatter_params{
            .dim = dim,
            .num_links = num_links,
            .ring_size = static_cast<uint32_t>(devices.size()),
            .output_mem_config = memory_config_rs.value_or(input_tensor.memory_config()),
            .optional_intermediate_mem_config = intermediate_memory_config_rs.value_or(input_tensor.memory_config()),
            .topology = topology,
            .semaphore = multi_device_global_semaphore,
            .barrier_semaphore = barrier_semaphore,
            .using_persistent_buffers = using_persistent_buffers,
            .sub_device_id = sub_device_id,
            .cluster_axis = std::nullopt,
            .chunks_per_sync = std::nullopt,
            .num_workers_per_link = num_workers_per_link,
            .num_buffers_per_channel = std::nullopt,
        };

    auto operation_attributes = OperationType::operation_attributes_t(
        reduce_scatter_params, matmul_struct, reduce_scatter_core_grid_offset, devices);
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .weight = weight_tensor,
        .bias = bias,
        .persistent_intermediate = persistent_intermediate_buffer,
        .persistent_output = persistent_output_buffer};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
