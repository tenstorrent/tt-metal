// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation.hpp"

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

/* Reduce Scatter Matmul fusion includes */
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_line_program_factory.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/matmul.hpp"  // import is_input_batched

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation_types.hpp"

namespace ttnn::experimental::prim {
void MatmulReduceScatterAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensors) {
    TT_FATAL(
        args.reduce_scatter_params.semaphore.size() >=
            (args.reduce_scatter_params.topology == ttnn::ccl::Topology::Ring ? 3u : 1u),
        "Insufficient CCL semaphores");
    for (const auto* tensor :
         {&tensors.input, &tensors.weight, &tensors.persistent_intermediate, &tensors.persistent_output}) {
        TT_FATAL(
            tensor->buffer() != nullptr && tensor->device() == tensors.input.device(),
            "MMRS tensors must be allocated on the same device");
    }
}

void MatmulReduceScatterAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(args, tensor_args);
    TT_FATAL(args.reduce_scatter_params.chunks_per_sync.value_or(1) > 0, "chunks_per_sync must be positive");
    TT_FATAL(
        args.reduce_scatter_params.num_buffers_per_channel.value_or(1) > 0, "num_buffers_per_channel must be positive");
    // Matmul validate
    ttnn::prim::MatmulDeviceOperation::validate_on_program_cache_miss(
        args.matmul_struct,
        {.input_tensors = {tensor_args.input, tensor_args.weight},
         .optional_input_tensors = {tensor_args.bias},
         .optional_output_tensors = {}});

    // Matmul Reduce Scatter validate
    TT_FATAL(
        args.reduce_scatter_params.dim == 2 || args.reduce_scatter_params.dim == 3,
        "MatmulReduceScatterAsync requires dim=2 or dim=3 for the ReduceScatter operations.");

    const auto specs = compute_output_specs(args, tensor_args);
    TT_FATAL(
        tensor_args.persistent_output.tensor_spec() == specs.reduce_scatter,
        "MMRS persistent output must match the reduced matmul output spec");
    auto intermediate_shape = specs.mm.logical_shape();
    if (args.reduce_scatter_params.topology == ttnn::ccl::Topology::Linear) {
        intermediate_shape[0] *= 2;
    }
    TT_FATAL(
        tensor_args.persistent_intermediate.logical_shape() == intermediate_shape &&
            tensor_args.persistent_intermediate.dtype() == specs.mm.data_type() &&
            tensor_args.persistent_intermediate.layout() == tt::tt_metal::Layout::TILE,
        "MMRS intermediate must be input-shaped for Ring, doubled along dim 0 for Linear");

    if (args.matmul_struct.program_config.has_value()) {
        std::visit(
            [&](const auto& config) {
                using ProgramConfigType = std::decay_t<decltype(config)>;
                if (not(std::is_same_v<
                        ProgramConfigType,
                        operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>)) {
                    TT_THROW(
                        "Unsupported MatmulProgramConfig type for MatmulReduceScatterAsync. Needs to be 2D Multicast.");
                }
            },
            args.matmul_struct.program_config.value());
    }
}

MatmulReduceScatterAsyncDeviceOperation::spec_return_value_t
MatmulReduceScatterAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    std::vector<Tensor> input_tensors = {tensor_args.input, tensor_args.weight};

    // Matmul shape
    tt::tt_metal::TensorSpec matmul_output_specs = ttnn::prim::MatmulDeviceOperation::compute_output_specs(
        args.matmul_struct, {.input_tensors = input_tensors})[0];

    auto shape = matmul_output_specs.logical_shape();
    const auto& rs = args.reduce_scatter_params;
    TT_FATAL(
        (rs.dim == 2 || rs.dim == 3) && rs.dim < shape.rank() && rs.ring_size > 0 && shape[rs.dim] % rs.ring_size == 0,
        "MMRS scatter dimension must divide the matmul output");
    shape[rs.dim] /= rs.ring_size;
    return {
        .mm = matmul_output_specs,
        .reduce_scatter = tt::tt_metal::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                matmul_output_specs.data_type(),
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                args.reduce_scatter_params.output_mem_config))};
}

MatmulReduceScatterAsyncDeviceOperation::tensor_return_value_t
MatmulReduceScatterAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor = ttnn::prim::MatmulDeviceOperation::create_output_tensors(
        args.matmul_struct, {.input_tensors = {tensor_args.input, tensor_args.weight}})[0];

    return {.mm = matmul_output_tensor, .reduce_scatter = tensor_args.persistent_output};
}

ttsl::hash::hash_t MatmulReduceScatterAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "MatmulReduceScatterAsyncDeviceOperation::compute_program_hash is called");

    const ttnn::Tensor& input_tensor = tensor_args.input;
    const ttnn::Tensor& weight_tensor = tensor_args.weight;
    const std::optional<ttnn::Tensor>& bias_tensor = tensor_args.bias;
    const ttnn::Tensor& persistent_intermediate_tensor = tensor_args.persistent_intermediate;
    const ttnn::Tensor& persistent_output_tensor = tensor_args.persistent_output;
    return tt::tt_metal::operation::hash_operation<MatmulReduceScatterAsyncDeviceOperation>(
        args.reduce_scatter_params.dim,
        args.reduce_scatter_params.num_links,
        args.reduce_scatter_params.ring_size,
        args.reduce_scatter_params.output_mem_config,
        args.reduce_scatter_params.optional_intermediate_mem_config,
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
        args.reduce_scatter_params.compute_kernel_config,
        args.matmul_struct,
        args.reduce_scatter_core_grid_offset,
        input_tensor,
        weight_tensor,
        bias_tensor,
        persistent_intermediate_tensor,
        persistent_output_tensor);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::MatmulReduceScatterAsyncDeviceOperation::tensor_return_value_t matmul_reduce_scatter_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    Tensor& persistent_intermediate_buffer,
    Tensor& persistent_output_buffer,
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
    const std::optional<ttnn::MemoryConfig>& memory_config_mm,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<ttnn::DeviceComputeKernelConfig> rs_compute_kernel_config,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_buffers_per_channel) {
    using OperationType = ttnn::experimental::prim::MatmulReduceScatterAsyncDeviceOperation;
    const auto ring_size = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    /* Matmul setup */
    bool user_run_batched = ttnn::operations::matmul::detail::is_input_batched(weight_tensor.logical_shape());
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }

    auto matmul_struct = ttnn::prim::create_matmul_attributes(
        input_tensor,
        weight_tensor,
        /*parameters=*/
        ttnn::prim::MatmulParams{
            program_config,
            /*bcast_batch=*/std::nullopt,
            memory_config_mm.value_or(input_tensor.memory_config()),
            dtype.value_or(input_tensor.dtype()),
            compute_kernel_config,
            /*untilize_out=*/false,
            user_core_coord,
            ttnn::operations::matmul::get_fused_activation(activation),
            user_run_batched,
            transpose_a,
            transpose_b,
            /*output_tile=*/std::nullopt,
            /*global_cb=*/std::nullopt},
        {});

    // Not using persistent buffers not currently supported by the RSMM API
    bool using_persistent_buffers = true;

    /* ReduceScatter setup */
    ttnn::experimental::prim::ReduceScatterMinimalAsyncParams reduce_scatter_params{
        .dim = dim,
        .num_links = num_links,
        .ring_size = ring_size,
        .output_mem_config = memory_config_rs.value_or(input_tensor.memory_config()),
        .optional_intermediate_mem_config = intermediate_memory_config_rs.value_or(input_tensor.memory_config()),
        .topology = topology,
        .semaphore = multi_device_global_semaphore,
        .barrier_semaphore = barrier_semaphore,
        .using_persistent_buffers = using_persistent_buffers,
        .sub_device_id = sub_device_id,
        .cluster_axis = cluster_axis,
        .chunks_per_sync = chunks_per_sync,
        .num_workers_per_link = num_workers_per_link,
        .num_buffers_per_channel = num_buffers_per_channel,
        .compute_kernel_config = rs_compute_kernel_config,
    };

    auto operation_attributes =
        OperationType::operation_attributes_t(reduce_scatter_params, matmul_struct, reduce_scatter_core_grid_offset);
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .weight = weight_tensor,
        .bias = bias,
        .persistent_intermediate = persistent_intermediate_buffer,
        .persistent_output = persistent_output_buffer};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
