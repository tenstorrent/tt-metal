// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation.hpp"

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

/* All Gather Matmul fusion includes */
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation_types.hpp"

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async {

MatmulReduceScatterAsyncDeviceOperation::program_factory_t
MatmulReduceScatterAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return program::MatmulReduceScatterAsyncProgramFactory{};
}

void MatmulReduceScatterAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MatmulReduceScatterAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Matmul validate
    args.matmul_struct.validate({tensor_args.input, tensor_args.weight}, {tensor_args.bias}, {});

    // Matmul Reduce Scatter validate
    TT_FATAL(
        args.reduce_scatter_minimal_async_struct.dim == 3,
        "MatmulReduceScatterAsync requires dim=3 for the AllGather operations.");

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

spec_return_value_t MatmulReduceScatterAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    std::vector<Tensor> input_tensors = {tensor_args.input, tensor_args.weight};

    // Matmul shape
    ttnn::TensorSpec matmul_output_specs = args.matmul_struct.compute_output_specs(input_tensors, {})[0];

    // Reduce Scatter shape
    ttnn::TensorSpec reduce_scatter_output_specs =
        args.reduce_scatter_minimal_async_struct.compute_output_specs(input_tensors)[0];

    return {.mm = matmul_output_specs, .reduce_scatter = reduce_scatter_output_specs};
}

tensor_return_value_t MatmulReduceScatterAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor =
        args.matmul_struct.create_output_tensors({tensor_args.input, tensor_args.weight})[0];

    return {.mm = matmul_output_tensor, .reduce_scatter = args.persistent_output_buffer};
}

tt::stl::hash::hash_t MatmulReduceScatterAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = tensor_args.input.padded_shape();
    auto input_memory_layout = tensor_args.input.layout();
    auto input_dtype = tensor_args.input.dtype();
    auto input_memory_config = tensor_args.input.memory_config();

    return tt::tt_metal::operation::hash_operation<MatmulReduceScatterAsyncDeviceOperation>(
        args.reduce_scatter_minimal_async_struct.dim,
        args.reduce_scatter_minimal_async_struct.num_links,
        args.reduce_scatter_minimal_async_struct.ring_size,
        args.reduce_scatter_minimal_async_struct.output_mem_config,
        args.reduce_scatter_minimal_async_struct.optional_intermediate_mem_config.value(),
        args.reduce_scatter_minimal_async_struct.topology,
        args.reduce_scatter_minimal_async_struct.sub_device_id.has_value(),
        args.reduce_scatter_minimal_async_struct.sub_device_id.has_value()
            ? tensor_args.input.device()->worker_cores(
                  tt::tt_metal::HalProgrammableCoreType::TENSIX,
                  args.reduce_scatter_minimal_async_struct.sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        args.reduce_scatter_minimal_async_struct.cluster_axis,
        args.reduce_scatter_minimal_async_struct.barrier_semaphore.has_value(),
        args.reduce_scatter_minimal_async_struct.using_persistent_buffers,
        args.reduce_scatter_minimal_async_struct.chunks_per_sync,
        args.reduce_scatter_minimal_async_struct.num_workers_per_link,
        args.reduce_scatter_minimal_async_struct.num_buffers_per_channel,
        args.reduce_scatter_core_grid_offset,
        input_shape,
        input_memory_layout,
        input_dtype,
        input_memory_config);
}

std::tuple<
    MatmulReduceScatterAsyncDeviceOperation::operation_attributes_t,
    MatmulReduceScatterAsyncDeviceOperation::tensor_args_t>
MatmulReduceScatterAsyncDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    ttnn::Tensor& persistent_intermediate_buffer,
    ttnn::Tensor& persistent_output_buffer,
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
    const std::optional<const ttnn::CoreGrid> core_grid) {
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);

    /* Matmul setup */
    bool user_run_batched = ttnn::operations::matmul::detail::is_input_batched(weight_tensor.logical_shape());
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }

    operations::matmul::Matmul matmul_struct = operations::matmul::create_matmul_struct(
        input_tensor,
        weight_tensor,
        /*parameters=*/
        operations::matmul::Matmul{
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
            /*global_cb=*/std::nullopt});

    // Not using persistent buffers not currently supported by the RSMM API
    bool using_persistent_buffers = true;

    /* ReduceScatter setup */
    constexpr uint32_t DEFAULT_WORKERS_PER_LINK = 1;
    ttnn::ReduceScatterMinimalAsync reduce_scatter_minimal_async_struct = ttnn::ReduceScatterMinimalAsync(
        dim,
        num_links,
        devices.size(),
        memory_config_rs.value_or(input_tensor.memory_config()),
        intermediate_memory_config_rs.value_or(input_tensor.memory_config()),
        topology,
        multi_device_global_semaphore,
        barrier_semaphore,
        using_persistent_buffers,
        sub_device_id,
        std::nullopt,
        std::nullopt,
        DEFAULT_WORKERS_PER_LINK,
        std::nullopt);

    return {
        operation_attributes_t(
            reduce_scatter_minimal_async_struct,
            matmul_struct,
            reduce_scatter_core_grid_offset,
            devices,
            persistent_intermediate_buffer,
            persistent_output_buffer),
        tensor_args_t{.input = input_tensor, .weight = weight_tensor, .bias = bias}};
}

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async
