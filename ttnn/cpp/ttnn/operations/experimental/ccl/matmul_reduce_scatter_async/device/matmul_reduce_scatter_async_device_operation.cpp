// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_async_device_operation.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async {

MatmulReduceScatterAsyncDeviceOperation::program_factory_t
MatmulReduceScatterAsyncDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // This operation always uses mesh workloads for multi-device execution
    // Use MeshWorkloadFactory for proper multi-device support
    return program::MatmulReduceScatterAsyncMeshWorkloadFactory{};
}

void MatmulReduceScatterAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MatmulReduceScatterAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_ASSERT(
        args.reduce_scatter_minimal_async_struct.dim == 3,
        "MatmulReduceScatterAsync requires dim=3 for the reduce scatter operation.");

    // Validate matmul
    std::vector<Tensor> input_tensors = {tensor_args.input_tensor, tensor_args.weight_tensor};
    std::vector<std::optional<const Tensor>> optional_input_tensors;
    if (tensor_args.bias.has_value()) {
        optional_input_tensors.push_back(tensor_args.bias.value());
    } else {
        optional_input_tensors.push_back(std::nullopt);
    }
    args.matmul_struct.validate(input_tensors, optional_input_tensors, {});

    // Validate matmul program config
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
    // Compute matmul output spec
    std::vector<Tensor> input_tensors = {tensor_args.input_tensor, tensor_args.weight_tensor};
    ttnn::TensorSpec matmul_output_spec = args.matmul_struct.compute_output_specs(input_tensors, {}, {})[0];

    // Compute reduce scatter output spec
    // Note: The reduce scatter uses the matmul output as input, but we use the original input_tensor
    // as a proxy since the matmul output shape is derived from it
    operations::experimental::ccl::reduce_scatter_minimal_async::tensor_args_t reduce_scatter_tensor_args{
        tensor_args.input_tensor,  // input_tensor (proxy - actual input is matmul output)
        std::nullopt               // persistent_output_buffers
    };
    ttnn::TensorSpec reduce_scatter_output_spec =
        operations::experimental::ccl::reduce_scatter_minimal_async::ReduceScatterMinimalAsyncDeviceOperation::
            compute_output_specs(args.reduce_scatter_minimal_async_struct, reduce_scatter_tensor_args)[1];

    // Return order: (matmul_output_spec, reduce_scatter_output_spec) to match tuple return type
    return std::make_tuple(matmul_output_spec, reduce_scatter_output_spec);
}

MatmulReduceScatterAsyncDeviceOperation::tensor_return_value_t
MatmulReduceScatterAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Create matmul output tensor
    std::vector<Tensor> input_tensors = {tensor_args.input_tensor, tensor_args.weight_tensor};
    ttnn::Tensor matmul_output_tensor = operation_attributes.matmul_struct.create_output_tensors(input_tensors, {})[0];

    // Use provided persistent buffers for reduce scatter
    // Note: The reduce scatter output is the persistent_output_buffer
    return std::make_tuple(matmul_output_tensor, tensor_args.persistent_output_buffer);
}

tt::stl::hash::hash_t MatmulReduceScatterAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "compute_program_hash is called");
    auto input_shape = tensor_args.input_tensor.padded_shape();
    auto input_memory_layout = tensor_args.input_tensor.layout();
    auto input_dtype = tensor_args.input_tensor.dtype();
    auto input_memory_config = tensor_args.input_tensor.memory_config();

    return tt::tt_metal::operation::hash_operation<MatmulReduceScatterAsyncDeviceOperation>(
        args.reduce_scatter_minimal_async_struct.dim,
        args.reduce_scatter_minimal_async_struct.num_links,
        args.reduce_scatter_minimal_async_struct.ring_size,
        args.reduce_scatter_minimal_async_struct.output_mem_config,
        args.reduce_scatter_minimal_async_struct.optional_intermediate_mem_config.value(),
        args.reduce_scatter_minimal_async_struct.topology,
        args.reduce_scatter_minimal_async_struct.sub_device_id.has_value(),
        args.reduce_scatter_minimal_async_struct.sub_device_id.has_value()
            ? tensor_args.input_tensor.device()->worker_cores(
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
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    Tensor& persistent_intermediate_buffer,
    Tensor& persistent_output_buffer,
    uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    CoreCoord reduce_scatter_core_grid_offset,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const std::optional<const Tensor>& bias,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_rs,
    const std::optional<MemoryConfig>& intermediate_memory_config_rs,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<MemoryConfig>& memory_config_mm,
    bool transpose_a,
    bool transpose_b,
    std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<const ttnn::CoreGrid> core_grid) {
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
            /*global_cb=*/std::nullopt},
        {});

    // Not using persistent buffers not currently supported by the RSMM API
    bool using_persistent_buffers = true;

    /* ReduceScatter setup */
    constexpr uint32_t DEFAULT_WORKERS_PER_LINK = 1;
    ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t
        reduce_scatter_minimal_async_struct{
            dim,
            num_links,
            static_cast<uint32_t>(devices.size()),
            memory_config_rs.value_or(input_tensor.memory_config()),
            intermediate_memory_config_rs.value_or(input_tensor.memory_config()),
            topology,
            multi_device_global_semaphore,
            barrier_semaphore,
            using_persistent_buffers,
            sub_device_id,
            std::nullopt,  // cluster_axis
            std::nullopt,  // chunks_per_sync
            DEFAULT_WORKERS_PER_LINK,
            std::nullopt  // num_buffers_per_channel
        };

    operation_attributes_t operation_attributes{
        reduce_scatter_minimal_async_struct, matmul_struct, reduce_scatter_core_grid_offset};

    tensor_args_t tensor_args{
        input_tensor, weight_tensor, persistent_intermediate_buffer, persistent_output_buffer, bias};

    return std::make_tuple(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async
