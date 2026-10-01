// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation.hpp"

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"
#include "ttnn/device_operation_detail.hpp"

/* Reduce Scatter Matmul fusion includes */
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_line_program_factory.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/matmul.hpp"  // import is_input_batched

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation_types.hpp"

namespace ttnn::experimental::prim {
void MatmulReduceScatterAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Matmul validate
    ttnn::prim::MatmulDeviceOperation::validate_on_program_cache_miss(
        args.matmul_struct,
        {.input_tensors = {tensor_args.input, tensor_args.weight},
         .optional_input_tensors = {tensor_args.bias},
         .optional_output_tensors = {}});

    // Matmul Reduce Scatter validate
    TT_FATAL(
        args.reduce_scatter_params.dim == 3,
        "MatmulReduceScatterAsync requires dim=3 for the ReduceScatter operations.");

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

    // Reduce Scatter shape - use the device operation's compute_output_specs
    using ReduceScatterOp = ttnn::experimental::prim::ReduceScatterMinimalAsyncDeviceOperation;
    ttnn::experimental::prim::ReduceScatterMinimalAsyncInputs reduce_scatter_tensor_args{
        input_tensors[0], std::nullopt, std::nullopt};

    auto reduce_scatter_output_specs =
        ReduceScatterOp::compute_output_specs(args.reduce_scatter_params, reduce_scatter_tensor_args);

    return {.mm = matmul_output_specs, .reduce_scatter = reduce_scatter_output_specs[0]};
}

MatmulReduceScatterAsyncDeviceOperation::tensor_return_value_t
MatmulReduceScatterAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor = ttnn::prim::MatmulDeviceOperation::create_output_tensors(
        args.matmul_struct, {.input_tensors = {tensor_args.input, tensor_args.weight}})[0];

    return {.mm = matmul_output_tensor, .reduce_scatter = tensor_args.persistent_output};
}

std::vector<tt::tt_metal::TensorTopology> MatmulReduceScatterAsyncDeviceOperation::compute_output_topologies(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using tt::tt_metal::TensorTopology;
    // {mm, reduce_scatter}. The matmul output takes the union-default label of input, weight and bias (what launch()
    // gives a plain matmul). The reduce_scatter consumes that matmul output, so its label is the reduce_scatter of
    // the matmul LABEL -- not of any tensor in tensor_args -- along reduce_scatter_params.cluster_axis (nullopt: the
    // whole mesh) for reduce_scatter_params.dim; the caller's persistent output buffer IS that result and takes the
    // label. Not delegated to ReduceScatterMinimalAsyncDeviceOperation, whose hook labels its own 2-3 tensors from
    // its own input. No honest label (nullopt, already warned about): {} keeps the union default for both.
    const auto& input_topology = tensor_args.input.tensor_topology();
    std::vector<std::reference_wrapper<const TensorTopology>> operands{
        std::cref(input_topology), std::cref(tensor_args.weight.tensor_topology())};
    if (tensor_args.bias.has_value()) {
        operands.emplace_back(tensor_args.bias->tensor_topology());
    }
    auto [placements, shape] = ttnn::device_operation::detail::compute_output_placements_and_shape(operands);
    TensorTopology mm_topology(std::move(shape), std::move(placements), input_topology.mesh_coords());

    auto reduce_scatter_topology = ttnn::operations::ccl::common::reduce_scatter_output_topology(
        mm_topology,
        args.reduce_scatter_params.cluster_axis,
        tensor_args.input.device()->shape(),
        static_cast<uint32_t>(tensor_args.input.logical_shape().rank()),
        static_cast<int32_t>(args.reduce_scatter_params.dim));
    if (!reduce_scatter_topology.has_value()) {
        return {};
    }
    return {std::move(mm_topology), std::move(*reduce_scatter_topology)};
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
    const std::optional<const ttnn::CoreGrid> core_grid) {
    using OperationType = ttnn::experimental::prim::MatmulReduceScatterAsyncDeviceOperation;
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);

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
    constexpr uint32_t DEFAULT_WORKERS_PER_LINK = 1;
    ttnn::experimental::prim::ReduceScatterMinimalAsyncParams reduce_scatter_params{
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
        .num_workers_per_link = DEFAULT_WORKERS_PER_LINK,
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
