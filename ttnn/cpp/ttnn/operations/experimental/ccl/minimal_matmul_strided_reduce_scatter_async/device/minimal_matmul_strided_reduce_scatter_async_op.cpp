// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/device/minimal_matmul_strided_reduce_scatter_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

using matmul_device_operation_t = ttnn::experimental::prim::MinimalMatmulDeviceOperation;

namespace ttnn::experimental::prim {

MinimalMatmulStridedReduceScatterAsync::program_factory_t
MinimalMatmulStridedReduceScatterAsync::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return MinimalMatmulStridedReduceScatterAsyncProgramFactory{};
}

void MinimalMatmulStridedReduceScatterAsync::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

void MinimalMatmulStridedReduceScatterAsync::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        attributes.dim == 3, "MinimalMatmulStridedReduceScatterAsync requires dim=3 for the ReduceScatter operation.");
    TT_FATAL(
        tensor_args.input_tensor.padded_shape()[0] == 1 && tensor_args.input_tensor.padded_shape()[1] == 1,
        "MinimalMatmulStridedReduceScatterAsync requires input tensor to have batch size of 1.");
    TT_FATAL(
        attributes.topology == ttnn::ccl::Topology::Ring,
        "MinimalMatmulStridedReduceScatterAsync only supports Ring topology.");
}

MinimalMatmulStridedReduceScatterAsync::spec_return_value_t
MinimalMatmulStridedReduceScatterAsync::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Output tensor[0]: MM output spec (= RS input)
    ttnn::TensorSpec mm_output_spec = matmul_device_operation_t::compute_output_specs(
        attributes.matmul_struct, {tensor_args.input_tensor, tensor_args.weight_tensor});

    // Derive RS intermediate and output specs from the MM output shape
    auto mm_output_shape = mm_output_spec.logical_shape();

    // RS intermediate shape: same as MM output for Ring topology
    MemoryConfig rs_intermediate_mem_config =
        attributes.rs_intermediate_mem_config.value_or(mm_output_spec.memory_config());

    ttnn::TensorSpec rs_intermediate_spec(
        mm_output_shape,
        tt::tt_metal::TensorLayout(
            mm_output_spec.data_type(), mm_output_spec.page_config(), rs_intermediate_mem_config));

    // RS output shape: scatter dim divided by ring_size
    auto rs_output_shape = mm_output_shape;
    rs_output_shape[attributes.dim] /= attributes.ring_size;

    ttnn::TensorSpec rs_output_spec(
        rs_output_shape,
        tt::tt_metal::TensorLayout(
            mm_output_spec.data_type(), mm_output_spec.page_config(), attributes.rs_output_mem_config));

    return {mm_output_spec, rs_intermediate_spec, rs_output_spec};
}

MinimalMatmulStridedReduceScatterAsync::tensor_return_value_t
MinimalMatmulStridedReduceScatterAsync::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto tensor_specs = compute_output_specs(attributes, tensor_args);

    // MM output tensor
    ttnn::Tensor mm_output_tensor = create_device_tensor(tensor_specs[0], tensor_args.input_tensor.device());

    // RS intermediate tensor (use provided or create new)
    ttnn::Tensor rs_intermediate_tensor =
        tensor_args.optional_rs_intermediate_tensor.has_value()
            ? tensor_args.optional_rs_intermediate_tensor.value()
            : create_device_tensor(tensor_specs[1], tensor_args.input_tensor.device());

    // RS output tensor (use provided or create new)
    ttnn::Tensor rs_output_tensor = tensor_args.optional_rs_output_tensor.has_value()
                                        ? tensor_args.optional_rs_output_tensor.value()
                                        : create_device_tensor(tensor_specs[2], tensor_args.input_tensor.device());

    return {mm_output_tensor, rs_intermediate_tensor, rs_output_tensor};
}

tt::tt_metal::operation::Hash MinimalMatmulStridedReduceScatterAsync::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "MinimalMatmulStridedReduceScatterAsync::compute_program_hash is called");

    auto program_factory = select_program_factory(attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<MinimalMatmulStridedReduceScatterAsync>(
        // RS params
        attributes.dim,
        attributes.num_links,
        attributes.ring_size,
        attributes.rs_output_mem_config,
        attributes.rs_intermediate_mem_config,
        attributes.topology,
        attributes.barrier_semaphore.has_value(),
        attributes.using_persistent_buffers,
        attributes.sub_device_id.has_value(),
        attributes.cluster_axis,
        attributes.chunks_per_sync,
        attributes.num_workers_per_link,
        attributes.num_buffers_per_channel,
        attributes.chunk_width_in_mm_blocks,
        attributes.reduce_scatter_core_grid_offset,
        // MM params
        attributes.matmul_struct,
        // Tensor info
        tensor_args.input_tensor.logical_shape(),
        tensor_args.input_tensor.padded_shape(),
        tensor_args.input_tensor.tensor_spec().page_config(),
        tensor_args.input_tensor.dtype(),
        tensor_args.input_tensor.layout(),
        tensor_args.input_tensor.memory_config(),
        tensor_args.weight_tensor.logical_shape(),
        tensor_args.weight_tensor.padded_shape(),
        tensor_args.weight_tensor.memory_config(),
        program_factory.index());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> minimal_matmul_strided_reduce_scatter_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord reduce_scatter_core_grid_offset,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_mm,
    const MemoryConfig& rs_output_mem_config,
    const std::optional<MemoryConfig>& rs_intermediate_mem_config,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    const std::optional<const Tensor>& bias,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    std::optional<const ttnn::experimental::prim::MinimalMatmulConfig> config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool using_persistent_buffers,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<uint32_t> chunk_width_in_mm_blocks,
    const std::optional<Tensor>& optional_rs_intermediate_tensor,
    const std::optional<Tensor>& optional_rs_output_tensor) {
    using OperationType = ttnn::experimental::prim::MinimalMatmulStridedReduceScatterAsync;

    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);
    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    const auto resolved_sub_device_id =
        sub_device_id.has_value()
            ? sub_device_id
            : std::optional<tt::tt_metal::SubDeviceId>(input_tensor.device()->get_sub_device_ids().at(0));

    /* Matmul setup */
    auto matmul_struct =
        decltype(ttnn::experimental::prim::MinimalMatmulStridedReduceScatterAsyncParams::matmul_struct){
            .config = config,
            .fused_activation = std::move(fused_activation),
            .output_mem_config = memory_config_mm,
            .compute_kernel_config = compute_kernel_config.value()};

    auto operation_attributes = OperationType::operation_attributes_t{
        /* matmul_struct */ matmul_struct,
        /* dim */ dim,
        /* num_links */ num_links,
        /* ring_size */ num_devices,
        /* rs_output_mem_config */ rs_output_mem_config,
        /* rs_intermediate_mem_config */ rs_intermediate_mem_config,
        /* topology */ topology,
        /* semaphore */ multi_device_global_semaphore,
        /* barrier_semaphore */ barrier_semaphore,
        /* using_persistent_buffers */ using_persistent_buffers,
        /* sub_device_id */ resolved_sub_device_id,
        /* cluster_axis */ cluster_axis,
        /* chunks_per_sync */ chunks_per_sync,
        /* num_workers_per_link */ num_workers_per_link,
        /* num_buffers_per_channel */ num_buffers_per_channel,
        /* chunk_width_in_mm_blocks */ chunk_width_in_mm_blocks,
        /* reduce_scatter_core_grid_offset */ reduce_scatter_core_grid_offset,
        /* devices */ devices};

    auto tensor_args = OperationType::tensor_args_t{
        input_tensor, weight_tensor, optional_rs_intermediate_tensor, optional_rs_output_tensor, bias};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
