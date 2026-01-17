// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "strided_all_gather_async_op.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

StridedAllGatherAsync::program_factory_t StridedAllGatherAsync::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return StridedAllGatherAsyncProgramFactory{};
}

void StridedAllGatherAsync::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

void StridedAllGatherAsync::validate_on_program_cache_miss(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensors_args*/) {}

StridedAllGatherAsync::spec_return_value_t StridedAllGatherAsync::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto shape = input_tensor.logical_shape();
    shape[attributes.dim] *= attributes.ring_size;
    return {TensorSpec(
        shape,
        TensorLayout(input_tensor.dtype(), input_tensor.tensor_spec().page_config(), attributes.output_mem_config))};
}

StridedAllGatherAsync::tensor_return_value_t StridedAllGatherAsync::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.persistent_output_buffer.has_value()) {
        return {tensor_args.persistent_output_buffer.value()};
    }
    return {create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input_tensor.device())};
}

tt::tt_metal::operation::Hash StridedAllGatherAsync::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "StridedAllGatherAsync::compute_program_hash is called");

    auto program_factory = select_program_factory(attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<StridedAllGatherAsync>(
        attributes.dim,
        attributes.num_links,
        attributes.ring_size,
        attributes.output_mem_config,
        attributes.topology,
        attributes.cluster_axis,
        attributes.tiles_per_chunk,
        attributes.num_workers_per_link,
        attributes.num_buffers_per_channel,
        attributes.mm_cores_y,
        attributes.mm_block_ht,
        attributes.mm_block_wt,
        tensor_args,
        program_factory.index());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor strided_all_gather_async(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    const std::optional<uint32_t>& cluster_axis,
    const std::optional<uint32_t>& tiles_per_chunk,
    const std::optional<uint32_t>& num_workers_per_link,
    const std::optional<uint32_t>& num_buffers_per_channel,
    const std::optional<uint32_t>& mm_cores_y,
    const std::optional<uint32_t>& mm_block_ht,
    const std::optional<uint32_t>& mm_block_wt) {
    using OperationType = ttnn::experimental::prim::StridedAllGatherAsync;

    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "strided_all_gather_async op is only supported for Fast Dispatch");

    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    TT_FATAL(
        num_devices > 1, "strided_all_gather_async op will only work for num_devices > 1, but has {}", num_devices);
    ttnn::ccl::Topology ccl_topology = topology;

    if (num_devices == 2) {
        ccl_topology = ttnn::ccl::Topology::Linear;
    }
    log_debug(tt::LogOp, "DEBUG: line_fabric is created");

    auto operation_attributes = OperationType::operation_attributes_t{
        ttnn::ccl::get_active_physical_devices(input_tensor),
        dim,
        num_links,
        num_devices,
        memory_config.value_or(input_tensor.memory_config()),
        ccl_topology,
        multi_device_global_semaphore,
        cluster_axis,
        tiles_per_chunk,
        num_workers_per_link,
        num_buffers_per_channel,
        mm_cores_y,
        mm_block_ht,
        mm_block_wt};
    auto tensor_args = OperationType::tensor_args_t{input_tensor, persistent_output_buffer};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
