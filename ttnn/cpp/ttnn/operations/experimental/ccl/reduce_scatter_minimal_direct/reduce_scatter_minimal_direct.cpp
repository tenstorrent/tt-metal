// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_direct.hpp"
#include "device/reduce_scatter_minimal_direct_op_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::experimental {

ttnn::Tensor reduce_scatter_minimal_direct(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_buffers,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    std::optional<ttnn::Tensor> persistent_output;
    std::optional<ttnn::Tensor> persistent_staging;
    if (persistent_buffers.has_value()) {
        const auto& b = persistent_buffers.value();
        TT_FATAL(
            b.size() == 2,
            "reduce_scatter_minimal_direct persistent_buffers must be {{output, staging}}, got {}",
            b.size());
        persistent_output = b[0];
        persistent_staging = b[1];
    }

    auto result = ttnn::prim::reduce_scatter_minimal_direct(
        input_tensor,
        dim,
        memory_config.value_or(input_tensor.memory_config()),
        cluster_axis,
        num_links,
        persistent_output,
        persistent_staging,
        sub_device_id,
        sub_core_grid);

    return result.at(0);
}

namespace {

// Both allocators below go through the device op's own compute_output_specs, so a caller-provided
// buffer is byte-identical to what the op would have allocated itself -- including the staging
// L1-sharded / L1-interleaved / DRAM placement decision, which depends on the shape and would be
// impossible for a caller to reproduce by hand.
std::vector<ttnn::TensorSpec> reduce_scatter_minimal_direct_buffer_specs(
    const ttnn::Tensor& input_tensor, int32_t dim, std::optional<uint32_t> cluster_axis) {
    ttnn::experimental::prim::ReduceScatterMinimalDirectInputs inputs{.input_tensor = input_tensor};
    const uint32_t rank = input_tensor.logical_shape().rank();

    ttnn::experimental::prim::ReduceScatterMinimalDirectParams params{};
    params.dim = (dim < 0) ? static_cast<int32_t>(rank) + dim : dim;
    params.output_mem_config = input_tensor.memory_config();
    params.cluster_axis = cluster_axis;
    params.num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);

    return ttnn::experimental::prim::ReduceScatterMinimalDirectDeviceOperation::compute_output_specs(params, inputs);
}

}  // namespace

std::vector<ttnn::Tensor> reduce_scatter_minimal_direct_create_persistent_buffers(
    const ttnn::Tensor& input_tensor, int32_t dim, std::optional<uint32_t> cluster_axis) {
    auto* device = input_tensor.device();
    TT_FATAL(device != nullptr, "input tensor must be on device to allocate persistent buffers");

    auto specs = reduce_scatter_minimal_direct_buffer_specs(input_tensor, dim, cluster_axis);
    std::vector<ttnn::Tensor> buffers;
    buffers.reserve(specs.size());
    for (const auto& spec : specs) {
        buffers.push_back(create_device_tensor(spec, device));
    }
    return buffers;
}

ttnn::Tensor reduce_scatter_minimal_direct_create_staging_buffer(
    const ttnn::Tensor& input_tensor, int32_t dim, std::optional<uint32_t> cluster_axis) {
    auto* device = input_tensor.device();
    TT_FATAL(device != nullptr, "input tensor must be on device to allocate the staging buffer");

    // Index 1 is the staging spec, by the same create_output_tensors convention the op uses.
    auto specs = reduce_scatter_minimal_direct_buffer_specs(input_tensor, dim, cluster_axis);
    return create_device_tensor(specs.at(1), device);
}

}  // namespace ttnn::experimental
