// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/device/strided_all_gather_minimal_matmul_async_op.hpp"

/* All Gather Matmul fusion includes */
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp"

using matmul_device_operation_t = ttnn::experimental::prim::MinimalMatmulDeviceOperation;

namespace ttnn::experimental::prim {
void StridedAllGatherMinimalMatmulAsync::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        attributes.strided_all_gather_async_struct.dim == 3,
        "StridedAllGatherMinimalMatmulAsync requires dim=3 for the AllGather operations.");
    TT_FATAL(
        tensor_args.input_tensor.padded_shape()[0] == 1 && tensor_args.input_tensor.padded_shape()[1] == 1,
        "StridedAllGatherMinimalMatmulAsync requires input tensor to have batch size of 1.");
    TT_FATAL(
        tensor_args.fused_ternary_input_a.has_value() == tensor_args.fused_ternary_input_b.has_value(),
        "StridedAllGatherMinimalMatmulAsync fused addcmul requires both ternary inputs (a and b) or neither.");

    // SwiGLU packs gate/up tile-pairs along the weight's N, so the weight width must be an even
    // number of tiles; each pair collapses to one output tile. Restated here because this op does
    // not route through MinimalMatmulDeviceOperation::validate_*.
    if (attributes.matmul_struct.fuse_swiglu) {
        TT_FATAL(
            !attributes.matmul_struct.fused_activation.has_value(),
            "StridedAllGatherMinimalMatmulAsync cannot combine fuse_swiglu with a unary fused_activation");
        TT_FATAL(
            !attributes.matmul_struct.fused_ternary_scalar.has_value() &&
                !tensor_args.fused_ternary_input_a.has_value(),
            "StridedAllGatherMinimalMatmulAsync cannot combine fuse_swiglu with fused ternary (addcmul)");
        const uint32_t N = tensor_args.weight_tensor.logical_shape()[-1];
        const int32_t chunks = attributes.matmul_struct.chunks;
        TT_FATAL(
            N % (2 * tt::constants::TILE_WIDTH) == 0,
            "StridedAllGatherMinimalMatmulAsync fuse_swiglu requires weight width N={} to be a multiple of "
            "2*TILE_WIDTH={}",
            N,
            2 * tt::constants::TILE_WIDTH);
        TT_FATAL(
            (N / chunks) % (2 * tt::constants::TILE_WIDTH) == 0,
            "StridedAllGatherMinimalMatmulAsync fuse_swiglu requires per-chunk weight width N/chunks={} to be a "
            "multiple of 2*TILE_WIDTH={}",
            N / chunks,
            2 * tt::constants::TILE_WIDTH);
    }
    if (tensor_args.fused_ternary_input_a.has_value()) {
        TT_FATAL(
            !attributes.matmul_struct.fused_activation.has_value(),
            "StridedAllGatherMinimalMatmulAsync cannot combine fused_activation with ternary (addcmul) inputs.");

        // Matmul output is [.., M, N]
        auto mm_specs = matmul_device_operation_t::compute_output_specs(
            attributes.matmul_struct, {tensor_args.input_tensor, tensor_args.weight_tensor});
        const auto& mm_logical = mm_specs[0].logical_shape();
        const uint32_t M = mm_logical[-2];
        const uint32_t N = mm_logical[-1];

        const auto& ternary_a = tensor_args.fused_ternary_input_a.value();
        const auto& ternary_b = tensor_args.fused_ternary_input_b.value();
        TT_FATAL(
            ternary_a.layout() == tt::tt_metal::Layout::TILE && ternary_b.layout() == tt::tt_metal::Layout::TILE,
            "StridedAllGatherMinimalMatmulAsync ternary inputs must be TILE layout.");
        const auto& a_logical = ternary_a.logical_shape();
        const auto& b_logical = ternary_b.logical_shape();
        TT_FATAL(
            a_logical[-2] == M && a_logical[-1] == N,
            "fused_ternary_input_a shape must match matmul output [M={}, N={}], got [{}, {}].",
            M,
            N,
            a_logical[-2],
            a_logical[-1]);
        TT_FATAL(
            (b_logical[-2] == 1 || b_logical[-2] == M) && b_logical[-1] == N,
            "fused_ternary_input_b shape must be [1, N={}] (broadcast) or [M={}, N={}] (full), got [{}, {}].",
            N,
            M,
            N,
            b_logical[-2],
            b_logical[-1]);
    }
}

StridedAllGatherMinimalMatmulAsync::spec_return_value_t StridedAllGatherMinimalMatmulAsync::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // All Gather shape
    tt::tt_metal::TensorSpec strided_all_gather_output_shape = StridedAllGatherAsync::compute_output_specs(
        attributes.strided_all_gather_async_struct, StridedAllGatherAsyncInputs{tensor_args.input_tensor});

    // Matmul specs: one per output chunk (chunks == 1 by default)
    auto minimal_matmul_output_specs_vec = matmul_device_operation_t::compute_output_specs(
        attributes.matmul_struct, {tensor_args.input_tensor, tensor_args.weight_tensor});

    spec_return_value_t specs;
    specs.reserve(1 + minimal_matmul_output_specs_vec.size());
    specs.push_back(strided_all_gather_output_shape);
    for (auto& spec : minimal_matmul_output_specs_vec) {
        specs.push_back(spec);
    }
    return specs;
}

StridedAllGatherMinimalMatmulAsync::tensor_return_value_t StridedAllGatherMinimalMatmulAsync::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // All Gather output tensor
    ttnn::Tensor strided_all_gather_output_tensor = StridedAllGatherAsync::create_output_tensors(
        attributes.strided_all_gather_async_struct,
        StridedAllGatherAsyncInputs{tensor_args.input_tensor, tensor_args.persistent_output_buffer});

    // Matmul outputs: one per chunk (chunks == 1 by default), appended after the all-gather output.
    auto minimal_matmul_output_tensors_vec = matmul_device_operation_t::create_output_tensors(
        attributes.matmul_struct, {strided_all_gather_output_tensor, tensor_args.weight_tensor});

    tensor_return_value_t outputs;
    outputs.reserve(1 + minimal_matmul_output_tensors_vec.size());
    outputs.push_back(strided_all_gather_output_tensor);
    for (auto& t : minimal_matmul_output_tensors_vec) {
        outputs.push_back(t);
    }
    return outputs;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> strided_all_gather_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const uint32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord strided_all_gather_core_grid_offset,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_ag,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    const std::optional<const Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config_mm,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    std::optional<const ttnn::experimental::prim::MinimalMatmulConfig> config,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<bool> read_local_slice_from_input,
    const std::optional<const Tensor>& fused_ternary_input_a,
    const std::optional<const Tensor>& fused_ternary_input_b,
    std::optional<float> fused_ternary_scalar,
    int32_t chunks,
    ttnn::experimental::prim::MMSignalAggregatorMode mm_signal_aggregator_mode,
    bool fuse_swiglu) {
    using OperationType = ttnn::experimental::prim::StridedAllGatherMinimalMatmulAsync;

    // addcmul uses value=1 (torch default) when ternary inputs are given without an explicit scalar
    if (fused_ternary_input_a.has_value() && !fused_ternary_scalar.has_value()) {
        fused_ternary_scalar = 1.0f;
    }

    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    std::vector<IDevice*> devices = ttnn::ccl::get_active_physical_devices(input_tensor);
    if (bias.has_value()) {
        optional_input_tensors.push_back(bias);
    } else {
        optional_input_tensors.push_back(std::nullopt);
    }

    /* AllGather setup */
    uint32_t num_devices = ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    ttnn::experimental::prim::StridedAllGatherAsyncParams strided_all_gather_async_struct =
        ttnn::experimental::prim::StridedAllGatherAsyncParams(
            devices,
            dim,
            num_links,
            num_devices,
            memory_config_ag.value_or(input_tensor.memory_config()),
            topology,
            multi_device_global_semaphore,
            cluster_axis,
            num_workers_per_link,
            num_buffers_per_channel,
            config->compute_with_storage_grid_size.y,
            config->M_block_size,
            config->K_block_size);

    /* Matmul setup */
    auto matmul_struct = decltype(ttnn::experimental::prim::StridedAllGatherMinimalMatmulAsyncParams::matmul_struct){
        .config = config,
        .fused_activation = std::move(fused_activation),
        .output_mem_config = memory_config_mm,
        .fused_ternary_scalar = fused_ternary_scalar,
        .compute_kernel_config = compute_kernel_config.value(),
        .chunks = chunks,
        .fuse_swiglu = fuse_swiglu};
    ttnn::experimental::prim::StridedAllGatherAsync ag_op{};

    bool read_local_from_input = read_local_slice_from_input.value_or(false);

    auto operation_attributes = OperationType::operation_attributes_t{
        strided_all_gather_async_struct,
        matmul_struct,
        strided_all_gather_core_grid_offset,
        read_local_from_input,
        devices,
        ag_op,
        mm_signal_aggregator_mode};
    auto tensor_args = OperationType::tensor_args_t{
        input_tensor, weight_tensor, persistent_output_buffer, bias, fused_ternary_input_a, fused_ternary_input_b};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
