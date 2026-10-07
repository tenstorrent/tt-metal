// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chain_affine_transforms_device_operation.hpp"

#include <array>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

namespace ttnn::experimental::prim {

ChainAffineTransformsOperation::program_factory_t ChainAffineTransformsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChainAffineTransformsProgramFactory{};
}

void ChainAffineTransformsOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "chain_affine_transforms";
    TT_FATAL(
        attrs.local_rows > 0 && attrs.local_rows % tt::constants::TILE_HEIGHT == 0,
        "{}: local_rows must be positive and 32-aligned",
        operation_name);
    kda_factory_detail::check_actual_start(in.transforms, in.actual_start, operation_name);
    kda_factory_detail::check_allocated_device_tensor(in.transforms, operation_name, "transforms");
    TT_FATAL(
        in.transforms.device()->arch() == tt::ARCH::BLACKHOLE,
        "{} is only supported on Blackhole architecture, got {}",
        operation_name,
        in.transforms.device()->arch());
    kda_factory_detail::check_layout(in.transforms, tt::tt_metal::Layout::TILE, operation_name, "transforms");
    // BF16 b widens exactly through srcA; the kernel has no lossless path for FP32 transforms.
    kda_factory_detail::check_dtype(in.transforms, tt::tt_metal::DataType::BFLOAT16, operation_name, "transforms");
    kda_factory_detail::check_interleaved(in.transforms, operation_name, "transforms");
    kda_factory_detail::check_allocated_device_tensor(in.initial_state, operation_name, "initial_state");
    kda_factory_detail::check_layout(in.initial_state, tt::tt_metal::Layout::TILE, operation_name, "initial_state");
    kda_factory_detail::check_dtype(in.initial_state, tt::tt_metal::DataType::FLOAT32, operation_name, "initial_state");
    kda_factory_detail::check_interleaved(in.initial_state, operation_name, "initial_state");
    kda_factory_detail::check_same_device(in.transforms, in.initial_state, operation_name, "initial_state");
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, operation_name);
    kda_factory_detail::check_compute_config(attrs.compute_kernel_config, operation_name);
    TT_FATAL(
        attrs.compute_kernel_config.fp32_dest_acc_en,
        "{}: fp32_dest_acc_en must be enabled; the FP32 product unpacks to DST for the add",
        operation_name);

    const auto& t_shape = in.transforms.logical_shape();
    const auto& s_shape = in.initial_state.logical_shape();
    // The launcher checks both ranks before building the attributes from these shapes.
    TT_FATAL(
        t_shape[1] == s_shape[0] && t_shape[2] == s_shape[1] && t_shape[3] == s_shape[1] + s_shape[2],
        "{}: transforms [P, B*H, K, K + V] must match initial_state [B*H, K, V]",
        operation_name);
    TT_FATAL(s_shape[0] > 0, "{}: B*H must be positive", operation_name);
    TT_FATAL(
        s_shape[1] > 0 && s_shape[2] > 0 && s_shape[1] % tt::constants::TILE_WIDTH == 0 &&
            s_shape[2] % tt::constants::TILE_WIDTH == 0,
        "{}: K and V must be positive and tile aligned",
        operation_name);
    const auto* mesh = in.transforms.device();
    TT_FATAL(attrs.sequence_parallel_axis < mesh->shape().dims(), "{}: invalid sequence_parallel_axis", operation_name);
    TT_FATAL(
        t_shape[0] == mesh->shape()[attrs.sequence_parallel_axis],
        "{}: transforms must hold one transition per sequence-parallel rank",
        operation_name);
    // One core per head.
    const auto grid = mesh->compute_with_storage_grid_size();
    TT_FATAL(
        attrs.batch_heads <= grid.x * grid.y,
        "{}: supports at most {} batch-heads on this device, got {}",
        operation_name,
        grid.x * grid.y,
        attrs.batch_heads);
    const uint64_t dfb_bytes = chain_affine_transforms_l1_bytes(
        attrs.key_dim / tt::constants::TILE_WIDTH, attrs.value_dim / tt::constants::TILE_WIDTH);
    const uint64_t l1_bytes =
        mesh->l1_size_per_core() - mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    TT_FATAL(
        dfb_bytes <= l1_bytes,
        "{}: K = {} and V = {} need {} bytes of L1 per core, only {} are available",
        operation_name,
        attrs.key_dim,
        attrs.value_dim,
        dfb_bytes,
        l1_bytes);
}

ChainAffineTransformsOperation::spec_return_value_t ChainAffineTransformsOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto spec = tt::tt_metal::TensorSpec(
        Shape({attrs.batch_heads, attrs.key_dim, attrs.value_dim}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            attrs.output_mem_config));
    return {spec, spec};
}

ChainAffineTransformsOperation::tensor_return_value_t ChainAffineTransformsOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {
        create_device_tensor(specs[0], in.transforms.device()), create_device_tensor(specs[1], in.transforms.device())};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<ChainAffineTransformsOperation::tensor_return_value_t>
ChainAffineTransformsOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    using namespace kda_performance_model;
    const double head_steps = static_cast<double>(attrs.steps) * attrs.batch_heads;
    const double key_dim = attrs.key_dim;
    const double value_dim = attrs.value_dim;
    const KdaFpuWork work{
        .fpu_matrix_flops = head_steps * 2.0 * key_dim * key_dim * value_dim,
        .fpu_add_ops = head_steps * key_dim * value_dim,
    };
    const std::array<const Tensor*, 2> inputs = {&in.transforms, &in.initial_state};
    return make_profiler_model(work, inputs, outputs, attrs.compute_kernel_config.math_fidelity);
}

std::pair<Tensor, Tensor> chain_affine_transforms(
    const Tensor& transforms,
    const Tensor& initial_state,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const Tensor& actual_start,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows) {
    // Cache-miss validation cannot protect attribute construction on cache hits. Keep these guards here because the
    // launcher indexes both shapes before dispatching validation.
    const auto& t_shape = transforms.logical_shape();
    const auto& s_shape = initial_state.logical_shape();
    TT_FATAL(t_shape.rank() == 4, "chain_affine_transforms: transforms must be rank 4 [P, B*H, K, K + V]");
    TT_FATAL(s_shape.rank() == 3, "chain_affine_transforms: initial_state must be rank 3 [B*H, K, V]");
    auto outputs = ttnn::device_operation::launch<ChainAffineTransformsOperation>(
        ChainAffineTransformsParams{
            .steps = static_cast<uint32_t>(t_shape[0]),
            .batch_heads = static_cast<uint32_t>(s_shape[0]),
            .key_dim = static_cast<uint32_t>(s_shape[1]),
            .value_dim = static_cast<uint32_t>(s_shape[2]),
            .sequence_parallel_axis = sequence_parallel_axis,
            .local_rows = local_rows,
            .output_mem_config = memory_config,
            .compute_kernel_config = compute_kernel_config},
        ChainAffineTransformsInputs{
            .transforms = transforms, .initial_state = initial_state, .actual_start = actual_start});
    return {outputs[0], outputs[1]};
}

}  // namespace ttnn::experimental::prim
