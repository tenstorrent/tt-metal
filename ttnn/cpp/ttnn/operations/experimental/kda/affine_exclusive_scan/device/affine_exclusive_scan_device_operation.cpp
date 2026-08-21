// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "affine_exclusive_scan_device_operation.hpp"

#include <array>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

AffineExclusiveScanOperation::program_factory_t AffineExclusiveScanOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return AffineExclusiveScanProgramFactory{};
}

void AffineExclusiveScanOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "affine_exclusive_scan";
    constexpr std::array accepted_summary_dtypes = {tt::tt_metal::DataType::FLOAT32, tt::tt_metal::DataType::BFLOAT16};
    kda_factory_detail::check_allocated_device_tensor(in.a, operation_name, "a");
    kda_factory_detail::check_layout(in.a, tt::tt_metal::Layout::TILE, operation_name, "a");
    kda_factory_detail::check_dtype_in(in.a, accepted_summary_dtypes, "FLOAT32 or BFLOAT16", operation_name, "a");
    kda_factory_detail::check_allocated_device_tensor(in.b, operation_name, "b");
    kda_factory_detail::check_layout(in.b, tt::tt_metal::Layout::TILE, operation_name, "b");
    kda_factory_detail::check_dtype_in(in.b, accepted_summary_dtypes, "FLOAT32 or BFLOAT16", operation_name, "b");
    kda_factory_detail::check_allocated_device_tensor(in.initial_state, operation_name, "initial_state");
    kda_factory_detail::check_layout(in.initial_state, tt::tt_metal::Layout::TILE, operation_name, "initial_state");
    kda_factory_detail::check_dtype(in.initial_state, tt::tt_metal::DataType::FLOAT32, operation_name, "initial_state");
    kda_factory_detail::check_same_device(in.a, in.b, operation_name, "b");
    kda_factory_detail::check_same_device(in.a, in.initial_state, operation_name, "initial_state");
    kda_factory_detail::check_matching_dtype(in.a, in.b, operation_name, "a and b");
    TT_FATAL(attrs.groups_per_head > 0, "affine_exclusive_scan: groups_per_head must be positive");
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, operation_name);
    kda_factory_detail::check_compute_config(attrs.compute_kernel_config, operation_name);

    const auto& a_shape = in.a.logical_shape();
    const auto& b_shape = in.b.logical_shape();
    const auto& state_shape = in.initial_state.logical_shape();
    TT_FATAL(
        a_shape.rank() == 3 && b_shape.rank() == 3 && state_shape.rank() == 3,
        "affine_exclusive_scan: inputs must be rank 3");
    TT_FATAL(a_shape[0] > 0, "affine_exclusive_scan: leading dimension must be positive");
    TT_FATAL(
        a_shape[0] % attrs.groups_per_head == 0,
        "affine_exclusive_scan: leading dimension must be divisible by groups_per_head");
    TT_FATAL(a_shape[0] == b_shape[0], "affine_exclusive_scan: a and b must have matching leading dimensions");
    TT_FATAL(a_shape[1] == a_shape[2], "affine_exclusive_scan: a must contain square KxK matrices");
    TT_FATAL(a_shape[1] == b_shape[1], "affine_exclusive_scan: a and b must have matching K dimensions");
    TT_FATAL(
        a_shape[1] > 0 && b_shape[2] > 0 && a_shape[1] % tt::constants::TILE_WIDTH == 0 &&
            b_shape[2] % tt::constants::TILE_WIDTH == 0,
        "affine_exclusive_scan: K and V must be positive and tile aligned");
    TT_FATAL(
        a_shape[0] == attrs.batch_heads * attrs.groups_per_head && a_shape[1] == attrs.key_dim &&
            b_shape[2] == attrs.value_dim,
        "affine_exclusive_scan: input shapes must match operation attributes");
    TT_FATAL(
        state_shape[0] == attrs.batch_heads && state_shape[1] == attrs.key_dim && state_shape[2] == attrs.value_dim,
        "affine_exclusive_scan: initial_state shape must be [batch_heads, K, V]");
}

AffineExclusiveScanOperation::spec_return_value_t AffineExclusiveScanOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    return {tt::tt_metal::TensorSpec(
        tt::tt_metal::Shape({a.batch_heads * a.groups_per_head, a.key_dim, a.value_dim}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            a.output_mem_config))};
}
AffineExclusiveScanOperation::tensor_return_value_t AffineExclusiveScanOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return {ttnn::create_device_tensor(compute_output_specs(a, in)[0], in.a.device())};
}
Tensor affine_exclusive_scan(
    const Tensor& a,
    const Tensor& b,
    const Tensor& state,
    uint32_t groups,
    const tt::tt_metal::MemoryConfig& mem,
    const DeviceComputeKernelConfig& cfg) {
    TT_FATAL(groups > 0, "affine_exclusive_scan: groups_per_head must be positive");
    const auto& shape = a.logical_shape();
    const auto& b_shape = b.logical_shape();
    const auto& state_shape = state.logical_shape();
    TT_FATAL(
        shape.rank() == 3 && b_shape.rank() == 3 && state_shape.rank() == 3,
        "affine_exclusive_scan: inputs must be rank 3");
    TT_FATAL(shape[0] > 0, "affine_exclusive_scan: leading dimension must be positive");
    TT_FATAL(shape[0] % groups == 0, "affine_exclusive_scan: leading dimension must be divisible by groups_per_head");
    auto outputs = ::ttnn::device_operation::launch<AffineExclusiveScanOperation>(
        AffineExclusiveScanParams{
            .batch_heads = static_cast<uint32_t>(shape[0]) / groups,
            .groups_per_head = groups,
            .key_dim = static_cast<uint32_t>(shape[1]),
            .value_dim = static_cast<uint32_t>(b.logical_shape()[2]),
            .output_mem_config = mem,
            .compute_kernel_config = cfg},
        AffineExclusiveScanInputs{.a = a, .b = b, .initial_state = state});
    return outputs[0];
}
}  // namespace ttnn::experimental::prim
