// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "affine_exclusive_scan_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
namespace {
void check_device_tensor(const Tensor& tensor, const char* name) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "affine_exclusive_scan: {} must be an allocated device tensor",
        name);
    TT_FATAL(tensor.layout() == Layout::TILE, "affine_exclusive_scan: {} must use TILE layout", name);
    TT_FATAL(!tensor.is_sharded(), "affine_exclusive_scan: {} must use interleaved memory", name);
}

void check_affine_tensor(const Tensor& tensor, const char* name) {
    check_device_tensor(tensor, name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "affine_exclusive_scan: {} must be FLOAT32 or BFLOAT16",
        name);
}
}  // namespace

AffineExclusiveScanOperation::program_factory_t AffineExclusiveScanOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return AffineExclusiveScanProgramFactory{};
}

void AffineExclusiveScanOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check_affine_tensor(in.a, "a");
    check_affine_tensor(in.b, "b");
    check_device_tensor(in.initial_state, "initial_state");
    TT_FATAL(
        in.a.device() == in.b.device() && in.a.device() == in.initial_state.device(),
        "affine_exclusive_scan: all inputs must be on the same device");
    TT_FATAL(in.a.dtype() == in.b.dtype(), "affine_exclusive_scan: a and b must have matching dtypes");
    TT_FATAL(in.initial_state.dtype() == DataType::FLOAT32, "affine_exclusive_scan: initial_state must be FLOAT32");
    TT_FATAL(attrs.groups_per_head > 0, "affine_exclusive_scan: groups_per_head must be positive");
    TT_FATAL(
        !attrs.output_mem_config.is_sharded(),
        "affine_exclusive_scan: output memory configuration must be interleaved");

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
    return {TensorSpec(
        Shape({a.batch_heads * a.groups_per_head, a.key_dim, a.value_dim}),
        TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), a.output_mem_config))};
}
AffineExclusiveScanOperation::tensor_return_value_t AffineExclusiveScanOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(a, in)[0], in.a.device())};
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
