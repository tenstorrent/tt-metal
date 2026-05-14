// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_rmsnorm_pre_all_gather_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {
void FusedRMSNormPreAllGatherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    const auto& tensor = tensor_args.input_tensor;

    TT_FATAL(tensor.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", tensor.layout());
    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor must use INTERLEAVED memory layout, got: {}",
        tensor.memory_config().memory_layout());
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "Input tensor must be BFLOAT16, got: {}", tensor.dtype());
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device, got: {}", tensor.storage_type());
    TT_FATAL(tensor.buffer() != nullptr, "Input tensor must be allocated in device buffers (buffer is null)");

    TT_FATAL(args.num_heads >= 1, "num_heads must be >= 1, got: {}", args.num_heads);
    if (args.num_heads > 1) {
        // Per-head stats path: input last dim must split evenly into num_heads, and each
        // per-head slice (head_dim) must be a multiple of TILE_WIDTH so the reduce inside
        // the kernel covers a whole number of tile columns per head.
        const auto& padded = tensor.padded_shape();
        TT_FATAL(
            padded[-1] % args.num_heads == 0,
            "Input last dim {} must be divisible by num_heads {}",
            padded[-1],
            args.num_heads);
        const uint32_t head_dim = padded[-1] / args.num_heads;
        TT_FATAL(
            head_dim % TILE_WIDTH == 0,
            "Per-head head_dim {} must be a multiple of TILE_WIDTH ({}); got num_heads={}, hidden={}",
            head_dim,
            TILE_WIDTH,
            args.num_heads,
            padded[-1]);
    }
}

TensorSpec FusedRMSNormPreAllGatherDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& input_tensor = tensor_args.input_tensor;

    auto output_shape = input_tensor.logical_shape();
    // num_heads=1 → one stat tile per row (legacy behavior). num_heads>1 → one stat tile per
    // head per row, packed contiguously along the last dim. Each tile still holds the
    // sum-of-squares for its head in the leftmost column (rest is padding).
    const uint32_t num_tiles_w = args.num_heads;
    output_shape[-1] = num_tiles_w * TILE_WIDTH;

    return TensorSpec(output_shape, TensorLayout(args.dtype, PageConfig(Layout::TILE), input_tensor.memory_config()));
}

Tensor FusedRMSNormPreAllGatherDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor fused_rmsnorm_pre_all_gather(
    const Tensor& input_tensor,
    tt::tt_metal::DataType dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t num_heads) {
    using OperationType = ttnn::experimental::prim::FusedRMSNormPreAllGatherDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .dtype = dtype,
        .compute_kernel_config = compute_kernel_config,
        .num_heads = num_heads,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
