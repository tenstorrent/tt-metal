// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generalized_moe_gate_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include <tt-metalium/buffer.hpp>

#include "ttnn/operation.hpp"  // tt::tt_metal::operation::hash_operation

#include "generalized_moe_gate_program_descriptor_builder.hpp"

namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate {

void GeneralizedMoeGateDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

void GeneralizedMoeGateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    using tt::tt_metal::DataType;

    // topk is a COMPILE-TIME arg to the templated finalize kernel, whose rank-mask is correct ONLY for
    // {4, 6, 8} (the values the op tests cover): topk 1-3 fall into the `topk <= 4` branch but leave ranks
    // 0-3 unmasked (the kernel itself notes "topk<4 would also need offset-0 lane masking"), so they would
    // silently normalize/output the first FOUR ranks; topk 5/7 take the masked branch but are untested. Reject
    // everything else here (matches the Python docstring + the TTMoEGate fallback's `_KERNEL_TOPK`) rather
    // than let an unsupported value through to a silently-wrong route.
    TT_FATAL(
        attrs.topk == 4 || attrs.topk == 6 || attrs.topk == 8,
        "topk must be one of {{4, 6, 8}} (the kernel rank-mask + op tests cover only these), got {}",
        attrs.topk);
    // Grouped mode = the DeepSeek grouped gate (kernel `#else`): hardwired 8 groups × 32 → top-8 with linear
    // renorm + scale. It ignores topk/output_softmax and is single-256-block only (num_blocks==1, enforced in
    // the descriptor builder). Require the consistent attrs so a caller can't silently pass values the grouped
    // kernel won't honor.
    if (attrs.grouped) {
        TT_FATAL(attrs.topk == 8, "grouped mode (DeepSeek gate) is hardwired to top-8; got topk={}", attrs.topk);
        TT_FATAL(
            !attrs.output_softmax,
            "grouped mode (DeepSeek gate) normalizes linearly (score/Σ); output_softmax must be false");
    }
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& bias_tensor = tensor_args.bias_tensor;
    const auto& input_indices_tensor = tensor_args.input_indices_tensor;
    const auto& output_tensor = tensor_args.output_tensor;
    const auto& output_indices_tensor = tensor_args.output_indices_tensor;

    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "input_tensor must be on device");
    TT_FATAL(bias_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "bias_tensor must be on device");
    TT_FATAL(
        input_indices_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "input_indices_tensor must be on device");
    TT_FATAL(output_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "output_tensor must be on device");
    TT_FATAL(
        output_indices_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "output_indices_tensor must be on device");

    TT_FATAL(input_tensor.device() == bias_tensor.device(), "All tensors must be on the same device");
    TT_FATAL(input_tensor.device() == input_indices_tensor.device(), "All tensors must be on the same device");
    TT_FATAL(input_tensor.device() == output_tensor.device(), "All tensors must be on the same device");
    TT_FATAL(input_tensor.device() == output_indices_tensor.device(), "All tensors must be on the same device");

    TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "input_tensor must be BFLOAT16");
    TT_FATAL(bias_tensor.dtype() == DataType::BFLOAT16, "bias_tensor must be BFLOAT16");
    TT_FATAL(input_indices_tensor.dtype() == DataType::UINT16, "input_indices_tensor must be UINT16");
    TT_FATAL(output_tensor.dtype() == DataType::BFLOAT16, "output_tensor must be BFLOAT16");
    TT_FATAL(output_indices_tensor.dtype() == DataType::UINT16, "output_indices_tensor must be UINT16");

    TT_FATAL(input_tensor.is_sharded(), "input_tensor must be sharded");
    TT_FATAL(bias_tensor.is_sharded(), "bias_tensor must be sharded");
    TT_FATAL(input_indices_tensor.is_sharded(), "input_indices_tensor must be sharded");
    TT_FATAL(output_tensor.is_sharded(), "output_tensor must be sharded");
    TT_FATAL(output_indices_tensor.is_sharded(), "output_indices_tensor must be sharded");

    const auto& in_shape = input_tensor.logical_shape();
    const auto& bias_shape = bias_tensor.logical_shape();
    const auto& out_shape = output_tensor.logical_shape();
    const auto& in_idx_shape = input_indices_tensor.logical_shape();
    const auto& out_idx_shape = output_indices_tensor.logical_shape();

    TT_FATAL(bias_shape == in_shape, "Bias and input tensors must have the same shape");
    TT_FATAL(out_idx_shape == out_shape, "Output indices and output tensors must have the same shape");

    TT_FATAL(in_shape.size() >= 2, "input_tensor must have rank >= 2");
    uint32_t h = in_shape[in_shape.size() - 2];
    uint32_t w = in_shape[in_shape.size() - 1];
    TT_FATAL(h * w == 256, "Input tensor must have 256 elements per block (last two dims = one 256-block)");
    // input_indices holds one tile per 256-block, each uploaded with GLOBAL expert ids (block b = arange + b*256).
    uint32_t idx_h = in_idx_shape[in_idx_shape.size() - 2];
    uint32_t idx_w = in_idx_shape[in_idx_shape.size() - 1];
    TT_FATAL(idx_h * idx_w == 256, "input_indices must have 256 elements (one block, last two dims)");

    const auto& input_shard = input_tensor.shard_spec().value();
    const auto& output_shard = output_tensor.shard_spec().value();
    const auto& bias_shard = bias_tensor.memory_config().shard_spec().value();
    const auto& in_indices_shard = input_indices_tensor.memory_config().shard_spec().value();
    const auto& out_indices_shard = output_indices_tensor.memory_config().shard_spec().value();

    auto all_cores = input_shard.grid;

    TT_FATAL(input_shard.shape == bias_shard.shape, "Input and bias shard shapes must match");
    TT_FATAL(input_shard.orientation == bias_shard.orientation, "Input and bias shard orientations must match");
    TT_FATAL(bias_shard.grid.contains(all_cores), "Bias shard grid must contain input shard grid");

    TT_FATAL(
        input_shard.shape == in_indices_shard.shape,
        "Input-indices shard shape must equal input shard shape: the kernel consumes num_blocks index tiles "
        "(derived from the input shard), so a mismatched index shard would hang or route with invalid ids "
        "(one 32x32 index tile per 256-block; block b holds global ids)");
    TT_FATAL(
        input_shard.orientation == in_indices_shard.orientation, "Input and input-indices orientations must match");
    TT_FATAL(in_indices_shard.grid.contains(all_cores), "Input-indices shard grid must contain input shard grid");

    TT_FATAL(output_shard.grid == out_indices_shard.grid, "Output and output-indices shard grids must match");
    TT_FATAL(output_shard.shape == out_indices_shard.shape, "Output and output-indices shard shapes must match");
    TT_FATAL(output_shard.orientation == out_indices_shard.orientation, "Output orientations must match");
    TT_FATAL(output_shard.grid.contains(all_cores), "Output shard grid must contain input compute grid");

    const auto& in_tile = input_tensor.tensor_spec().tile();
    const auto& out_tile = output_tensor.tensor_spec().tile();
    TT_FATAL(in_tile == bias_tensor.tensor_spec().tile(), "Input and bias tiles must match");
    TT_FATAL(in_tile == input_indices_tensor.tensor_spec().tile(), "Input and input-indices tiles must match");
    TT_FATAL(out_tile == output_indices_tensor.tensor_spec().tile(), "Output tiles must match");

    TT_FATAL(in_tile.get_height() == 32 && in_tile.get_width() == 32, "Input tile must be 32x32");
    TT_FATAL(out_tile.get_height() == 32 && out_tile.get_width() == 32, "Output tile must be 32x32");
    // input shard holds num_blocks 32x32 tiles (one 256-block per tile); just require tile-alignment.
    TT_FATAL(
        input_shard.shape[0] % 32 == 0 && input_shard.shape[1] % 32 == 0,
        "Input shard must be 32x32 tile-aligned (num_blocks tiles per core)");
    TT_FATAL(output_shard.shape[0] == 32 && output_shard.shape[1] == 32, "Output shard shape must be 32x32");

    // num_blocks = 32x32 tiles per input shard (one 256-expert block per tile): 1 (≤256 experts) or 2 (≤512,
    // the 2-block combine). The kernel handles at most 2; grouped (DeepSeek) is single-256-block only. The
    // descriptor builder derives num_blocks the same way for CB sizing and assumes these bounds hold.
    const uint32_t num_blocks = (input_shard.shape[0] / 32) * (input_shard.shape[1] / 32);
    TT_FATAL(num_blocks >= 1, "input shard must hold at least one 32x32 tile");
    TT_FATAL(num_blocks <= 2, "generalized_moe_gate supports up to 2 blocks (<=512 experts), got {}", num_blocks);
    TT_FATAL(
        !attrs.grouped || num_blocks == 1,
        "grouped mode (DeepSeek gate) is single-256-block only; got num_blocks={}",
        num_blocks);
}

GeneralizedMoeGateDeviceOperation::spec_return_value_t GeneralizedMoeGateDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return {
        tensor_args.output_tensor.tensor_spec(),
        tensor_args.output_indices_tensor.tensor_spec(),
    };
}

GeneralizedMoeGateDeviceOperation::tensor_return_value_t GeneralizedMoeGateDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return {tensor_args.output_tensor, tensor_args.output_indices_tensor};
}

std::uint64_t GeneralizedMoeGateDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Cheap structural hash. Everything that changes the compiled program — the kernel compile-time args
    // (eps/scaling/enable_sigmoid/topk/output_softmax), the GMG_UNGROUPED_TOP8 define (grouped), and the
    // CB sizes/grids — is a function of the op attributes plus the tensor specs (dtype + tile + shard
    // grid/shape; num_blocks is derived from the input shard shape). Everything else the builder emits is
    // constant (kernel path, CB indices, HiFi4, configs). So hash those inputs directly instead of building
    // the full ProgramDescriptor: that path runs the builder's TT_FATALs + constructs every CB/kernel
    // descriptor + strings on EVERY dispatch (including cache hits) just to derive a key. Validation lives in
    // validate_on_program_cache_{hit,miss}; the build happens in create_program. hash_operation<> folds the op
    // TYPE into the key (the per-device program cache is shared across op types) — same as the framework's
    // default hash; previously this was carried implicitly by the kernel-source path in the full descriptor.
    return tt::tt_metal::operation::hash_operation<GeneralizedMoeGateDeviceOperation>(
        attrs.eps,
        attrs.scaling_factor,
        attrs.enable_sigmoid,
        attrs.topk,
        attrs.output_softmax,
        attrs.grouped,
        tensor_args.input_tensor.tensor_spec(),
        tensor_args.bias_tensor.tensor_spec(),
        tensor_args.input_indices_tensor.tensor_spec(),
        tensor_args.output_tensor.tensor_spec(),
        tensor_args.output_indices_tensor.tensor_spec());
}

std::tuple<GeneralizedMoeGateDeviceOperation::operation_attributes_t, GeneralizedMoeGateDeviceOperation::tensor_args_t>
GeneralizedMoeGateDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& bias_tensor,
    const Tensor& input_indices_tensor,
    const Tensor& output_tensor,
    const Tensor& output_indices_tensor,
    float eps,
    float scaling_factor,
    bool enable_sigmoid,
    uint32_t topk,
    bool output_softmax,
    bool grouped) {
    return {
        operation_attributes_t{
            .eps = eps,
            .scaling_factor = scaling_factor,
            .enable_sigmoid = enable_sigmoid,
            .topk = topk,
            .output_softmax = output_softmax,
            .grouped = grouped},
        tensor_args_t{
            .input_tensor = input_tensor,
            .bias_tensor = bias_tensor,
            .input_indices_tensor = input_indices_tensor,
            .output_tensor = output_tensor,
            .output_indices_tensor = output_indices_tensor,
        },
    };
}

}  // namespace ttnn::operations::experimental::deepseek::moe::generalized_moe_gate
