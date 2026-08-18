// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prepare_chunk_recurrence_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
namespace {

void check_device_tensor(const Tensor& tensor, const char* name, DataType dtype) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "prepare_chunk_recurrence: {} must be an allocated device tensor",
        name);
    TT_FATAL(tensor.layout() == Layout::TILE, "prepare_chunk_recurrence: {} must use TILE layout", name);
    TT_FATAL(tensor.dtype() == dtype, "prepare_chunk_recurrence: {} has wrong dtype", name);
    TT_FATAL(!tensor.is_sharded(), "prepare_chunk_recurrence: {} must use interleaved memory", name);
}

}  // namespace

PrepareChunkRecurrenceOperation::program_factory_t PrepareChunkRecurrenceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PrepareChunkRecurrenceProgramFactory{};
}

void PrepareChunkRecurrenceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check_device_tensor(in.q, "q", DataType::BFLOAT16);
    check_device_tensor(in.k, "k", DataType::BFLOAT16);
    check_device_tensor(in.v, "v", DataType::BFLOAT16);
    check_device_tensor(in.g, "g", DataType::BFLOAT16);
    check_device_tensor(in.beta, "beta", DataType::FLOAT32);
    check_device_tensor(in.eye, "eye", DataType::FLOAT32);
    check_device_tensor(in.tril, "tril", DataType::FLOAT32);
    check_device_tensor(in.ones, "ones", DataType::FLOAT32);
    check_device_tensor(in.masks, "masks", DataType::FLOAT32);

    TT_FATAL(
        in.q.device() == in.k.device() && in.q.device() == in.v.device() && in.q.device() == in.g.device() &&
            in.q.device() == in.beta.device() && in.q.device() == in.eye.device() &&
            in.q.device() == in.tril.device() && in.q.device() == in.ones.device() &&
            in.q.device() == in.masks.device(),
        "prepare_chunk_recurrence: all inputs must be on the same device");
    TT_FATAL(!attrs.output_mem_config.is_sharded(), "prepare_chunk_recurrence: output memory must be interleaved");
    TT_FATAL(
        !attrs.compute_kernel_config.packer_l1_acc,
        "prepare_chunk_recurrence: packer_l1_acc=true is unsupported because the compute kernel does not accumulate "
        "through L1");
    TT_FATAL(
        attrs.compute_kernel_config.throttle_level ==
            ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE,
        "prepare_chunk_recurrence: compute throttling is unsupported because this kernel does not implement throttled "
        "math");

    const auto& q_shape = in.q.logical_shape();
    const auto& k_shape = in.k.logical_shape();
    const auto& v_shape = in.v.logical_shape();
    const auto& g_shape = in.g.logical_shape();
    const auto& beta_shape = in.beta.logical_shape();
    TT_FATAL(
        q_shape.rank() == 3 && k_shape.rank() == 3 && v_shape.rank() == 3 && g_shape.rank() == 3,
        "prepare_chunk_recurrence: q, k, v, and g must be rank 3 production-flat tensors");
    TT_FATAL(
        q_shape[0] == 1 && k_shape[0] == 1 && v_shape[0] == 1 && g_shape[0] == 1,
        "prepare_chunk_recurrence: q, k, v, and g must have leading dimension 1");
    TT_FATAL(
        k_shape == q_shape && g_shape == q_shape, "prepare_chunk_recurrence: q, k, and g must have matching shapes");
    TT_FATAL(v_shape[1] == q_shape[1], "prepare_chunk_recurrence: q, k, v, and g must have matching sequence lengths");
    TT_FATAL(
        q_shape[1] > 0 && q_shape[1] % tt::constants::TILE_HEIGHT == 0,
        "prepare_chunk_recurrence: sequence length must be positive and divisible by 32");
    TT_FATAL(attrs.num_heads > 0, "prepare_chunk_recurrence: num_heads must be positive");
    TT_FATAL(
        q_shape[2] % attrs.num_heads == 0 && v_shape[2] % attrs.num_heads == 0,
        "prepare_chunk_recurrence: flat widths must be divisible by num_heads");
    TT_FATAL(
        attrs.key_dim > 0 && attrs.value_dim > 0 && attrs.key_dim % tt::constants::TILE_WIDTH == 0 &&
            attrs.value_dim % tt::constants::TILE_WIDTH == 0,
        "prepare_chunk_recurrence: K and V must be positive and tile aligned");
    TT_FATAL(
        q_shape[2] == attrs.num_heads * attrs.key_dim && v_shape[2] == attrs.num_heads * attrs.value_dim &&
            q_shape[1] == attrs.num_chunks * tt::constants::TILE_HEIGHT,
        "prepare_chunk_recurrence: flat input shapes must match operation attributes");

    TT_FATAL(
        beta_shape.rank() == 4 && beta_shape[0] == attrs.num_heads && beta_shape[1] == attrs.num_chunks &&
            beta_shape[2] == tt::constants::TILE_HEIGHT && beta_shape[3] == 1,
        "prepare_chunk_recurrence: beta shape must be [num_heads, num_chunks, 32, 1]");
    const Shape constant_shape({1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
    TT_FATAL(in.eye.logical_shape() == constant_shape, "prepare_chunk_recurrence: eye shape must be [1, 1, 32, 32]");
    TT_FATAL(in.tril.logical_shape() == constant_shape, "prepare_chunk_recurrence: tril shape must be [1, 1, 32, 32]");
    TT_FATAL(in.ones.logical_shape() == constant_shape, "prepare_chunk_recurrence: ones shape must be [1, 1, 32, 32]");
    TT_FATAL(
        in.masks.logical_shape() == Shape({1, 1, tt::constants::TILE_HEIGHT, 3 * tt::constants::TILE_WIDTH}),
        "prepare_chunk_recurrence: masks shape must be [1, 1, 32, 96]");
    constexpr uint32_t allowed_bf16_mask = 0x37;
    TT_FATAL(
        (attrs.output_bf16_mask & ~allowed_bf16_mask) == 0,
        "unsupported KDA prep BF16 mask 0x{:x}",
        attrs.output_bf16_mask);
}

PrepareChunkRecurrenceOperation::spec_return_value_t PrepareChunkRecurrenceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto spec = [&](const Shape& shape, uint32_t index) {
        const auto dtype = (attrs.output_bf16_mask & (1U << index)) ? DataType::BFLOAT16 : DataType::FLOAT32;
        return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), attrs.output_mem_config));
    };
    const auto BH = attrs.num_heads;
    const auto NC = attrs.num_chunks;
    constexpr uint32_t C = tt::constants::TILE_HEIGHT;
    const auto K = attrs.key_dim;
    const auto V = attrs.value_dim;
    return {
        spec(Shape({BH, NC, C, V}), 0),
        spec(Shape({BH, NC, C, K}), 1),
        spec(Shape({BH, NC, C, K}), 2),
        spec(Shape({BH, NC, C, C}), 3),
        spec(Shape({BH, NC, K, C}), 4),
        spec(Shape({BH, NC, K, 1}), 5),
        spec(Shape({BH, NC, C, C}), 6)};
}

PrepareChunkRecurrenceOperation::tensor_return_value_t PrepareChunkRecurrenceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.q.device()));
    }
    return outputs;
}

std::vector<Tensor> prepare_chunk_recurrence(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye,
    const Tensor& tril,
    const Tensor& ones,
    const Tensor& masks,
    uint32_t num_heads,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t output_bf16_mask) {
    const auto& q_shape = q.logical_shape();
    const auto& v_shape = v.logical_shape();
    TT_FATAL(
        q_shape.rank() == 3 && v_shape.rank() == 3,
        "prepare_chunk_recurrence: q and v must be rank 3 production-flat tensors");
    TT_FATAL(num_heads > 0, "prepare_chunk_recurrence: num_heads must be positive");
    TT_FATAL(
        q_shape[1] > 0 && q_shape[1] % tt::constants::TILE_HEIGHT == 0,
        "prepare_chunk_recurrence: sequence length must be positive and divisible by 32");
    TT_FATAL(
        q_shape[2] % num_heads == 0 && v_shape[2] % num_heads == 0,
        "prepare_chunk_recurrence: flat widths must be divisible by num_heads");
    const uint32_t num_chunks = q_shape[1] / tt::constants::TILE_HEIGHT;
    const uint32_t key_dim = q_shape[2] / num_heads;
    const uint32_t value_dim = v_shape[2] / num_heads;
    return ttnn::device_operation::launch<PrepareChunkRecurrenceOperation>(
        PrepareChunkRecurrenceParams{
            .num_heads = num_heads,
            .num_chunks = num_chunks,
            .key_dim = key_dim,
            .value_dim = value_dim,
            .output_bf16_mask = output_bf16_mask,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        PrepareChunkRecurrenceInputs{
            .q = q, .k = k, .v = v, .g = g, .beta = beta, .eye = eye, .tril = tril, .ones = ones, .masks = masks});
}

}  // namespace ttnn::experimental::prim
