// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_fused_qk_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

void validate_rope_tensor(const Tensor& t, const char* name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(t.buffer() != nullptr, "{} must be allocated in a buffer", name);
    TT_FATAL(t.layout() == Layout::TILE, "{} must be tilized", name);
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{} must be INTERLEAVED (fused-qk RoPE supports interleaved only)",
        name);
}

}  // namespace

void RotaryEmbeddingFusedQKDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& q = tensor_args.q;
    const auto& k = tensor_args.k;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;

    validate_rope_tensor(q, "q");
    validate_rope_tensor(k, "k");
    validate_rope_tensor(cos, "cos");
    validate_rope_tensor(sin, "sin");

    auto* ref_device = q.device();
    TT_FATAL(k.device() == ref_device, "q and k must be on same device");
    TT_FATAL(cos.device() == ref_device, "cos must be on same device");
    TT_FATAL(sin.device() == ref_device, "sin must be on same device");

    // Share one reader/writer kernel for both q and k: identical dtype + buffer type required so
    // the baked TensorAccessor compile-time args are valid for both buffers.
    TT_FATAL(q.dtype() == k.dtype(), "q and k dtypes must match (shared kernel)");
    TT_FATAL(q.buffer()->buffer_type() == k.buffer()->buffer_type(), "q and k buffer types must match (shared kernel)");

    uint32_t head_dim = q.padded_shape()[-1];
    TT_FATAL(k.padded_shape()[-1] == head_dim, "q and k head_dim must match");
    TT_FATAL(
        head_dim % (TILE_WIDTH * 2) == 0,
        "head_dim ({}) must be divisible by {} (rotate_half midpoint must align with a tile boundary)",
        head_dim,
        TILE_WIDTH * 2);

    uint32_t seq_q = q.padded_shape()[-2];
    uint32_t seq_k = k.padded_shape()[-2];
    TT_FATAL(seq_q == seq_k, "q and k seq_len must match ({} vs {})", seq_q, seq_k);

    TT_FATAL(cos.dtype() == sin.dtype(), "cos and sin dtypes must match");
    TT_FATAL(cos.padded_shape() == sin.padded_shape(), "cos and sin dims must match");
    TT_FATAL(
        cos.padded_shape()[0] == 1 && cos.padded_shape()[1] == 1 && cos.padded_shape()[-1] == head_dim,
        "cos/sin must be (1, 1, seq, head_dim={})",
        head_dim);
    TT_FATAL(cos.padded_shape()[-2] >= seq_q, "cos/sin seq must cover input seq_len");

    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "output memory config must be INTERLEAVED");
}

RotaryEmbeddingFusedQKDeviceOperation::spec_return_value_t RotaryEmbeddingFusedQKDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& q = tensor_args.q;
    const auto& k = tensor_args.k;

    auto make_spec = [&](const Tensor& t) {
        auto shape = t.padded_shape();
        shape[-2] = tt::round_up(args.seq_len, TILE_HEIGHT);
        return TensorSpec(
            shape, tt::tt_metal::TensorLayout(t.dtype(), tt::tt_metal::PageConfig(t.layout()), args.output_mem_config));
    };
    return {make_spec(q), make_spec(k)};
}

RotaryEmbeddingFusedQKDeviceOperation::tensor_return_value_t
RotaryEmbeddingFusedQKDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.q.device();
    return {create_device_tensor(std::get<0>(specs), device), create_device_tensor(std::get<1>(specs), device)};
}

ttsl::hash::hash_t RotaryEmbeddingFusedQKDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<RotaryEmbeddingFusedQKDeviceOperation>(
        args.seq_len, args.output_mem_config, tensor_args.q, tensor_args.k, tensor_args.cos, tensor_args.sin);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor> rotary_embedding_fused_qk(
    const Tensor& q,
    const Tensor& k,
    const Tensor& cos,
    const Tensor& sin,
    uint32_t seq_len,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    using OperationType = ttnn::experimental::prim::RotaryEmbeddingFusedQKDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .seq_len = seq_len,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{.q = q, .k = k, .cos = cos, .sin = sin};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
