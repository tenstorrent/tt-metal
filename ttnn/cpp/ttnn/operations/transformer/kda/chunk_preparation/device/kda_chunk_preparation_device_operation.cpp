// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_chunk_preparation_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {

void validate_chunk_preparation_tensor(const Tensor& tensor, const char* name, DataType dtype) {
    TT_FATAL(tensor.layout() == Layout::TILE, "kda_chunk_preparation: {} must be TILE layout", name);
    TT_FATAL(tensor.dtype() == dtype, "kda_chunk_preparation: {} has wrong dtype", name);
    TT_FATAL(tensor.buffer() != nullptr, "kda_chunk_preparation: {} must be on device", name);
}

void validate_chunk_preparation_gate(const Tensor& tensor) {
    TT_FATAL(tensor.layout() == Layout::TILE, "kda_chunk_preparation: g must be TILE layout");
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "kda_chunk_preparation: g must be FLOAT32 or BFLOAT16");
    TT_FATAL(tensor.buffer() != nullptr, "kda_chunk_preparation: g must be on device");
}

}  // namespace

KdaChunkPreparationOperation::program_factory_t KdaChunkPreparationOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaChunkPreparationProgramFactory{};
}

void KdaChunkPreparationOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    validate_chunk_preparation_tensor(in.q, "q", DataType::BFLOAT16);
    validate_chunk_preparation_tensor(in.k, "k", DataType::BFLOAT16);
    validate_chunk_preparation_tensor(in.v, "v", DataType::BFLOAT16);
    validate_chunk_preparation_gate(in.g);
    validate_chunk_preparation_tensor(in.beta, "beta", DataType::FLOAT32);
    validate_chunk_preparation_tensor(in.eye, "eye", DataType::FLOAT32);
    validate_chunk_preparation_tensor(in.tril, "tril", DataType::FLOAT32);
    validate_chunk_preparation_tensor(in.ones, "ones", DataType::FLOAT32);
    validate_chunk_preparation_tensor(in.masks, "masks", DataType::FLOAT32);

    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.value_dim % TILE_WIDTH == 0, "value_dim must be a multiple of 32");
    TT_FATAL(attrs.batch_heads > 0 && attrs.num_chunks > 0, "batch-head and chunk counts must be positive");

    if (attrs.v_flat) {
        const auto& shape = in.v.logical_shape();
        TT_FATAL(attrs.value_heads > 0, "v_flat requires value_heads > 0");
        TT_FATAL(shape.rank() == 3, "v_flat expects [B,T,Hv*V]");
        TT_FATAL(shape[2] == attrs.value_heads * attrs.value_dim, "flat v width mismatch");
    }
    if (attrs.qk_flat) {
        const auto& q_shape = in.q.logical_shape();
        TT_FATAL(attrs.key_heads > 0, "qk_flat requires key_heads > 0");
        TT_FATAL(q_shape.rank() == 3 && in.k.logical_shape() == q_shape, "qk_flat expects matching [B,T,Hk*K]");
        TT_FATAL(q_shape[2] == attrs.key_heads * attrs.key_dim, "flat q/k width mismatch");
        TT_FATAL(attrs.normalize_qk, "qk_flat requires in-kernel Q/K normalization");
    }
    if (attrs.gate_flat) {
        const auto& shape = in.g.logical_shape();
        TT_FATAL(attrs.value_heads > 0, "gate_flat requires value_heads > 0");
        TT_FATAL(shape.rank() == 3, "gate_flat expects [B,T,Hv*K]");
        TT_FATAL(shape[2] == attrs.value_heads * attrs.key_dim, "flat gate width mismatch");
    }
    constexpr uint32_t allowed_bf16_mask = 0x37;
    TT_FATAL(
        (attrs.output_bf16_mask & ~allowed_bf16_mask) == 0,
        "unsupported KDA prep BF16 mask 0x{:x}",
        attrs.output_bf16_mask);
}

KdaChunkPreparationOperation::spec_return_value_t KdaChunkPreparationOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto spec = [&](const Shape& shape, uint32_t index) {
        const auto dtype = (attrs.output_bf16_mask & (1U << index)) ? DataType::BFLOAT16 : DataType::FLOAT32;
        return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), attrs.output_mem_config));
    };
    const auto BH = attrs.batch_heads;
    const auto NC = attrs.num_chunks;
    const auto C = attrs.chunk_size;
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

KdaChunkPreparationOperation::tensor_return_value_t KdaChunkPreparationOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.q.device()));
    }
    return outputs;
}

std::vector<Tensor> kda_chunk_preparation(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye,
    const Tensor& tril,
    const Tensor& ones,
    const Tensor& masks,
    uint32_t chunk_size,
    const MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool v_flat,
    uint32_t value_heads,
    bool normalize_qk,
    float scale,
    bool qk_flat,
    uint32_t key_heads,
    bool gate_flat,
    uint32_t output_bf16_mask) {
    const auto& q_shape = q.logical_shape();
    const auto& v_shape = v.logical_shape();
    TT_FATAL(!qk_flat || value_heads > 0, "qk_flat requires value_heads > 0");
    TT_FATAL(!qk_flat || key_heads > 0, "qk_flat requires key_heads > 0");
    TT_FATAL(!v_flat || value_heads > 0, "v_flat requires value_heads > 0");
    const uint32_t batch_heads = qk_flat ? q_shape[0] * value_heads : q_shape[0];
    const uint32_t num_chunks = qk_flat ? q_shape[1] / chunk_size : q_shape[1];
    const uint32_t key_dim = qk_flat ? q_shape[2] / key_heads : q_shape[3];
    const uint32_t value_dim = v_flat ? v_shape[2] / value_heads : v_shape[3];
    return ttnn::device_operation::launch<KdaChunkPreparationOperation>(
        KdaChunkPreparationParams{
            .batch_heads = batch_heads,
            .num_chunks = num_chunks,
            .chunk_size = chunk_size,
            .key_dim = key_dim,
            .value_dim = value_dim,
            .v_flat = v_flat,
            .value_heads = value_heads,
            .qk_flat = qk_flat,
            .key_heads = key_heads,
            .gate_flat = gate_flat,
            .normalize_qk = normalize_qk,
            .scale = scale,
            .output_bf16_mask = output_bf16_mask,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config},
        KdaChunkPreparationInputs{
            .q = q, .k = k, .v = v, .g = g, .beta = beta, .eye = eye, .tril = tril, .ones = ones, .masks = masks});
}

}  // namespace ttnn::prim
