// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_preprocess_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

static void validate_f32_dram_tile(const Tensor& t, const std::string& name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(t.buffer() != nullptr, "{} must be allocated", name);
    TT_FATAL(t.buffer()->buffer_type() == BufferType::DRAM, "{} must be in DRAM", name);
    TT_FATAL(t.layout() == Layout::TILE, "{} must be tiled", name);
    TT_FATAL(t.dtype() == DataType::FLOAT32, "{} must be float32, got {}", name, t.dtype());
}

void GatedDeltaAttnPreprocessDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    validate_f32_dram_tile(in.q, "q");
    validate_f32_dram_tile(in.k, "k");
    validate_f32_dram_tile(in.v, "v");
    validate_f32_dram_tile(in.beta, "beta");
    validate_f32_dram_tile(in.g, "g");
    validate_f32_dram_tile(in.triu_ones, "triu_ones");
    validate_f32_dram_tile(in.tril_mask, "tril_mask");
    validate_f32_dram_tile(in.eye, "eye");
    validate_f32_dram_tile(in.lower_causal, "lower_causal");
    validate_f32_dram_tile(in.eye_32, "eye_32");

    constexpr uint32_t kRequiredDim = 4 * tt::constants::TILE_HEIGHT;  // 128
    TT_FATAL(attrs.chunk_size == kRequiredDim, "chunk_size must be {}, got {}", kRequiredDim, attrs.chunk_size);
    TT_FATAL(attrs.key_dim == kRequiredDim, "key_dim must be {}, got {}", kRequiredDim, attrs.key_dim);
    TT_FATAL(attrs.val_dim == kRequiredDim, "val_dim must be {}, got {}", kRequiredDim, attrs.val_dim);

    const auto q_shape = in.q.logical_shape();
    const uint32_t BH = attrs.num_heads;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t C = attrs.chunk_size;
    const uint32_t Dk = attrs.key_dim;
    const uint32_t Dv = attrs.val_dim;
    const uint32_t L = NC * C;

    TT_FATAL(q_shape.rank() == 3, "q must be rank 3");
    TT_FATAL(static_cast<uint32_t>(q_shape[0]) == BH, "q dim0 mismatch");
    TT_FATAL(static_cast<uint32_t>(q_shape[1]) == L, "q dim1 must equal NC*chunk_size");
    TT_FATAL(static_cast<uint32_t>(q_shape[2]) == Dk, "q dim2 mismatch");

    auto check_shape = [](const Tensor& t, std::initializer_list<uint32_t> expected, const std::string& nm) {
        auto s = t.logical_shape();
        TT_FATAL(s.rank() == expected.size(), "{} rank mismatch: {} vs {}", nm, s.rank(), expected.size());
        size_t i = 0;
        for (auto e : expected) {
            TT_FATAL(static_cast<uint32_t>(s[i]) == e, "{} dim[{}] expected {} got {}", nm, i, e, s[i]);
            i++;
        }
    };

    check_shape(in.k, {BH, L, Dk}, "k");
    check_shape(in.v, {BH, L, Dv}, "v");
    check_shape(in.beta, {BH, L, 1}, "beta");
    check_shape(in.g, {BH, L, 1}, "g");
    check_shape(in.triu_ones, {1, C, C}, "triu_ones");
    check_shape(in.tril_mask, {1, C, C}, "tril_mask");
    check_shape(in.eye, {1, C, C}, "eye");
    check_shape(in.lower_causal, {1, C, C}, "lower_causal");
    check_shape(in.eye_32, {1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}, "eye_32");
}

void GatedDeltaAttnPreprocessDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    validate_on_program_cache_miss(attrs, in);
}

GatedDeltaAttnPreprocessDeviceOperation::spec_return_value_t
GatedDeltaAttnPreprocessDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, [[maybe_unused]] const tensor_args_t& in) {
    const uint32_t BH = attrs.num_heads;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t C = attrs.chunk_size;
    const uint32_t Dk = attrs.key_dim;
    const uint32_t Dv = attrs.val_dim;
    const auto& mc = attrs.output_mem_config;
    const DataType value_dtype = attrs.bf16_value_path ? DataType::BFLOAT16 : DataType::FLOAT32;

    return {
        TensorSpec(ttnn::Shape({BH, NC, C, C}), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), mc)),
        TensorSpec(ttnn::Shape({BH, NC, C, Dv}), TensorLayout(value_dtype, PageConfig(Layout::TILE), mc)),
        TensorSpec(ttnn::Shape({BH, NC, C, Dk}), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), mc)),
        TensorSpec(ttnn::Shape({BH, NC, C, C}), TensorLayout(value_dtype, PageConfig(Layout::TILE), mc)),
        TensorSpec(ttnn::Shape({BH, NC, C, Dk}), TensorLayout(value_dtype, PageConfig(Layout::TILE), mc)),
        TensorSpec(ttnn::Shape({BH, NC, Dk, C}), TensorLayout(value_dtype, PageConfig(Layout::TILE), mc)),
        TensorSpec(ttnn::Shape({BH, NC, 1, 1}), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), mc)),
        TensorSpec(
            ttnn::Shape({BH, NC, C, tt::constants::TILE_HEIGHT}),
            TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), mc)),
    };
}

GatedDeltaAttnPreprocessDeviceOperation::tensor_return_value_t
GatedDeltaAttnPreprocessDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    std::vector<Tensor> outputs;
    outputs.reserve(specs.size());
    for (const auto& spec : specs) {
        outputs.push_back(create_device_tensor(spec, in.q.device()));
    }
    return outputs;
}

ttsl::hash::hash_t GatedDeltaAttnPreprocessDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return operation::hash_operation<GatedDeltaAttnPreprocessDeviceOperation>(
        attrs.num_heads,
        attrs.num_chunks,
        attrs.chunk_size,
        attrs.key_dim,
        attrs.val_dim,
        attrs.diag_alpha,
        attrs.bf16_value_path,
        attrs.output_mem_config,
        attrs.compute_kernel_config,
        in.q,
        in.k,
        in.v,
        in.beta,
        in.g,
        in.triu_ones,
        in.tril_mask,
        in.eye,
        in.lower_causal,
        in.eye_32);
}

std::vector<Tensor> gated_delta_attn_preprocess(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& beta,
    const Tensor& g,
    const Tensor& triu_ones,
    const Tensor& tril_mask,
    const Tensor& eye,
    const Tensor& lower_causal,
    const Tensor& eye_32,
    uint32_t chunk_size,
    float diag_alpha,
    bool bf16_value_path,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using Op = GatedDeltaAttnPreprocessDeviceOperation;
    auto shape = q.logical_shape();
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_heads = static_cast<uint32_t>(shape[0]),
            .num_chunks = static_cast<uint32_t>(shape[1]) / chunk_size,
            .chunk_size = chunk_size,
            .key_dim = static_cast<uint32_t>(shape[2]),
            .val_dim = static_cast<uint32_t>(v.logical_shape()[2]),
            .diag_alpha = diag_alpha,
            .bf16_value_path = bf16_value_path,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
        },
        Op::tensor_args_t{
            .q = q,
            .k = k,
            .v = v,
            .beta = beta,
            .g = g,
            .triu_ones = triu_ones,
            .tril_mask = tril_mask,
            .eye = eye,
            .lower_causal = lower_causal,
            .eye_32 = eye_32,
        });
}

}  // namespace ttnn::prim
