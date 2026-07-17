// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

static void validate_tensor(const Tensor& t, const std::string& name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(t.buffer() != nullptr, "{} must be allocated", name);
    TT_FATAL(t.buffer()->buffer_type() == BufferType::DRAM, "{} must be in DRAM", name);
    TT_FATAL(t.layout() == Layout::TILE, "{} must be tiled", name);
    TT_FATAL(t.dtype() == DataType::FLOAT32, "{} must be float32, got {}", name, t.dtype());
}

void GatedDeltaAttnSeqDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    validate_tensor(in.L_unit, "L_unit");
    validate_tensor(in.v_beta_sc, "v_beta_sc");
    validate_tensor(in.k_bd_sc, "k_bd_sc");
    validate_tensor(in.intra_attn, "intra_attn");
    validate_tensor(in.q_decay, "q_decay");
    validate_tensor(in.k_decay_t, "k_decay_t");
    validate_tensor(in.dl_exp, "dl_exp");
    validate_tensor(in.L_inv, "L_inv");
    if (in.initial_state.has_value()) {
        validate_tensor(*in.initial_state, "initial_state");
    }

    const uint32_t BH = attrs.num_heads;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t C = attrs.chunk_size;
    const uint32_t Dk = attrs.key_dim;
    const uint32_t Dv = attrs.val_dim;

    // The device kernels are hardwired to exactly four 32-wide block rows (Ct == Kt == Vt == 4):
    // the compute kernel calls fwd_sub_4rows unconditionally and the reader always fetches the 4
    // L_inv diagonal blocks. Anything other than 128 reads past L_inv or stalls the compute pipeline
    // waiting on tiles that are never produced, so reject it here until the kernels are generalized.
    constexpr uint32_t kRequiredDim = 4 * tt::constants::TILE_HEIGHT;  // 128
    TT_FATAL(C == kRequiredDim, "chunk_size must be exactly {}, got {}", kRequiredDim, C);
    TT_FATAL(Dk == kRequiredDim, "key_dim must be exactly {}, got {}", kRequiredDim, Dk);
    TT_FATAL(Dv == kRequiredDim, "val_dim must be exactly {}, got {}", kRequiredDim, Dv);

    auto check_shape = [&](const Tensor& t, std::initializer_list<uint32_t> expected, const std::string& nm) {
        auto s = t.logical_shape();
        TT_FATAL(s.rank() == expected.size(), "{} rank mismatch: {} vs {}", nm, s.rank(), expected.size());
        size_t i = 0;
        for (auto e : expected) {
            TT_FATAL(static_cast<uint32_t>(s[i]) == e, "{} dim[{}] expected {} got {}", nm, i, e, s[i]);
            i++;
        }
    };

    check_shape(in.L_unit, {BH, NC, C, C}, "L_unit");
    check_shape(in.v_beta_sc, {BH, NC, C, Dv}, "v_beta_sc");
    check_shape(in.k_bd_sc, {BH, NC, C, Dk}, "k_bd_sc");
    check_shape(in.intra_attn, {BH, NC, C, C}, "intra_attn");
    check_shape(in.q_decay, {BH, NC, C, Dk}, "q_decay");
    check_shape(in.k_decay_t, {BH, NC, Dk, C}, "k_decay_t");
    check_shape(in.dl_exp, {BH, NC, 1, 1}, "dl_exp");
    // L_inv: [BH, NC, C, 32] — 4 diagonal block inverses per chunk (Ct=C/32 tiles × 32 columns)
    check_shape(in.L_inv, {BH, NC, C, tt::constants::TILE_HEIGHT}, "L_inv");
    if (in.initial_state.has_value()) {
        check_shape(*in.initial_state, {BH, Dk, Dv}, "initial_state");
    }

    if (attrs.token_major_output) {
        TT_FATAL(attrs.num_v_heads > 0, "token_major_output requires num_v_heads > 0");
        TT_FATAL(
            BH % attrs.num_v_heads == 0,
            "token_major_output: BH (batch*heads) {} must be divisible by num_v_heads {}",
            BH,
            attrs.num_v_heads);
        TT_FATAL(
            attrs.seq_len > 0 && attrs.seq_len <= NC * C,
            "token_major_output: seq_len {} must be in (0, NC*C={}]",
            attrs.seq_len,
            NC * C);
    }
}

void GatedDeltaAttnSeqDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    validate_on_program_cache_miss(attrs, in);
}

GatedDeltaAttnSeqDeviceOperation::spec_return_value_t GatedDeltaAttnSeqDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, [[maybe_unused]] const tensor_args_t& in) {
    const uint32_t BH = attrs.num_heads;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t C = attrs.chunk_size;
    const uint32_t Dk = attrs.key_dim;
    const uint32_t Dv = attrs.val_dim;
    const auto& mc = attrs.output_mem_config;

    // Output layout: head-major or token-major
    TensorSpec out_spec =
        attrs.token_major_output
            ? TensorSpec(
                  ttnn::Shape({BH / attrs.num_v_heads, attrs.seq_len, attrs.num_v_heads * Dv}),
                  TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), mc))
            : TensorSpec(ttnn::Shape({BH, NC, C, Dv}), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), mc));
    TensorSpec state_spec(ttnn::Shape({BH, Dk, Dv}), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), mc));
    return {out_spec, state_spec};
}

GatedDeltaAttnSeqDeviceOperation::tensor_return_value_t GatedDeltaAttnSeqDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {
        create_device_tensor(specs[0], in.L_unit.device()),
        create_device_tensor(specs[1], in.L_unit.device()),
    };
}

ttsl::hash::hash_t GatedDeltaAttnSeqDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return operation::hash_operation<GatedDeltaAttnSeqDeviceOperation>(
        attrs.num_heads,
        attrs.num_chunks,
        attrs.chunk_size,
        attrs.key_dim,
        attrs.val_dim,
        attrs.output_mem_config,
        attrs.compute_kernel_config,
        attrs.token_major_output,
        attrs.num_v_heads,
        attrs.seq_len,
        in.L_unit,
        in.v_beta_sc,
        in.k_bd_sc,
        in.intra_attn,
        in.q_decay,
        in.k_decay_t,
        in.dl_exp,
        // L_inv and initial_state bake TensorAccessorArgs into the reader, so they must be hashed:
        // otherwise a cache hit can reuse a reader compiled for a different L_inv layout or for absent
        // initial_state. Hashing the optional also distinguishes state present vs. absent.
        in.L_inv,
        in.initial_state);
}

std::vector<Tensor> gated_delta_attn_seq(
    const Tensor& L_unit,
    const Tensor& v_beta_sc,
    const Tensor& k_bd_sc,
    const Tensor& intra_attn,
    const Tensor& q_decay,
    const Tensor& k_decay_t,
    const Tensor& dl_exp,
    const Tensor& L_inv,
    const std::optional<Tensor>& initial_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool token_major_output,
    uint32_t num_v_heads,
    uint32_t seq_len) {
    using Op = GatedDeltaAttnSeqDeviceOperation;
    auto shape = L_unit.logical_shape();
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_heads = static_cast<uint32_t>(shape[0]),
            .num_chunks = static_cast<uint32_t>(shape[1]),
            .chunk_size = static_cast<uint32_t>(shape[2]),
            .key_dim = static_cast<uint32_t>(k_bd_sc.logical_shape()[3]),
            .val_dim = static_cast<uint32_t>(v_beta_sc.logical_shape()[3]),
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
            .token_major_output = token_major_output,
            .num_v_heads = num_v_heads,
            .seq_len = seq_len,
        },
        Op::tensor_args_t{
            .L_unit = L_unit,
            .v_beta_sc = v_beta_sc,
            .k_bd_sc = k_bd_sc,
            .intra_attn = intra_attn,
            .q_decay = q_decay,
            .k_decay_t = k_decay_t,
            .dl_exp = dl_exp,
            .L_inv = L_inv,
            .initial_state = initial_state,
        });
}

}  // namespace ttnn::prim
