// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gdn_phased.hpp"

#include <cstdlib>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
void check(const Tensor& t, const char* name, DataType dt) {
    TT_FATAL(t.layout() == Layout::TILE, "chunk_gdn: {} must be TILE layout", name);
    TT_FATAL(t.dtype() == dt, "chunk_gdn: {} has wrong dtype", name);
    TT_FATAL(t.buffer() != nullptr, "chunk_gdn: {} must be on device", name);
}
}  // namespace

// ---------------------------------------------------------------------------
// PREP
// ---------------------------------------------------------------------------
ChunkGdnPrepOperation::program_factory_t ChunkGdnPrepOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChunkGdnPrepProgramFactory{};
}

void ChunkGdnPrepOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    check(in.q, "q", DataType::BFLOAT16);
    check(in.k, "k", DataType::BFLOAT16);
    check(in.v, "v", DataType::BFLOAT16);  // flat [B,T,HV*V] when attrs.v_flat; else [BH,NC,C,V]
    if (attrs.v_flat) {
        TT_FATAL(attrs.HV > 0, "v_flat requires HV > 0");
        const auto& vs = in.v.logical_shape();
        TT_FATAL(vs.rank() == 3, "v_flat expects a flat [B,T,HV*V] v (got rank {})", vs.rank());
        TT_FATAL(vs[2] == attrs.HV * attrs.val_dim, "v_flat width {} != HV*V ({}*{})", vs[2], attrs.HV, attrs.val_dim);
    }
    if (attrs.qk_flat) {
        TT_FATAL(attrs.Hk > 0, "qk_flat requires Hk > 0");
        const auto& qsf = in.q.logical_shape();
        TT_FATAL(qsf.rank() == 3, "qk_flat expects a flat [B,T,Hk*K] q (got rank {})", qsf.rank());
        TT_FATAL(
            qsf[2] == attrs.Hk * attrs.key_dim, "qk_flat width {} != Hk*K ({}*{})", qsf[2], attrs.Hk, attrs.key_dim);
        TT_FATAL(attrs.qk_norm, "qk_flat requires qk_norm (flat q/k are unnormalized; norm is in-kernel)");
    }
    check(in.g, "g", DataType::FLOAT32);
    check(in.beta, "beta", DataType::FLOAT32);
    check(in.eye_c, "eye_c", DataType::FLOAT32);
    check(in.tril_c, "tril_c", DataType::FLOAT32);
    check(in.ones_c, "ones_c", DataType::FLOAT32);
    check(in.masks_c, "masks_c", DataType::FLOAT32);
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
}

ChunkGdnPrepOperation::spec_return_value_t ChunkGdnPrepOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto f32 = [&](const ttnn::Shape& s) {
        return tt::tt_metal::TensorSpec(
            s, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config));
    };
    const uint32_t BH = attrs.BH, NC = attrs.num_chunks, C = attrs.chunk_size, K = attrs.key_dim, V = attrs.val_dim;
    return {
        f32(ttnn::Shape({BH, NC, C, V})),  // v_beta
        f32(ttnn::Shape({BH, NC, C, K})),  // kd
        f32(ttnn::Shape({BH, NC, C, K})),  // q_decay
        f32(ttnn::Shape({BH, NC, C, C})),  // intra
        f32(ttnn::Shape({BH, NC, K, C})),  // k_dec_t
        f32(ttnn::Shape({BH, NC, 1, 1})),  // dl (1 tile per chunk)
        f32(ttnn::Shape({BH, NC, C, C})),  // t_inv
    };
}

ChunkGdnPrepOperation::tensor_return_value_t ChunkGdnPrepOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.q.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, device));
    }
    return outs;
}

std::vector<Tensor> chunk_gdn_prep(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye_c,
    const Tensor& tril_c,
    const Tensor& ones_c,
    const Tensor& masks_c,
    uint32_t chunk_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool v_flat,
    uint32_t HV,
    bool qk_norm,
    float scale,
    bool qk_flat,
    uint32_t Hk) {
    const auto& q_shape = q.logical_shape();  // [BH,NC,C,K] head-major, or flat [B,T,Hk*K] when qk_flat
    const auto& v_shape = v.logical_shape();  // [BH,NC,C,V] head-major, or flat [B,T,HV*V] when v_flat
    // Derive dims. Head-major q gives BH/NC/K directly; flat q [B,T,Hk*K] gives B/T, so BH=B*HV,
    // NC=T/chunk (pad==0 required), K=flat_width/Hk. val_dim = v_shape[3] or v_flat width / HV.
    const uint32_t BH = qk_flat ? (q_shape[0] * HV) : q_shape[0];
    const uint32_t num_chunks = qk_flat ? (q_shape[1] / chunk_size) : q_shape[1];
    const uint32_t key_dim = qk_flat ? (q_shape[2] / Hk) : q_shape[3];
    const uint32_t val_dim = v_flat ? (v_shape[2] / HV) : v_shape[3];
    auto attrs = ChunkGdnPrepOperation::operation_attributes_t{
        .BH = BH,
        .num_chunks = num_chunks,
        .chunk_size = chunk_size,
        .key_dim = key_dim,
        .val_dim = val_dim,
        .v_flat = v_flat,
        .HV = HV,
        .qk_flat = qk_flat,
        .Hk = Hk,
        .qk_norm = qk_norm,
        .scale = scale,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = ChunkGdnPrepOperation::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .eye_c = eye_c,
        .tril_c = tril_c,
        .ones_c = ones_c,
        .masks_c = masks_c};
    return ttnn::device_operation::launch<ChunkGdnPrepOperation>(attrs, tensor_args);
}

// ---------------------------------------------------------------------------
// SCAN
// ---------------------------------------------------------------------------
ChunkGdnScanOperation::program_factory_t ChunkGdnScanOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChunkGdnScanProgramFactory{};
}

void ChunkGdnScanOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    check(in.v_beta, "v_beta", DataType::FLOAT32);
    check(in.kd, "kd", DataType::FLOAT32);
    check(in.q_decay, "q_decay", DataType::FLOAT32);
    check(in.intra, "intra", DataType::FLOAT32);
    check(in.k_dec_t, "k_dec_t", DataType::FLOAT32);
    check(in.dl, "dl", DataType::FLOAT32);
    check(in.t_inv, "t_inv", DataType::FLOAT32);
    if (in.initial_state.has_value()) {
        check(*in.initial_state, "initial_state", DataType::FLOAT32);
    }
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
}

ChunkGdnScanOperation::spec_return_value_t ChunkGdnScanOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    // o is fp32; the recurrent final state is fp32 too. (A bf16 o output — feeding a bf16 attention
    // result into every GDN layer — measurably degraded full-model quality, so it was removed; the
    // seq path also keeps o fp32.)
    const auto o_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto s_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    ttnn::Shape o_shape({attrs.BH, attrs.num_chunks, attrs.chunk_size, attrs.val_dim});
    ttnn::Shape s_shape({attrs.BH, attrs.key_dim, attrs.val_dim});
    return {tt::tt_metal::TensorSpec(o_shape, o_layout), tt::tt_metal::TensorSpec(s_shape, s_layout)};
}

ChunkGdnScanOperation::tensor_return_value_t ChunkGdnScanOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.v_beta.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, device));
    }
    return outs;
}

std::vector<Tensor> chunk_gdn_scan(
    const Tensor& v_beta,
    const Tensor& kd,
    const Tensor& q_decay,
    const Tensor& intra,
    const Tensor& k_dec_t,
    const Tensor& dl,
    const Tensor& t_inv,
    const std::optional<Tensor>& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& vb_shape = v_beta.logical_shape();  // [BH, NC, C, V]
    const auto& kd_shape = kd.logical_shape();      // [BH, NC, C, K]
    auto attrs = ChunkGdnScanOperation::operation_attributes_t{
        .BH = vb_shape[0],
        .num_chunks = vb_shape[1],
        .chunk_size = chunk_size,
        .key_dim = kd_shape[3],
        .val_dim = vb_shape[3],
        .has_initial_state = initial_state.has_value(),
        .output_final_state = output_final_state,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = ChunkGdnScanOperation::tensor_args_t{
        .v_beta = v_beta,
        .kd = kd,
        .q_decay = q_decay,
        .intra = intra,
        .k_dec_t = k_dec_t,
        .dl = dl,
        .t_inv = t_inv,
        .initial_state = initial_state};
    return ttnn::device_operation::launch<ChunkGdnScanOperation>(attrs, tensor_args);
}

}  // namespace ttnn::prim
