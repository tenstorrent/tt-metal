// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gdn_fused.hpp"

#include <algorithm>
#include <cstdlib>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
// Uniquely named (vs the phased prim's `check`) so it does not clash under unity builds.
void check_fused(const Tensor& t, const char* name, DataType dt) {
    TT_FATAL(t.layout() == Layout::TILE, "chunk_gdn_fused: {} must be TILE layout", name);
    TT_FATAL(t.dtype() == dt, "chunk_gdn_fused: {} has wrong dtype", name);
    TT_FATAL(t.buffer() != nullptr, "chunk_gdn_fused: {} must be on device", name);
}
}  // namespace

ChunkGdnFusedOperation::program_factory_t ChunkGdnFusedOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChunkGdnFusedProgramFactory{};
}

void ChunkGdnFusedOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    // Input-side checks: identical to the phased PREP prim (the fused producer runs the unchanged
    // prep reader/compute, so its input contract is prep's).
    check_fused(in.q, "q", DataType::BFLOAT16);
    check_fused(in.k, "k", DataType::BFLOAT16);
    check_fused(in.v, "v", DataType::BFLOAT16);  // flat [B,T,HV*V] when attrs.v_flat; else [BH,NC,C,V]
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
    check_fused(in.g, "g", DataType::FLOAT32);
    check_fused(in.beta, "beta", DataType::FLOAT32);
    check_fused(in.eye_c, "eye_c", DataType::FLOAT32);
    check_fused(in.tril_c, "tril_c", DataType::FLOAT32);
    check_fused(in.ones_c, "ones_c", DataType::FLOAT32);
    check_fused(in.masks_c, "masks_c", DataType::FLOAT32);
    if (in.initial_state.has_value()) {
        check_fused(*in.initial_state, "initial_state", DataType::FLOAT32);
    }
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
    // NP producers + one receiver core per head (NV=1). NP=1 unless QWEN_GDN_NP opted in.
    TT_FATAL(attrs.np >= 1, "chunk_gdn_fused: np must be >= 1 (got {})", attrs.np);
    const auto grid = in.q.device()->compute_with_storage_grid_size();
    TT_FATAL(
        attrs.BH * (1 + attrs.np) <= grid.x * grid.y,
        "chunk_gdn_fused needs BH*(1+NP) = {}*{} = {} cores, grid has {}x{}={}",
        attrs.BH,
        1 + attrs.np,
        attrs.BH * (1 + attrs.np),
        grid.x,
        grid.y,
        grid.x * grid.y);
}

ChunkGdnFusedOperation::spec_return_value_t ChunkGdnFusedOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    // EXACTLY ChunkGdnScanOperation::compute_output_specs: o and final_state are both fp32 (a bf16
    // o degraded full-model quality and was removed — see the phased scan op).
    const auto o_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto s_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    ttnn::Shape o_shape({attrs.BH, attrs.num_chunks, attrs.chunk_size, attrs.val_dim});
    ttnn::Shape s_shape({attrs.BH, attrs.key_dim, attrs.val_dim});
    return {tt::tt_metal::TensorSpec(o_shape, o_layout), tt::tt_metal::TensorSpec(s_shape, s_layout)};
}

ChunkGdnFusedOperation::tensor_return_value_t ChunkGdnFusedOperation::create_output_tensors(
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

std::vector<Tensor> chunk_gdn_fused(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye_c,
    const Tensor& tril_c,
    const Tensor& ones_c,
    const Tensor& masks_c,
    const std::optional<Tensor>& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
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
    // Dim derivation identical to chunk_gdn_prep (the fused op consumes prep's inputs).
    const uint32_t BH = qk_flat ? (q_shape[0] * HV) : q_shape[0];
    const uint32_t num_chunks = qk_flat ? (q_shape[1] / chunk_size) : q_shape[1];
    const uint32_t key_dim = qk_flat ? (q_shape[2] / Hk) : q_shape[3];
    const uint32_t val_dim = v_flat ? (v_shape[2] / HV) : v_shape[3];
    // F3a producers per head: read HERE (attrs construction), never in the factory — np is hashed,
    // so an env toggle compiles a fresh program instead of silently serving a stale cached one.
    // Clamped to num_chunks: a producer beyond NC would own no chunks (wasted core, and the
    // receiver's rotating credit c % NP would skip it anyway).
    uint32_t np = 1;
    if (const char* e = std::getenv("QWEN_GDN_NP")) {
        const int v_np = std::atoi(e);
        TT_FATAL(v_np >= 1, "QWEN_GDN_NP must be a positive integer (got '{}')", e);
        np = std::min<uint32_t>(static_cast<uint32_t>(v_np), num_chunks);
    }
    auto attrs = ChunkGdnFusedOperation::operation_attributes_t{
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
        .np = np,
        .has_initial_state = initial_state.has_value(),
        .output_final_state = output_final_state,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = ChunkGdnFusedOperation::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .eye_c = eye_c,
        .tril_c = tril_c,
        .ones_c = ones_c,
        .masks_c = masks_c,
        .initial_state = initial_state};
    return ttnn::device_operation::launch<ChunkGdnFusedOperation>(attrs, tensor_args);
}

}  // namespace ttnn::prim
