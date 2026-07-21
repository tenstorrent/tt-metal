// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase-split chunked Gated Delta Rule (2 prim ops with a DRAM hand-off):
//   PREP (state-independent, parallel over head x chunk): produces per-chunk
//        u, w, q_decay, intra, k_dec_t, dl.
//   SCAN (sequential over chunk, parallel over head): consumes those + the
//        initial state, carries S [K,V], produces o and final_state.
// Splitting the monolithic kernel at the recurrence boundary lets the expensive
// state-independent work (incl. the WY inverse) fan out across cores, exactly as
// FLA's fwd_intra / fwd_h / fwd_o split does across GPU SMs.

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

// ---------------------------------------------------------------------------
// PREP
// ---------------------------------------------------------------------------
struct ChunkKdaPrepParams {
    uint32_t BH;
    uint32_t num_chunks;
    uint32_t chunk_size;
    uint32_t key_dim;
    uint32_t val_dim;
    // OPT-A (QWEN_GDN_FLAT_QKV): when v_flat, `v` is the FLAT token-major tensor [B, T, HV*V] and the
    // prep reader tile-addresses head hv's chunk c directly out of it (no head-split/permute/pad
    // materialization on the host). HV is the value-head count (needed for the flat row stride).
    // Only the v INPUT read changes; the prep still WRITES head-major v_beta, so the scan and every
    // downstream op are byte-identical. Requires the time dim to be a multiple of chunk_size (pad==0).
    bool v_flat = false;
    uint32_t HV = 0;
    // OPT-A q/k: when qk_flat, q and k are FLAT token-major [B,T,H*K]; the reader tile-addresses key
    // head hk=hv/G (GQA) out of the flat grid. Hk = key-head count (flat q/k row stride = Hk*Kt).
    bool qk_flat = false;
    uint32_t Hk = 0;
    // OPT-B: qk_norm => the prep compute L2-normalizes q/k over K in-kernel (host skipped it) and
    // folds `scale` into q's norm. Only valid for chunk_size==32 (Ct==1). scale defaults to no-op.
    bool qk_norm = false;
    float scale = 1.0f;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ChunkKdaPrepInputs {
    Tensor q;        // [BH, NC, C, K] bf16
    Tensor k;        // [BH, NC, C, K] bf16
    Tensor v;        // [BH, NC, C, V] bf16  (or FLAT [B, T, HV*V] bf16 when params.v_flat)
    Tensor g;        // [BH, NC, C, 1] fp32 (column)
    Tensor beta;     // [BH, NC, C, 1] fp32 (column)
    Tensor eye_c;    // [1,1,C,C] fp32
    Tensor tril_c;   // [1,1,C,C] fp32
    Tensor ones_c;   // [1,1,C,C] fp32
    Tensor masks_c;  // [1,1,32,96] fp32 — three 32x32 WY-inverse quadrant masks (Qtl|Qbr|Q10)
};

struct ChunkKdaPrepProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ChunkKdaPrepParams&, const ChunkKdaPrepInputs&, std::vector<Tensor>&);
};

struct ChunkKdaPrepOperation {
    using operation_attributes_t = ChunkKdaPrepParams;
    using tensor_args_t = ChunkKdaPrepInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ChunkKdaPrepProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Returns {v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv} (all fp32, per-chunk DRAM tensors).
// (WY hand-off is un-premultiplied: the scan applies t_inv AFTER the v_beta - kd@S subtraction,
//  so the inverse's fp error is not amplified by the cancellation.)
std::vector<Tensor> chunk_kda_prep(
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
    bool v_flat = false,
    uint32_t HV = 0,
    bool qk_norm = false,
    float scale = 1.0f,
    bool qk_flat = false,
    uint32_t Hk = 0);

// ---------------------------------------------------------------------------
// SCAN
// ---------------------------------------------------------------------------
struct ChunkKdaScanParams {
    uint32_t BH;
    uint32_t num_chunks;
    uint32_t chunk_size;
    uint32_t key_dim;
    uint32_t val_dim;
    bool has_initial_state;
    bool output_final_state;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ChunkKdaScanInputs {
    Tensor v_beta;                        // [BH, NC, C, V] fp32  (= v * beta)
    Tensor kd;                            // [BH, NC, C, K] fp32  (= k_beta * decay_exp)
    Tensor q_decay;                       // [BH, NC, C, K] fp32
    Tensor intra;                         // [BH, NC, C, C] fp32
    Tensor k_dec_t;                       // [BH, NC, K, C] fp32
    Tensor dl;                            // [BH, NC, 1, 1] fp32 (scalar per chunk in tile [0,0])
    Tensor t_inv;                         // [BH, NC, C, C] fp32  (WY inverse)
    std::optional<Tensor> initial_state;  // [BH, K, V] fp32 or absent (zeros)
};

struct ChunkKdaScanProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ChunkKdaScanParams&, const ChunkKdaScanInputs&, std::vector<Tensor>&);
};

struct ChunkKdaScanOperation {
    using operation_attributes_t = ChunkKdaScanParams;
    using tensor_args_t = ChunkKdaScanInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ChunkKdaScanProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Returns {o [BH,NC,C,V] bf16, final_state [BH,K,V] fp32}.
std::vector<Tensor> chunk_kda_scan(
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
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
