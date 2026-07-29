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
struct ChunkGdnPrepParams {
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
    // KDA vector gate may stay flat token-major [B,T,HV*K]; the prep reader gathers one head/chunk.
    bool g_flat = false;
    // OPT-B: qk_norm => the prep compute L2-normalizes q/k over K in-kernel (host skipped it) and
    // folds `scale` into q's norm. Only valid for chunk_size==32 (Ct==1). scale defaults to no-op.
    bool qk_norm = false;
    float scale = 1.0f;
    bool vector_gate = false;
    // Private KDA experiment: bit i stores prep output i as BF16 in DRAM.
    uint32_t output_bf16_mask = 0;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

uint32_t chunk_gdn_prep_cb_size_bytes(
    uint32_t chunk_size,
    uint32_t key_dim,
    uint32_t val_dim,
    bool vector_gate,
    DataType gate_dtype,
    uint32_t output_bf16_mask);

struct ChunkGdnPrepInputs {
    Tensor q;        // [BH, NC, C, K] bf16
    Tensor k;        // [BH, NC, C, K] bf16
    Tensor v;        // [BH, NC, C, V] bf16  (or FLAT [B, T, HV*V] bf16 when params.v_flat)
    Tensor g;        // [BH, NC, C, 1|K] fp32 (or KDA FLAT [B,T,HV*K] when params.g_flat)
    Tensor beta;     // [BH, NC, C, 1] fp32 (column)
    Tensor eye_c;    // [1,1,C,C] fp32
    Tensor tril_c;   // [1,1,C,C] fp32
    Tensor ones_c;   // [1,1,C,C] fp32
    Tensor masks_c;  // [1,1,32,96] fp32 — three 32x32 WY-inverse quadrant masks (Qtl|Qbr|Q10)
};

struct ChunkGdnPrepProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ChunkGdnPrepParams&, const ChunkGdnPrepInputs&, std::vector<Tensor>&);
};

struct ChunkGdnPrepOperation {
    using operation_attributes_t = ChunkGdnPrepParams;
    using tensor_args_t = ChunkGdnPrepInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ChunkGdnPrepProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Returns {v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv} (FP32 by default; selected KDA intermediates may use BF16
// storage). (WY hand-off is un-premultiplied: the scan applies t_inv AFTER the v_beta - kd@S subtraction,
//  so the inverse's fp error is not amplified by the cancellation.)
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
    bool v_flat = false,
    uint32_t HV = 0,
    bool qk_norm = false,
    float scale = 1.0f,
    bool qk_flat = false,
    uint32_t Hk = 0,
    bool g_flat = false,
    bool vector_gate = false,
    uint32_t output_bf16_mask = 0);

// ---------------------------------------------------------------------------
// SCAN
// ---------------------------------------------------------------------------
struct ChunkGdnScanParams {
    uint32_t BH;
    uint32_t num_chunks;
    uint32_t chunk_size;
    uint32_t key_dim;
    uint32_t val_dim;
    bool has_initial_state;
    bool identity_initial_state = false;
    bool output_final_state;
    bool state_only = false;
    bool summary_pair = false;
    bool vector_gate = false;
    bool fused_rms = false;
    uint32_t num_heads = 0;
    float rms_epsilon = 1e-5f;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ChunkGdnScanInputs {
    Tensor v_beta;                        // [BH, NC, C, V] fp32 or bf16  (= v * beta)
    Tensor kd;                            // [BH, NC, C, K] fp32 or bf16  (= k_beta * decay_exp)
    Tensor q_decay;                       // [BH, NC, C, K] fp32 or bf16
    Tensor intra;                         // [BH, NC, C, C] fp32
    Tensor k_dec_t;                       // [BH, NC, K, C] fp32 or bf16
    Tensor dl;                            // [BH, NC, 1|K, 1] fp32 or bf16 (scalar GDN or vector KDA decay)
    Tensor t_inv;                         // [BH, NC, C, C] fp32  (WY inverse)
    std::optional<Tensor> initial_state;  // [BH, K, V] fp32 or absent (zeros)
    std::optional<Tensor> identity_tile;  // [1,1,32,32] fp32; block identity when present
    std::optional<Tensor> rms_gate;       // [B,T,H*V] bf16 when fused_rms
    std::optional<Tensor> rms_weight;     // [V] bf16 when fused_rms
};

struct ChunkGdnScanProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ChunkGdnScanParams&, const ChunkGdnScanInputs&, std::vector<Tensor>&);
};

struct ChunkGdnScanOperation {
    using operation_attributes_t = ChunkGdnScanParams;
    using tensor_args_t = ChunkGdnScanInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ChunkGdnScanProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Returns {o [BH,NC,C,V] bf16, final_state [BH,K,V] fp32}.
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
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool vector_gate = false,
    bool state_only = false,
    const std::optional<Tensor>& identity_tile = std::nullopt,
    const std::optional<Tensor>& rms_gate = std::nullopt,
    const std::optional<Tensor>& rms_weight = std::nullopt,
    uint32_t num_heads = 0,
    float rms_epsilon = 1e-5f,
    bool summary_pair = false);

// ---------------------------------------------------------------------------
// KDA GROUPED AFFINE PREFIX
// ---------------------------------------------------------------------------
struct KdaAffinePrefixParams {
    uint32_t BH;
    uint32_t groups_per_head;
    uint32_t key_dim;
    uint32_t val_dim;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    bool compose_only;
};

struct KdaAffinePrefixInputs {
    Tensor transform_a;
    Tensor transform_b;
    std::optional<Tensor> initial_state;
};

struct KdaAffinePrefixProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KdaAffinePrefixParams&, const KdaAffinePrefixInputs&, std::vector<Tensor>&);
};

struct KdaAffinePrefixOperation {
    using operation_attributes_t = KdaAffinePrefixParams;
    using tensor_args_t = KdaAffinePrefixInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<KdaAffinePrefixProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor kda_affine_prefix(
    const Tensor& transform_a,
    const Tensor& transform_b,
    const Tensor& initial_state,
    uint32_t groups_per_head,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

std::pair<Tensor, Tensor> kda_affine_compose(
    const Tensor& transform_a,
    const Tensor& transform_b,
    uint32_t groups_per_head,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

struct KdaGatedRmsParams {
    uint32_t batch;
    uint32_t num_heads;
    uint32_t sequence;
    uint32_t value_dim;
    float epsilon;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct KdaGatedRmsInputs {
    Tensor input;
    Tensor gate;
    Tensor weight;
};

struct KdaGatedRmsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KdaGatedRmsParams&, const KdaGatedRmsInputs&, std::vector<Tensor>&);
};

struct KdaGatedRmsOperation {
    using operation_attributes_t = KdaGatedRmsParams;
    using tensor_args_t = KdaGatedRmsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<KdaGatedRmsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor kda_gated_rms_norm(
    const Tensor& input,
    const Tensor& gate,
    const Tensor& weight,
    uint32_t num_heads,
    float epsilon,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

struct KdaCausalConvParams {
    uint32_t sequence;
    uint32_t q_width;
    uint32_t k_width;
    uint32_t v_width;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct KdaCausalConvInputs {
    Tensor input;
    Tensor state;
    Tensor tap0;
    Tensor tap1;
    Tensor tap2;
    Tensor tap3;
};

struct KdaCausalConvProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KdaCausalConvParams&, const KdaCausalConvInputs&, std::vector<Tensor>&);
};

struct KdaCausalConvOperation {
    using operation_attributes_t = KdaCausalConvParams;
    using tensor_args_t = KdaCausalConvInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<KdaCausalConvProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> kda_causal_conv1d_split(
    const Tensor& input,
    const Tensor& state,
    const Tensor& tap0,
    const Tensor& tap1,
    const Tensor& tap2,
    const Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
