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
#include "ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule_config.hpp"

namespace ttnn::prim {

// WY-inverse method of the prep compute (the `tinv` attr of the prep and fused prims): the op's
// ChunkGdnWyInverse with AUTO resolved for this device and chunk size.
//   HORNER    : invert_block — quadrant split, two 15-term Horner inverses and an exact off-diagonal on
//               the matrix engine (~60 LLK calls per chunk). Runs on every architecture at every chunk
//               size; the reference the SFPU method is validated against.
//   SFPU_FP32 : one SFPU forward-substitution solve reading negN as fp32 in place
//               (kernels/compute/chunk_gdn_tinv_sfpu.hpp). Blackhole-only, chunk_size == 32. PCC-class
//               against HORNER — T_inv error no larger in any measured regime — for about a quarter less
//               producer time per chunk.
enum class GdnTinv : uint32_t { HORNER = 0, SFPU_FP32 = 1 };
// The method for one call, resolved at attrs construction (hashed): HORNER / SFPU as requested (an
// explicit SFPU the device or chunk size cannot honor FATALs in validate_gdn_tinv rather than falling
// back); AUTO is SFPU_FP32 wherever that is supported and HORNER elsewhere.
uint32_t gdn_tinv_resolve(
    ttnn::transformer::ChunkGdnWyInverse wy_inverse, uint32_t chunk_size, const Tensor& any_input);
// FATAL unless the method is supported for this chunk size on this device.
void validate_gdn_tinv(uint32_t tinv, uint32_t chunk_size, const Tensor& any_input);
// A pre-allocated final-state output (scan / fused prims): [BH, K, V] fp32 TILE interleaved, allocated on
// the device of `any_input`. It may share its buffer with the initial state (in-place state update).
void validate_gdn_final_state_out(
    const Tensor& final_state_out, uint32_t BH, uint32_t key_dim, uint32_t val_dim, const Tensor& any_input);

// ---------------------------------------------------------------------------
// PREP
// ---------------------------------------------------------------------------
struct ChunkGdnPrepParams {
    uint32_t BH;
    uint32_t num_chunks;
    uint32_t chunk_size;
    uint32_t key_dim;
    uint32_t val_dim;
    // Flat v: when v_flat, `v` is the FLAT token-major tensor [B, T, HV*V] and the
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
    // ChunkGdnPhasedProgramConfig::prep_serial: BH cores (one per head) instead of fanning the BH*NC
    // work-items over the whole grid. Measurement only. Hashed, like every field here.
    bool prep_serial = false;
    // gb_flat (Option B, enabled by passing `sel`): `g`/`beta` are the RAW [B,T,HV] fp32 tensors (one
    // tile wide, HV<=32) instead of [BH,NC,C,1]; the prep reader/compute select head h's column with a
    // one-hot matmul against `sel`. Consumed by the FUSED producer only (chunk_gdn_fused.hpp);
    // phased prep rejects it (validate_on_program_cache_miss) — its cores span several heads, and
    // the zero-CB-growth selector is loaded once per core. The field stays so both factories share
    // one reader/compute CT-arg layout.
    bool gb_flat = false;
    // WY-inverse method (GdnTinv): Horner quadrants on the matrix engine or the SFPU forward-substitution
    // solve — the op's wy_inverse kwarg resolved by gdn_tinv_resolve at attrs construction (hashed); the
    // fused prim carries the same field, so fused == phased stays bit-exact for any given method.
    uint32_t tinv = 0;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ChunkGdnPrepInputs {
    Tensor q;        // [BH, NC, C, K] bf16
    Tensor k;        // [BH, NC, C, K] bf16
    Tensor v;        // [BH, NC, C, V] bf16  (or FLAT [B, T, HV*V] bf16 when params.v_flat)
    Tensor g;        // [BH, NC, C, 1] fp32 (column) (or FLAT [B, T, HV] fp32 when params.gb_flat)
    Tensor beta;     // [BH, NC, C, 1] fp32 (column) (or FLAT [B, T, HV] fp32 when params.gb_flat)
    Tensor eye_c;    // [1,1,C,C] fp32
    Tensor tril_c;   // [1,1,C,C] fp32
    Tensor ones_c;   // [1,1,C,C] fp32
    Tensor masks_c;  // [1,1,32,96] fp32 — three 32x32 WY-inverse quadrant masks (Qtl|Qbr|Q10)
    // gb_flat one-hot head selector [1,1,32,32*HV] fp32 TILE (tile h = one-hot, row h col 0 = 1).
    // Absent unless params.gb_flat (mirrors initial_state's optional-tensor pattern below).
    std::optional<Tensor> sel;
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

// Returns {v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv} (all fp32, per-chunk DRAM tensors).
// (WY hand-off is un-premultiplied: the scan applies t_inv AFTER the v_beta - kd@S subtraction,
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
    bool prep_serial = false,
    bool gb_flat = false,
    const std::optional<Tensor>& sel = std::nullopt,
    ttnn::transformer::ChunkGdnWyInverse wy_inverse = ttnn::transformer::ChunkGdnWyInverse::AUTO);

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
    bool output_final_state;
    // ChunkGdnPhasedProgramConfig::use_mcast / scan_serial (see chunk_gated_delta_rule_config.hpp).
    bool use_mcast = true;
    bool force_serial = false;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ChunkGdnScanInputs {
    Tensor v_beta;                        // [BH, NC, C, V] fp32  (= v * beta)
    Tensor kd;                            // [BH, NC, C, K] fp32  (= k_beta * decay_exp)
    Tensor q_decay;                       // [BH, NC, C, K] fp32
    Tensor intra;                         // [BH, NC, C, C] fp32
    Tensor k_dec_t;                       // [BH, NC, K, C] fp32
    Tensor dl;                            // [BH, NC, 32, 32] fp32: dl*I, dl = exp(g_sum) of the chunk on the diagonal
    Tensor t_inv;                         // [BH, NC, C, C] fp32  (WY inverse)
    std::optional<Tensor> initial_state;  // [BH, K, V] fp32 or absent (zeros)
    // Pre-allocated final-state output [BH, K, V] fp32 (the op's final_state_output), or absent (a new
    // tensor is allocated). May share its buffer with initial_state: each scan core reads its s0 V-slice
    // once, before it writes the same slice of the final state.
    std::optional<Tensor> final_state_out;
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

// Returns {o [BH,NC,C,V] fp32, final_state [BH,K,V] fp32}. o is fp32 — see the scan factory's
// df_io and ChunkGdnScanOperation::compute_output_specs (a bf16 o degraded full-model quality
// and was removed).
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
    bool use_mcast = true,
    bool force_serial = false,
    const std::optional<Tensor>& final_state_out = std::nullopt);

}  // namespace ttnn::prim
