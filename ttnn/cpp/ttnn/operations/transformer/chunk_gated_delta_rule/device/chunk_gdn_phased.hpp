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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

namespace detail {

// Total tile capacity at each fixed PREP CB index.  Indices 0, 1, 2 and 16 are bf16; the rest are
// fp32.  Keep indices 0..30 byte-identical to the monolithic layout: the Horner matmul path is
// layout-sensitive.  Index 31 (s3) is last and is only a one-tile invert_block temporary, so
// right-sizing it cannot move any other CB.
constexpr std::array<uint32_t, 32> chunk_gdn_prep_cb_tile_capacities(
    uint32_t Ct, uint32_t Kt, uint32_t Vt) {
    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t kv = Kt * Vt;
    const uint32_t kc = Kt * Ct;
    const uint32_t scratch = std::max({cc, ck, cv, kv, kc});
    return {
        ck,       // q
        ck,       // k
        cv,       // v
        Ct,       // g
        Ct,       // beta
        cc,       // eye
        cc,       // tril
        cc,       // ones
        2 * kv,   // S
        Ct,       // decay
        Ct,       // decay_exp
        Ct,       // decayfac
        cc,       // lmask
        cc,       // Tinv
        cv,       // vbeta
        ck,       // kbeta
        2 * cv,   // out
        cv,       // u / quadrant masks
        ck,       // w / kd
        ck,       // qdecay
        cc,       // intra
        2 * kv,   // s2
        cv,       // vnew / dl
        cv,       // ointer
        kc,       // kdec_t
        kv,       // supd
        kv,       // stmp
        kv,       // final_s
        scratch,  // scr1
        scratch,  // scr2
        scratch,  // scr3
        1,        // s3: one-tile invert_block temporary; last CB preserves all preceding addresses
    };
}

constexpr uint32_t chunk_gdn_prep_cb_bytes(uint32_t Ct, uint32_t Kt, uint32_t Vt) {
    constexpr uint32_t bf16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    constexpr uint32_t fp32_tile_bytes = tt::tile_size(tt::DataFormat::Float32);
    const auto capacities = chunk_gdn_prep_cb_tile_capacities(Ct, Kt, Vt);
    uint32_t bytes = 0;
    for (std::size_t i = 0; i < capacities.size(); ++i) {
        bytes += capacities[i] * ((i == 0 || i == 1 || i == 2 || i == 16) ? bf16_tile_bytes : fp32_tile_bytes);
    }
    return bytes;
}

}  // namespace detail

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
    // OPT-B: qk_norm => the prep compute L2-normalizes q/k over K in-kernel (host skipped it) and
    // folds `scale` into q's norm. Only valid for chunk_size==32 (Ct==1). scale defaults to no-op.
    bool qk_norm = false;
    float scale = 1.0f;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ChunkGdnPrepInputs {
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
    uint32_t Hk = 0);

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
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ChunkGdnScanInputs {
    Tensor v_beta;                        // [BH, NC, C, V] fp32  (= v * beta)
    Tensor kd;                            // [BH, NC, C, K] fp32  (= k_beta * decay_exp)
    Tensor q_decay;                       // [BH, NC, C, K] fp32
    Tensor intra;                         // [BH, NC, C, C] fp32
    Tensor k_dec_t;                       // [BH, NC, K, C] fp32
    Tensor dl;                            // [BH, NC, 1, 1] fp32 (scalar per chunk in tile [0,0])
    Tensor t_inv;                         // [BH, NC, C, C] fp32  (WY inverse)
    std::optional<Tensor> initial_state;  // [BH, K, V] fp32 or absent (zeros)
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
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
