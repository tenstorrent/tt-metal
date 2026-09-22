// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused prep→scan chunked Gated Delta Rule (ONE prim, ONE program, zero DRAM intermediates):
// per head, a dedicated PRODUCER core runs the unchanged prep reader+compute and a writer that
// NoC-writes the 7 computed intermediates (v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv)
// straight into its paired RECEIVER core's CBs via the shipped ready/valid handshake; the
// receiver runs the unchanged scan compute+writer. NP >= 1 producers (QWEN_GDN_NP, default 1)
// and NV=1 (full V) per head.
// Takes prep's inputs, returns scan's outputs — the seven fp32 DRAM tensors of the phased
// hand-off simply never exist. The phased prims (chunk_gdn_phased.hpp) stay in-tree as the
// bit-exact reference: the DRAM round trip they perform is a byte copy, so fused == phased
// bit-for-bit as long as the shared math bodies and CB pack boundaries are untouched.

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

// Union of the prep and scan params (see chunk_gdn_phased.hpp for per-field semantics of the
// prep-side v_flat/HV/qk_flat/Hk/qk_norm/scale block and the scan-side state flags).
struct ChunkGdnFusedParams {
    uint32_t BH;
    uint32_t num_chunks;
    uint32_t chunk_size;
    uint32_t key_dim;
    uint32_t val_dim;
    bool v_flat = false;
    uint32_t HV = 0;
    bool qk_flat = false;
    uint32_t Hk = 0;
    bool qk_norm = false;
    float scale = 1.0f;
    // F3a: producers per head (NP). Producer p of a head owns chunks c = p, p+NP, ... and NoC-
    // writes them into the head's single receiver in order (receiver-driven rotating ready
    // credits). Read from QWEN_GDN_NP at attrs construction (never in the factory — this field
    // being hashed is what keeps the program cache honest) and clamped to num_chunks.
    uint32_t np = 1;
    // Receivers per head (NV): a head's NV receiver cores form a 1xNV row rectangle and each carries a
    // V-slice of Vt/NV tiles (the phased scan's V-block split, fed over the NoC). Read from QWEN_GDN_NV
    // at attrs construction (hashed); default 1 until the cost model (Phase 2) chooses it.
    uint32_t nv = 1;
    // Hand-off CB depth (slots per CB): how many chunks a producer may run ahead of a receiver's
    // consumption, and how early a receiver can reserve+credit the next chunk. 2 = F2's value; deeper
    // rings hide more of the per-chunk handshake round trip at +76 KB of L1 per slot on every core.
    // Read from QWEN_GDN_HANDOFF_NBUF at attrs construction (hashed). Phase 1b: the receiver keeps
    // nbuf-1 hand-offs in flight (per-slot VALID flags, BH x nbuf credit words), so 3 hides the unicast
    // round trip (5.8-6.1 us) behind two receiver steps at both NV=2 and NV=4.
    uint32_t nbuf = 2;  // measured (v0.3 §10b): 3 and 4 are slower than 2 in every transport
    // Phase 1 A/B (design D5/D16): ship the six shared tensors and the v_beta slices as NV plain unicast
    // writes per item instead of a linked multicast chain. Multicasts reserve router ports along their
    // path; the zone captures show sporadic 20-120 us multicast-issue stalls on individual producers.
    // Read from QWEN_GDN_UNICAST at attrs construction (hashed). Default since Phase 1b (QWEN_GDN_UNICAST=0
    // restores the multicast chain for A/B).
    bool unicast = true;
    // Phase 1b A/B (design D5): with the unicast transport, ship the data as POSTED writes (no acks,
    // no per-item write barrier) and order the VALID flag behind them by the NoC's in-order delivery
    // on one (source, destination, VC, command buffer) — the argument tt-metal's matmul multicast
    // sender uses. Requires unicast. Read from QWEN_GDN_POSTED at attrs construction (hashed).
    bool posted = false;
    // Placement (design D9): 0 = receivers row-major from row 0, producers fill the rest (v0.2);
    // 1 = ROW-LOCAL: one head per row (receivers in columns 0..NV-1, its NP producers to their east in
    // the same row), heads beyond grid.y in the leftover columns as vertical blocks. NOC_1 routes -x
    // then -y, so a head's hand-off traffic never leaves its own row (or column block) and heads do not
    // share NoC links (v0.3 §10b). Read from QWEN_GDN_PLACEMENT at attrs construction (hashed).
    uint32_t placement = 0;
    bool has_initial_state = false;
    bool output_final_state = false;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ChunkGdnFusedInputs {
    Tensor q;                             // [BH, NC, C, K] bf16 (or FLAT [B, T, Hk*K] bf16 when params.qk_flat)
    Tensor k;                             // [BH, NC, C, K] bf16 (or FLAT, as q)
    Tensor v;                             // [BH, NC, C, V] bf16 (or FLAT [B, T, HV*V] bf16 when params.v_flat)
    Tensor g;                             // [BH, NC, C, 1] fp32 (column)
    Tensor beta;                          // [BH, NC, C, 1] fp32 (column)
    Tensor eye_c;                         // [1,1,C,C] fp32
    Tensor tril_c;                        // [1,1,C,C] fp32
    Tensor ones_c;                        // [1,1,C,C] fp32
    Tensor masks_c;                       // [1,1,32,96] fp32 — three 32x32 WY-inverse quadrant masks (Qtl|Qbr|Q10)
    std::optional<Tensor> initial_state;  // [BH, K, V] fp32 or absent (zeros)
};

struct ChunkGdnFusedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ChunkGdnFusedParams&, const ChunkGdnFusedInputs&, std::vector<Tensor>&);
};

struct ChunkGdnFusedOperation {
    using operation_attributes_t = ChunkGdnFusedParams;
    using tensor_args_t = ChunkGdnFusedInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ChunkGdnFusedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The per-head row-local cost model, calibrated on QB2:
//   T_fused(NV, NP) = NC * max(w_p / NP, t_step(Vt / NV)) + fill   over (NV | Vt, NP) with a feasible
//   row-local layout, ties -> fewer cores, then smaller NV; T_phased(BH) from the measured table.
// The op host uses it for whichever of num_receivers / num_producers / row_local the fused program
// config leaves free; test_chunk_gdn_fused_geometry.py checks it against a Python oracle on several grids.
struct FusedGeometryChoice {
    uint32_t nv = 0;  // 0 => no fused geometry fits this grid
    uint32_t np = 0;
    uint32_t placement = 0;  // 1 row-local, 0 row-major fallback
    float t_fused_us = 0.0f;
    float t_phased_us = 0.0f;
    bool fused_pays = false;
};
bool fused_row_local_feasible(uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t NP);
// fixed_nv / fixed_np = 0 -> free; a non-zero value pins that field (an env override) and the model
// chooses the other one so the pair still fits (and prefers a row-local layout for it).
FusedGeometryChoice choose_fused_geometry(
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t BH,
    uint32_t NC,
    uint32_t Vt,
    uint32_t fixed_nv = 0,
    uint32_t fixed_np = 0);

// Design D9: the fused program's core map, a pure function of its arguments (no device), shared by
// the program factory and the nanobind geometry oracle. placement 0 = row-major 1xNV receiver
// rectangles with the producers on the remaining cores row-major; 1 = row-local (a head's receivers
// and producers in one row segment, leftover heads as column blocks). FATALs when the layout does
// not fit, exactly as the factory would.
struct FusedPlacement {
    std::vector<CoreCoord> receivers;  // index h*NV + v (logical coordinates)
    std::vector<CoreCoord> producers;  // index h*NP + j
};
FusedPlacement fused_placement(
    uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t NP, uint32_t placement);

// Returns {o [BH,NC,C,V] fp32, final_state [BH,K,V] fp32} — exactly the scan prim's output specs.
// Needs BH*(NV+NP) cores (NP producers + NV receivers per head; both default to 1 and are explicit
// QWEN_GDN_NP / QWEN_GDN_NV opt-ins) and BH <= (grid.x / NV) * grid.y receiver row rectangles;
// validate FATALs otherwise, so the op-level dispatch must gate on grid size before choosing this path.
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
    bool v_flat = false,
    uint32_t HV = 0,
    bool qk_norm = false,
    float scale = 1.0f,
    bool qk_flat = false,
    uint32_t Hk = 0);

}  // namespace ttnn::prim
