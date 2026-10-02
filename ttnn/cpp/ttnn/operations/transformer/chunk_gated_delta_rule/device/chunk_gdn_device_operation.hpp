// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The chunk_gated_delta_rule device operation: ONE device op, two program factories for one contract —
// inputs {q, k, v, g, beta, the constant tiles, initial_state} -> {o [BH,NC,C,V], final_state [BH,K,V]}.
//   Mono  — the original single-kernel program: one core per head runs prep and scan for every chunk.
//           The slowest path, kept as the benchmark/debug reference; o is bf16 (the kernel's cb_out).
//   Fused — ONE program with zero DRAM intermediates: per head, NP PRODUCER cores run the prep
//           reader+compute and a writer that NoC-writes the 7 computed intermediates (v_beta, nkd,
//           q_decay, intra, k_dec_t, dl, t_inv) straight into the CBs of the head's NV RECEIVER cores
//           (each carrying a V-slice) through a credit/valid handshake; the receivers run the scan
//           compute+writer. NP and NV come from ChunkGdnFusedProgramConfig::num_producers /
//           num_receivers, or from the cost model when unset. o is fp32 — exactly the phased scan's
//           output spec — and fused == phased bit for bit as long as the shared math bodies and CB pack
//           boundaries are untouched.
// The program-config alternative names the factory, as a matmul program config names its factory, and
// every field that changes the program is a hashed attribute. The phased prims (chunk_gdn_phased.hpp) are
// a different contract — prep -> seven fp32 DRAM tensors -> scan, two programs — and stay separate ops:
// they are the bit-exact reference the fused path is gated against.

#pragma once

#include "chunk_gdn_compute_config.hpp"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule_config.hpp"

namespace ttnn::prim {

// Which program factory runs. Hashed with the other attributes, so the two programs never share a cache entry.
enum class ChunkGdnImpl : uint8_t { Mono, Fused };

// Union of the prep and scan params (see chunk_gdn_phased.hpp for per-field semantics of the
// prep-side v_flat/HV/qk_flat/Hk/qk_norm/scale block and the scan-side state flags), plus the fused
// geometry. Mono uses the shape block and the state flags only; it rejects the flat forms in validate.
struct ChunkGdnParams {
    ChunkGdnImpl impl = ChunkGdnImpl::Fused;
    uint32_t BH = 0;
    uint32_t num_chunks = 0;
    uint32_t chunk_size = 0;
    uint32_t key_dim = 0;
    uint32_t val_dim = 0;
    bool v_flat = false;
    uint32_t HV = 0;
    bool qk_flat = false;
    uint32_t Hk = 0;
    bool qk_norm = false;
    float scale = 1.0f;
    // Every geometry/transport field below is resolved from the ChunkGdnFusedProgramConfig (or the
    // cost model, for the fields it leaves free) at attrs construction — never in the factory: these
    // fields being hashed is what keeps the program cache honest. Mono leaves them at their defaults.
    // Producers per head (NP) in placements 0 and 1: producer j of a head owns chunks c = j, j+NP, ... and
    // NoC-writes them into the head's receivers in order (receiver-driven rotating ready credits);
    // clamped to num_chunks. In placement 2 the SIZE of the producer pool P (at most BH * num_chunks).
    uint32_t np = 1;
    // Receivers per head (NV): a head's NV receiver cores form a 1xNV row rectangle and each carries a
    // V-slice of Vt/NV tiles (the phased scan's V-block split, fed over the NoC).
    uint32_t nv = 1;
    // Hand-off CB depth (slots per CB): how many chunks a producer may run ahead of a receiver's
    // consumption, and how early a receiver can reserve+credit the next chunk. The receiver keeps nbuf-1
    // hand-offs in flight (per-slot VALID flags, BH x nbuf credit words) at +76 KB of L1 per slot on
    // every core. The config's handoff_depth, or the cost model's pick (2, or 3 where the depth-2 round
    // trip is exposed: Vtl=1, or the supply within 10 % of the step).
    uint32_t nbuf = 2;
    // Ship the six shared tensors and the v_beta slices as NV plain unicast writes per item instead of
    // a linked multicast chain. Multicasts reserve router ports along their path; the zone captures show
    // sporadic 20-120 us multicast-issue stalls on individual producers. The default; unicast=false
    // restores the multicast chain for A/B.
    bool unicast = true;
    // With the unicast transport, ship the data as POSTED writes (no acks,
    // no per-item write barrier) and order the VALID flag behind them by the NoC's in-order delivery
    // on one (source, destination, VC, command buffer) — the argument tt-metal's matmul multicast
    // sender uses. Requires unicast.
    bool posted = false;
    // Placement: 0 = receivers row-major from row 0, producers fill the rest;
    // 1 = ROW-LOCAL: one head per row (receivers in columns 0..NV-1, its NP producers to their east in
    // the same row), heads beyond grid.y in the leftover columns as vertical blocks. NOC_1 routes -x
    // then -y, so a head's hand-off traffic never leaves its own row (or column block) and heads do not
    // share NoC links. The config's row_local, or row-local whenever it is feasible.
    // 2 = PRODUCER POOL: the row-local layout of NV receivers + NPH home producers per head for the
    // largest NPH with BH*NPH <= np, plus np - BH*NPH EXTRA producers on the remaining cores; the
    // extras serve every head (the item map of kernels/dataflow/chunk_gdn_fused_map.hpp).
    uint32_t placement = 0;
    // Placement 2: the extras' share of each head's chunks, pool_extra_num / pool_extra_den (0 / 1
    // without extras).
    uint32_t pool_extra_num = 0;
    uint32_t pool_extra_den = 1;
    // WY-inverse method of the producer's prep compute (GdnTinv, chunk_gdn_compute_config.hpp).
    GdnTinv tinv = GdnTinv::HORNER;
    bool output_final_state = false;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ChunkGdnInputs {
    Tensor q;        // [BH, NC, C, K] bf16 (or FLAT [B, T, Hk*K] bf16 when params.qk_flat)
    Tensor k;        // [BH, NC, C, K] bf16 (or FLAT, as q)
    Tensor v;        // [BH, NC, C, V] bf16 (or FLAT [B, T, HV*V] bf16 when params.v_flat)
    Tensor g;        // [BH, NC, C, 1] fp32 (column)
    Tensor beta;     // [BH, NC, C, 1] fp32 (column)
    Tensor eye_c;    // [1,1,C,C] fp32
    Tensor tril_c;   // [1,1,C,C] fp32
    Tensor ones_c;   // [1,1,C,C] fp32
    Tensor masks_c;  // [1,1,32,96] fp32 — three 32x32 WY-inverse quadrant masks (Qtl|Qbr|Q10); Fused only
    Tensor
        initial_state;  // [BH, K, V] fp32, REQUIRED: both programs read it unconditionally (zeros for a fresh sequence)
};

// One core per head, all chunks in sequence (chunk_gdn_mono_program_factory.cpp).
struct ChunkGdnMonoProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ChunkGdnParams&, const ChunkGdnInputs&, std::vector<Tensor>&);
};

// NP producers -> NV receivers per head over the NoC (chunk_gdn_fused_program_factory.cpp).
struct ChunkGdnFusedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ChunkGdnParams&, const ChunkGdnInputs&, std::vector<Tensor>&);
};

struct ChunkGdnDeviceOperation {
    using operation_attributes_t = ChunkGdnParams;
    using tensor_args_t = ChunkGdnInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ChunkGdnMonoProgramFactory, ChunkGdnFusedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// ---------------------------------------------------------------------------------------------------
// Fused geometry: the row-local cost model, calibrated on QB2:
//   T_fused = fill(BH, producers) + max(chain, home, extra) * (1 + balance) + tail, with chain = (NC-1) steps
//   plus the head skew, home / extra = the remaining items of the busiest home / extra producer, and balance
//   the bump two near-equal bounds cost each other (Vtl <= 2); over (NV | Vt, NP, depth in {2, 3}) with a
//   feasible layout (the row-major fallback is link-bound), and for a pool of P serving every head
//   (placement 2) at the balanced share NX / P. Ties -> fewer cores, then smaller NV, then the shallower
//   ring. T_phased(BH) from the measured table.
// The op host uses it for whichever of num_receivers / num_producers / row_local / handoff_depth the
// fused program config leaves free, and to decide fused vs phased when no program config is given;
// test_chunk_gdn_fused_geometry.py checks it against a Python oracle on several grids.
// ---------------------------------------------------------------------------------------------------
struct FusedGeometryChoice {
    uint32_t nv = 0;  // 0 => no fused geometry fits this grid
    uint32_t np = 0;  // producers per head; the pool size P when placement == 2
    uint32_t placement = 0;  // 1 row-local, 0 row-major fallback, 2 producer pool
    uint32_t nbuf = 2;       // hand-off depth
    float t_fused_us = 0.0f;
    float t_phased_us = 0.0f;
    bool fused_pays = false;
};
// Which geometries the model considers: NP producers per head (placements 0/1), the producer pool
// (placement 2: P = every core the receivers leave), or both (the default dispatch).
enum class FusedCandidates : uint8_t { PerHead = 0, Pool = 1, Both = 2 };
bool fused_row_local_feasible(uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t NP);
// Producer pool of P for NV receivers per head: the home producers per head — the largest NPH with
// BH*NPH <= P that has a row-local layout — or 0 when the pool does not fit.
uint32_t fused_pool_home_producers(uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t P);
bool fused_pool_feasible(uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t P);
// fixed_nv / fixed_np / fixed_nbuf = 0 -> free; a non-zero value pins that field and the model chooses
// the others so the pair still fits (and prefers a row-local layout for it). With the pool candidates
// fixed_np pins the pool size.
FusedGeometryChoice choose_fused_geometry(
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t BH,
    uint32_t NC,
    uint32_t Vt,
    uint32_t fixed_nv = 0,
    uint32_t fixed_np = 0,
    uint32_t fixed_nbuf = 0,
    FusedCandidates candidates = FusedCandidates::Both);

// Design D9: the fused program's core map, a pure function of its arguments (no device), shared by
// the program factory and the nanobind geometry oracle. placement 0 = row-major 1xNV receiver
// rectangles with the producers on the remaining cores row-major; 1 = row-local (a head's receivers
// and producers in one row segment, leftover heads as column blocks); 2 = producer pool (NP is the
// pool size: the row-local map of home_per_head producers per head, then the extras row-major on
// the remaining cores). FATALs when the layout does not fit, exactly as the factory would.
struct FusedPlacement {
    std::vector<tt::tt_metal::CoreCoord> receivers;  // index h*NV + v (logical coordinates)
    std::vector<tt::tt_metal::CoreCoord> producers;  // index h*home_per_head + j, then the extras
    uint32_t home_per_head = 0;                      // NP in placements 0/1
};
FusedPlacement fused_placement(
    uint32_t grid_x, uint32_t grid_y, uint32_t BH, uint32_t NV, uint32_t NP, uint32_t placement);

// The device-level program config: the Mono or Fused alternative of ChunkGdnProgramConfig (the phased
// alternative launches the two phased prims instead, see chunk_gated_delta_rule.cpp).
using ChunkGdnDeviceProgramConfig =
    std::variant<ttnn::transformer::ChunkGdnMonoProgramConfig, ttnn::transformer::ChunkGdnFusedProgramConfig>;

// Returns {o [BH,NC,C,V], final_state [BH,K,V]} — o bf16 from Mono, fp32 from Fused; final_state fp32.
// Fused: the geometry (NV receivers + NP producers per head, placement) comes from the program config,
// with the calibrated cost model filling whatever it leaves free. Needs BH*(NV+NP) cores and a placement
// that fits; validate FATALs otherwise, so the op-level dispatch must gate on grid size before choosing
// this path (choose_fused_geometry(...).nv == 0 means no geometry fits). Mono needs BH cores, does not
// accept flat q/k/v, and computes the WY inverse with Horner only: wy_inverse=AUTO resolves to Horner there
// and an explicit FORWARD_SUBSTITUTION TT_FATALs.
std::vector<Tensor> chunk_gdn(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye_c,
    const Tensor& tril_c,
    const Tensor& ones_c,
    const Tensor& masks_c,
    const Tensor& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const ChunkGdnDeviceProgramConfig& program_config,
    ttnn::transformer::ChunkGdnWyInverse wy_inverse,
    bool v_flat = false,
    uint32_t HV = 0,
    bool qk_norm = false,
    float scale = 1.0f,
    bool qk_flat = false,
    uint32_t Hk = 0);

}  // namespace ttnn::prim
