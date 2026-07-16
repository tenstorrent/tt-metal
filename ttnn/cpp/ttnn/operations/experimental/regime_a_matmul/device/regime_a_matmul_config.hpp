// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_plan.hpp"

namespace ttnn::experimental::prim {

// Public manual-config knobs for the Regime-A DRAM-BW-optimal matmul. Field names mirror the pure
// host planner's plan::RegimeAConfig. All values are in tiles / slice counts.
//
// IMPORTANT (program-cache identity): every field here feeds compile-time kernel args, so this whole
// struct must be part of RegimeAMatmulParams (the device-op operation_attributes) and is hashed via
// the framework's default reflection-based program hash (same mechanism as MinimalMatmulConfig).
struct RegimeAMatmulConfig {
    uint32_t k_slices{1};          // Pk : split-K depth (>=1). Reduction only when >1.
    uint32_t n_slices{1};          // Ns : N-slices per bank-band.
    uint32_t m_slices{1};          // Sm : M-split factor.
    uint32_t k_block_tiles{1};     // kb : K-block depth fed to compute (tiles).
    uint32_t n_subblock_tiles{0};  // nsb: N-subblock width (tiles). 0 => full N_own.
};

// Test-only diagnostic ablation bitmask (default 0 = the normal public path). NEVER exposed through the
// Python / nanobind API; set only via the internal ttnn::prim::regime_a_matmul_diag entry point. It lives
// in RegimeAMatmulParams::diag_mask so it participates in the program-cache hash — a diagnostic program can
// never alias (or be aliased by) a normal one. Each bit maps to a DIAG_* kernel #define; mask 0 adds NO
// defines and the compile is byte-identical to the public path. Ablations are non-additive critical-path
// counterfactuals (see MT8_FINDINGS.md), not correctness modes — output is intentionally unchecked.
enum RegimeADiag : uint32_t {
    DIAG_SKIP_IN1_READ = 1u << 0,     // in1_reader: suppress in1 DRAM reads/barrier/tail-init; keep CB+M-split+sems
    DIAG_SKIP_IN0_READ = 1u << 1,     // writer: suppress step-0 in0 DRAM read/barrier; keep ptr adv/CB/ring/reduce/out
    DIAG_SKIP_IN0_FORWARD = 1u << 2,  // writer: suppress ring payload write; still signal next core (no deadlock)
    DIAG_NO_REDUCE = 1u << 3,         // force bottom-band copy path everywhere; bypass reduce credits/recv/fwd
    DIAG_LOCAL_FEED = 1u << 4,        // (reserved) purely local CB feed: no DRAM/ring/fwd/M-split/reduce
    // NOTE: the bits below are CORRECT algorithmic VARIANTS (produce the right output), not ablations.
    DIAG_IN0_SCATTER = 1u << 5,  // writer: in0 all-gather via 1 direct-scatter round instead of G-1 serial
                                 // ring rotations. Identical cb0 layout (slot d = shard rp-d), so compute +
                                 // the in1 rotated read are unchanged; removes the serial-hop critical path.
    // Replicated shorter ring: each core reads R seed shards (stride G/R) and rotates the R-shard bundle
    // for G/R rounds (nearest-neighbor, incremental per-round push preserved). Cuts the forwarding
    // dependency depth from G-1 to G/R-1, trading R x in0 DRAM reads. cb0 slot (r*R+i) = shard (rp-r-i*G/R);
    // the in1 reader uses the SAME formula so compute is unchanged. Factory maps these to IN0_REPL=2/4.
    DIAG_IN0_REPL2 = 1u << 6,   // R=2: 2 seeds, 4 rounds, 6 forwarded shard-equiv (vs 7), depth 3
    DIAG_IN0_REPL4 = 1u << 7,   // R=4: 4 seeds, 2 rounds, 4 forwarded shard-equiv, depth 1
    DIAG_IN0_XCHG = 1u << 8,    // eager incremental direct exchange: per peer, write-then-signal (NoC
                                // ordering, NO flush) so each slot is exposed as its own write lands; push
                                // received slots in compute order. Ring cb0 layout -> in1/compute unchanged.
    DIAG_IN0_XCHGRR = 1u << 9,  // round-robin direct exchange: round d, write to the d-ahead peer + signal,
                                // wait own slot d, push it, advance. 1 transfer/core/round (less burst
                                // congestion than eager's G-1 at once) while keeping incremental push.
    // A/B baseline for the progressive-cumulative-wait schedule. The default (this bit CLEAR) resident-in0
    // compute path begins matmul as each ring shard arrives (cumulative cb_wait_front during the first N
    // sub-block); this bit restores the OLD single full-slice startup barrier before any matmul. Compute-only
    // define; identical config/tensors/transport/reduction, only the CB0 wait placement differs.
    DIAG_FULL_IN0_WAIT = 1u << 10,
    // Phase-2 (writer) drain discipline. DEFAULT (this bit CLEAR) = PIPELINED: reuse a source CB slot once its
    // payload has DEPARTED L1 (noc_async_writes_flushed), signal the reduction receiver with ordered
    // payload->semaphore ops (same peer + NoC, like the in0 ring), and defer completion to ONE final
    // write+atomic barrier before kernel return — so the split-K reduction chain and DRAM output pipeline
    // across bands / N-sub-blocks instead of stalling on a per-block completion barrier. This bit restores the
    // OLD per-block noc_async_write_barrier (wait remote completion before popping each CB2/CB7 slot), kept as
    // an A/B diagnostic. Identical config/tensors/compute/ring/reduction — only per-block write sync differs.
    // (bits 1<<12 .. 1<<14 are free — were grouped-K, removed; recover from 56b37f5d5e6/7b3f93ddaa5 +
    // a5b7986b18f/3593ecd4083. See tools/mm_sweep/GROUPED_K_REPORT.md.)
    DIAG_BARRIER_DRAIN = 1u << 11,
    // Physical-topology-aware in0 ring ordering (host-side; overrides ring_pos/next/prev in the factory, no
    // kernel change). DEFAULT (neither bit) = OPT: exhaustive 7! cycle search per ring group minimizing max
    // edge cost then total hops over the group's WRITER-NoC authoritative hop distance
    // (get_worker_noc_hop_distance; both orientations covered). These bits select A/B diagnostics: BANK = the
    // old bank order [0..7] (previous production baseline); GREEDY = greedy nearest-neighbour. Ring ordering
    // only differs; placement/work/reduction unchanged; output bit-identical for any permutation.
    // (bit 1<<14 is free — was grouped-K; see GROUPED_K_REPORT.md.)
    DIAG_RING_BANK = 1u << 12,
    // A/B diagnostics selecting the shared-permutation OBJECTIVE for the ring OPT (all score exhaustively over
    // the 7! cycles, aggregating across the Sm physical mm-rings of a (kk,nn) group). The DEFAULT (no ring
    // bit) is PARETO. Lexicographic tuples to MINIMIZE (aggmax = worst edge over rings; aggtot = summed hops
    // over all rings):
    //   OPT_MM0      : (ring0.max, ring0.total)                 — score mm==0 only (old objective / reference)
    //   RING_MAXEDGE : (aggmax, aggtot)                         — regressed the synthetic Sm=4 case
    //   RING_TOTAL   : (aggtot, aggmax)                         — regressed the common Sm=1 case
    //   [default]    : PARETO = min aggmax s.t. aggtot <= MM0's aggtot, then aggtot. Route-dominates MM0 by
    //                  construction (never worse total), so it cannot stably regress vs MM0; keeps the Sm=2
    //                  win and stays within noise of MM0 on Sm=1. Chosen after the two-run objective A/B.
    // These retained as the decision A/B set; the dominated `greedy` (heuristic) and `maxring` objectives were
    // removed after the decision (recover from the implementation commit; see GROUPED-K-style note in the
    // report). Bits 1<<13, 1<<15, 1<<17 are now free. All cache-hashed; none exposed via the public API.
    DIAG_RING_OPT_MM0 = 1u << 14,
    DIAG_RING_TOTAL = 1u << 16,
    DIAG_RING_MAXEDGE = 1u << 18,
};

namespace plan = ttnn::operations::experimental::regime_a_matmul::plan;

// Auto-select a (Pk, Ns, Sm, kb, nsb) config for a shape given in TILE counts. Ported from the
// validated FLUX/LTX picker (tools/mm_sweep/picker_table.py oracle + picker_v2.py cost-model
// fallback): a lookup table for the 20 production shapes, else enumerate feasible candidates (Sm=1)
// and pick the min-cost one. Used when the caller passes config=None.
RegimeAMatmulConfig auto_select_config(uint32_t Mt, uint32_t Kt, uint32_t Nt);

// Device adapter: read (Mt, Kt, Nt) from the tensors, fetch the compute grid + bank-adjacent worker
// assignments off the device, translate RegimeAMatmulConfig -> plan::RegimeAConfig, and run the pure
// planner. When cfg is nullopt, auto_select_config picks the config. Returns the plan result verbatim;
// the caller must TT_FATAL on !ok() with plan.error.
plan::PlanResult make_and_build_plan(
    tt::tt_metal::IDevice* device, const Tensor& in0, const Tensor& in1, const std::optional<RegimeAMatmulConfig>& cfg);

// Build the canonical DRAM width-sharded MemoryConfig for the Regime-A in1 (weight) tensor.
//
// Shard layout: [K_padded_rows, N_padded / 8] tiles across 8 DRAM banks (WIDTH_SHARDED, ROW_MAJOR).
// v1 pads with a CONFIG-INDEPENDENT alignment: K rounded up to a multiple of 8 tiles, N rounded up to
// a multiple of 8 tiles (so the shard spec is a function of (K, N) only, never of Pk/kb/Ns/nsb). See
// the design note in the .cpp — this is deliberately NOT the plan's config-padded Kt_s/Nt_s.
//
// weight_shape: logical [.., K, N] (elements). dtype/device carried for API completeness + validation.
tt::tt_metal::MemoryConfig create_regime_a_weight_memory_config(
    const ttnn::Shape& weight_shape, tt::tt_metal::DataType dtype, tt::tt_metal::IDevice* device);

}  // namespace ttnn::experimental::prim
