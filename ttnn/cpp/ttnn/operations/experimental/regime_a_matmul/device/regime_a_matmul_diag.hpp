// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::experimental::prim {

// Test-only diagnostic ablation bitmask (default 0 = the normal public path). INTERNAL-ONLY: NEVER exposed
// through the Python / nanobind API and NOT reachable from the public op header (regime_a_matmul.hpp) — it is
// set only via the internal ttnn::prim::regime_a_matmul_diag entry point and this header is included only by
// the device program factory. It lives in RegimeAMatmulParams::diag_mask (as a plain uint32_t) so it
// participates in the program-cache hash — a diagnostic program can never alias (or be aliased by) a normal
// one. Each bit maps to a DIAG_* kernel #define; mask 0 adds NO defines and the compile is byte-identical to
// the public path. Ablations are non-additive critical-path counterfactuals (see MT8_FINDINGS.md), not
// correctness modes — output is intentionally unchecked.
//
// The explicit `1u << N` values are STABLE (harness / program-cache compatibility): removed diagnostics leave
// their bits permanently unused rather than renumbering the survivors. Currently FREE bits: 0, 1, 2, 7, 8, 9,
// 13, 15, 17, 20, 23, 24, and >=26. (Bits 0/1/2 were skip-in1/in0-read + skip-in0-forward ablations;
// 4 was a reserved local-feed; 5 was in0-scatter; 6/7 were in0-replicated-ring; 8/9 were in0-direct-exchange
// — all removed after the in0-delivery study concluded the ring is optimal. Bits 13/15/17 were dominated ring
// objectives; 20/23/24 never assigned. Recover removed variants from git history if ever needed.)
enum RegimeADiag : uint32_t {
    DIAG_NO_REDUCE = 1u << 3,  // force bottom-band copy path everywhere; bypass reduce credits/recv/fwd
    // Test-only causal timing zones (DeviceZoneScopedN) around kernel phases: startup / in0-ring / in1-read /
    // compute / split-K reduce / output. Compile-gated so mask 0 has NO zone overhead (clean baseline); only
    // this bit activates the DeviceZoneScopedN markers. Perturbation = (DIAG_ZONES run - mask-0 run).
    DIAG_ZONES = 1u << 4,
    // Ring deferred-drain (test-only): replace the phase-1 completion write-barrier with the weakest safe
    // source-lifetime op (noc_async_writes_flushed) and defer remote-completion to the existing kernel-exit
    // write+atomic barrier. Hypothesis: the phase-1 barrier's remote-completion is redundant with the
    // fwd-semaphore protocol (payload->sem ordered, same NoC). Must stay PCC-exact + watcher-clean.
    DIAG_RINGDRAIN = 1u << 5,
    // Split-K reduction TOPOLOGY (test-only, Pk==4 ONLY): replace the linear reduction chain
    // (kk0->kk1->kk2->kk3, critical depth 3) with a fan-in-2 balanced tree (kk0->kk1 and kk2->kk3 at level 0,
    // then kk1->kk3 at level 1; critical depth 2). kk3 stays the root/is_top DRAM writer. Two disjoint receive
    // CHANNELS at the root (channel 0 = kk2's partial on red_sem, channel 1 = kk1's summed partial on a second
    // red_sem2) avoid the single-counter fungibility hazard of a shared semaphore; the root sums them serially
    // (round 0 then round 1) into the same reduce-add math as the chain, so output is bit-identical (addition
    // is associative). Host-side topology post-process only (factory rewrites red links + num_recv/channel);
    // the chain remains the production default and this bit selects the tree A/B. Must stay PCC-exact.
    // A/B OUTCOME (2026-07-22, fixed-config, 2 reversed batches x 10 relaunches, IQR-separated): the tree does
    // NOT beat the chain on the reduction-EXPOSED primary 256x2048x1024 (Sm=2) — it REGRESSES +1.07% (the
    // reduction tail is already overlapped by the 9us matmul floor, so the depth 3->2 saving is smaller than
    // the tree's extra channel-1 semaphore + serial 2-round root + reduced per-channel reduce-CB buffering).
    // It is neutral (-0.01%) on the deep-K read-bound control (32x6144x1536), but a real -4.31% WIN on the
    // tiny shallow 32x2048x512 (Sm=1, minimal compute -> reduction depth is a larger critical-path fraction).
    // => NOT adopted as the global default (regresses the primary); kept as this diagnostic. The 32x2048x512
    // win is a candidate for SHAPE-SELECTIVE (picker-gated) tree use — a separate, future decision.
    DIAG_REDTREE = 1u << 6,
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
    DIAG_BARRIER_DRAIN = 1u << 11,
    // Physical-topology-aware in0 ring ordering (host-side; overrides ring_pos/next/prev in the factory, no
    // kernel change). DEFAULT (neither bit) = OPT: exhaustive 7! cycle search per ring group minimizing max
    // edge cost then total hops over the group's WRITER-NoC authoritative hop distance
    // (get_worker_noc_hop_distance; both orientations covered). These bits select A/B diagnostics: BANK = the
    // old bank order [0..7] (previous production baseline); GREEDY = greedy nearest-neighbour. Ring ordering
    // only differs; placement/work/reduction unchanged; output bit-identical for any permutation.
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
    // removed after the decision. All cache-hashed; none exposed via the public API.
    DIAG_RING_OPT_MM0 = 1u << 14,
    DIAG_RING_TOTAL = 1u << 16,
    DIAG_RING_MAXEDGE = 1u << 18,
    // M-split (Sm>1) worker-PLACEMENT (host-side factory override of cp.coord only; logical core indices + all
    // ownership + the reader->i+s / slave->i-mm factory arg math are UNCHANGED; the PARETO ring order is
    // recomputed on the new coords). DEFAULT (no PLACE bit) = IN1_NEAR: place every mm==0 DRAM reader first
    // (so slaves can't displace later readers from bank-adjacent cores), then place each slave at the free
    // worker minimizing the directed reader->slave hop on the group's in1-reader NoC. Chosen by the two-run
    // placement A/B (production Sm=2 primary -6.3/-7.2%; all Sm>1 shapes faster; Sm=1 no-op). Diagnostics:
    //   DIAG_PLACE_CURRENT       : the planner's original reader-then-slaves logical-Manhattan spiral (baseline)
    //   DIAG_PLACE_READERS_FIRST : readers-first but bank-spiral slaves (inconsistent; +1.2% on one shape)
    // No effect at Sm==1. All cache-hashed; none exposed via the public API.
    DIAG_PLACE_READERS_FIRST = 1u << 19,
    DIAG_PLACE_CURRENT = 1u << 21,

    // in1-delivery optimization (M-split follow-up + optimization #5). ADOPTED as production defaults after a
    // gated A/B on the Mt<=8 campaign winners: forward-signal-first (per-block flush moved AFTER the valid
    // signal, releasing the slave without waiting on the reader's flush) is a consistent +1.0..+2.3% on Sm>1
    // shapes, and coalesced contiguous-block reads are +0.5..+3.8% (both vs the old order), zero-regression on
    // controls (wide-N / deep-K / bandwidth-bound) and PCC-exact. The DEFAULT path (mask 0) does BOTH; these
    // two bits select the OLD behaviour as the A/B baseline. All cache-hashed; none exposed via the public API.
    DIAG_FWD_FLUSH_FIRST = 1u << 22,  // A/B baseline: OLD write->flush->signal in1 forward. Default = write->
                                      // signal->flush: the per-block flush is RETAINED (moved after the signal)
                                      // so the async write departs this CB slot before it is reused (source
                                      // lifetime); it is merely off the slave-release critical path.
    DIAG_NO_COALESCE = 1u << 25,      // A/B baseline: OLD K_block per-row in1 reads (default now issues one
                                      // coalesced read per physically-contiguous block; falls back per-row).
};

}  // namespace ttnn::experimental::prim
