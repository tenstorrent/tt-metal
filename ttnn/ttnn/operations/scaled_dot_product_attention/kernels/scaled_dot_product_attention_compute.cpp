// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for scaled_dot_product_attention (FlashAttention-2).
//
// Per work unit (one q-chunk) the running online-softmax state lives in three
// resident fp32 CBs — cb_row_max (m), cb_row_sum (l), cb_out_accum (O) — and is
// folded over the KV loop. The S_q x S_kv score matrix is never materialized:
// cb_scores / cb_exp are sized to one Sq_chunk_t x Skv_chunk_t block.
//
// Every compute phase goes through a kernel_lib helper (matmul_block, reduce,
// eltwise mul/sub/add/copy, binary_sfpu<BinaryMax>, unary<Exp>/<Recip>). The
// first KV chunk initializes (m, l, O) directly (m_old = -inf => alpha = 0),
// matching the recurrence with the correction sub-steps elided.
//
// ---------------------------------------------------------------------------
// RAW-LLK HELPER SUBSTITUTION (R3e) — the fused exp + row-sum dual-pack.
// ---------------------------------------------------------------------------
// Phase 6 (P = exp(scores - m)) has TWO variants selected at compile time by
// `fuse_rowsum` (host-gated to fp32_dest_acc_en=False, the throughput regime):
//   * fuse_rowsum == false: the helper eltwise_chain (sub<Col> -> Exp -> Pack),
//     byte-identical to R3c/R3d. A separate reduce<SUM,REDUCE_ROW> computes the
//     per-chunk row-sum (phase 7). This is the max-precision path, untouched.
//   * fuse_rowsum == true: `fused_exp_dual_pack` (raw LLK) packs the exp result
//     to cb_exp AND, in the SAME DEST window, L1-accumulates the element-wise
//     partial row-sum across the skv column tiles into cb_sum_chunk[i] via
//     `pack_tile<true>` + `pack_reconfig_l1_acc`. The dedicated per-chunk
//     reduce<SUM> is then eliminated; the running sum is kept in partial
//     (rows x 1, 32-col, un-reduced) form across the KV loop and collapsed to a
//     scalar ONCE after the loop (phase 11) with a single reduce<SUM,REDUCE_ROW>
//     (the FPU matmul-with-ones). This is production SDPA's approach
//     (compute_common.hpp sub_exp_block_bcast_cols_inplace do_reduce +
//     matmul_reduce); it is exact because rowsum is linear, so it commutes with
//     the per-chunk alpha rescale and the running accumulate.
//
// WHY RAW LLK (helper limitation, not a shortcut): the kernel_lib eltwise chain
// is SINGLE-TERMINAL — a chain that uses L1 accumulation (OutputLifecycle::
// L1Accumulation) must have EVERY PackTile be L1-accumulating and target ONE CB
// (eltwise_chain.inl static_asserts `all_writers_l1_accumulation` and
// `pack_dfbs_consistent`). So a chain CANNOT co-pack a normal cb_exp AND an
// L1-accumulating cb_sum_chunk from one DEST window. The fused dual-pack is
// therefore not expressible with helpers and is hand-written here. cb_exp and
// cb_sum_chunk share `interm_format` (bf16 in this regime), so the two pack
// targets need no pack_reconfig_data_format between them.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/bcast.h"                 // sub_tiles_bcast_cols, sub_bcast_cols_init_short
#include "api/compute/eltwise_unary/exp.h"     // exp_tile, exp_tile_init
#include "api/compute/pack.h"                  // pack_tile, pack_reconfig_l1_acc
#include "api/compute/reconfig_data_format.h"  // reconfig_data_format, pack_reconfig_data_format
#include "api/compute/reg_api.h"               // tile_regs_acquire/commit/wait/release

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

// MEASUREMENT-ONLY per-phase profiling (Perf: attribute the compute-bound residual
// across the ~8 serialized helper phases per KV chunk). DeviceZoneScopedN records a
// begin/end cycle timestamp per zone per RISC into profile_log_device.csv; the MATH
// thread's per-phase span is the clock-invariant per-phase cost. Present ONLY when the
// descriptor injects -DSDPA_ZONE_PROFILE (env SDPA_ZONE_PROFILE=1) — ABSENT and fully
// no-op in every shipped build (byte-identical), so it never perturbs the perf harness.
#if defined(SDPA_ZONE_PROFILE)
#include "tools/profiler/kernel_profiler.hpp"
#define SDPA_ZONE(name) DeviceZoneScopedN(name)
#else
#define SDPA_ZONE(name)
#endif

namespace ckl = compute_kernel_lib;

namespace {
// Derive the matmul N-subblock decomposition for an N tile-count (R1b). Replaces the
// phase-0 host `_matmul_subblocks` (single source of truth now on-device — the
// partial chunk's subblock count re-derives without a second host formula).
// out_subblock_h is always 1 (=> SubblockMajor output is tile-row-major, which the
// reduce and Col-broadcast steps downstream require), so in0_num_subblocks = M is
// passed directly and only the N split (in1_num_subblocks, out_subblock_w) is derived
// here. Largest divisor of n that is <= dest_limit. n >= 1 always, so out_subblock_w
// lands in [1, min(n, dest_limit)].
FORCE_INLINE void decomp_n(uint32_t n, uint32_t dest_limit, uint32_t& in1_num_subblocks, uint32_t& out_subblock_w) {
    uint32_t w = (n < dest_limit) ? n : dest_limit;
    while (w > 1 && (n % w) != 0) {
        --w;
    }
    out_subblock_w = w;
    in1_num_subblocks = n / w;
}

// R5 (perf): grow the matmul output-subblock HEIGHT to fill the DEST budget. decomp_n
// (above, R3a) picks out_subblock_w toward dest_limit, but out_subblock_h was hard-1, so
// a matmul whose N tile-count < dest_limit (the PV matmul: N=Dt=4, dest_limit=8 in the
// fp32_dest_acc_en=False throughput regime) used only out_subblock_w=4 of the 8-tile DEST
// per subblock pass — half empty. Growing the height to (dest_limit / out_subblock_w)
// packs 2x the tiles per pass (fewer tile_regs_acquire/commit/pack cycles, same FMA work).
//
// SAFE ONLY when out_subblock_w == n (in1_num_subblocks == 1): with a single N-subblock the
// SubblockMajor pack lays tiles out tile-row-major for ANY height (subblock (sb_m,0) is rows
// [sb_m*h .. +h) x all N cols, row-major within, concatenated in sb_m order => full
// row-major), which the downstream reduce / Col-broadcast steps require. When n splits into
// multiple N-subblocks (out_subblock_w < n) a height > 1 would interleave subblock rows and
// break row-major, so h stays 1 there. Also requires h | m. Self-gating: the fp32-DEST
// regime (dest_limit=4, PV out_subblock_w=4=budget) yields hmax=1 => byte-identical.
FORCE_INLINE void decomp_h(
    uint32_t m,
    uint32_t out_subblock_w,
    uint32_t n,
    uint32_t dest_limit,
    uint32_t& in0_num_subblocks,
    uint32_t& out_subblock_h) {
    uint32_t h = 1;
    if (out_subblock_w == n) {            // single N-subblock => output stays tile-row-major for any h
        h = dest_limit / out_subblock_w;  // h * out_subblock_w <= dest_limit (fill DEST)
        if (h > m) {
            h = m;
        }
        while (h > 1 && (m % h) != 0) {  // largest divisor of m within the budget
            --h;
        }
    }
    out_subblock_h = h;
    in0_num_subblocks = m / h;
}

constexpr uint32_t cb_q_in = 0;
constexpr uint32_t cb_k_in = 1;
constexpr uint32_t cb_v_in = 2;
constexpr uint32_t cb_mask_in = 3;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_scale = 5;
constexpr uint32_t cb_m_new = 6;
constexpr uint32_t cb_sum_chunk = 7;
constexpr uint32_t cb_kv_mask = 8;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_q_scaled = 24;
constexpr uint32_t cb_scores = 25;
constexpr uint32_t cb_exp = 26;
constexpr uint32_t cb_row_max = 27;
constexpr uint32_t cb_row_sum = 28;
constexpr uint32_t cb_pv = 29;
constexpr uint32_t cb_out_accum = 30;
constexpr uint32_t cb_corr = 31;

// R3e: fused exp + partial-row-sum dual-pack (raw LLK — see file-head comment).
// P = exp(scores - m) is packed to cb_exp (row-major, for the PV matmul) AND, in
// the SAME DEST window, the element-wise partial row-sum across this chunk's
// skv_valid column tiles is L1-accumulated into cb_sum_chunk[i] (rows x 1, 32-col,
// un-reduced). No dedicated per-chunk reduce<SUM> read of cb_exp; the intra-tile
// column collapse is deferred to a single reduce after the KV loop (phase 11).
//
// Reached only when fp32_dest_acc_en=False, so DEST is 16 bf16 tiles (8 per
// half-sync section) and skv_valid <= Skv_chunk_t <= 8 fits one acquire section.
// cb_scores/cb_exp/cb_sum_chunk are all interm_format (bf16) here, so the two
// pack targets share a data format (no pack_reconfig_data_format between them).
FORCE_INLINE void fused_exp_dual_pack(uint32_t sq_valid, uint32_t skv_valid) {
    const uint32_t sq_skv = sq_valid * skv_valid;
    cb_wait_front(cb_scores, sq_skv);
    cb_wait_front(cb_row_max, sq_valid);  // running max (held; NOT popped by this phase)
    cb_reserve_back(cb_exp, sq_skv);
    cb_reserve_back(cb_sum_chunk, sq_valid);

    // LIGHTWEIGHT reconfig (mirrors the eltwise_chain's own lowering — see file-head
    // note), NOT a full init_bcast: reconfig the unpack/math src formats to
    // (scores, row_max), switch the math/unpack op-type to sub-bcast-col, init the
    // fast SFPU exp (matches ckl::Exp<Approx::Fast> == exp_tile<true>, default
    // ClampToNegative), and reconfig ONLY the pack DATA FORMAT to cb_exp. A full
    // init_bcast (llk_pack_hw_configure + llk_pack_init) would re-init the packer and
    // clobber the boot-time matmul_block_init packer state that the per-KV-chunk
    // matmul's InitMode::Short relies on (Short does not fully re-issue packer cfg),
    // drifting across chunks x work-units. L1-acc is forced off so the cb_exp packs
    // below never accumulate.
    ckernel::reconfig_data_format(cb_scores, cb_row_max);
    ckernel::sub_bcast_cols_init_short(cb_scores, cb_row_max);
    ckernel::exp_tile_init<true>();
    ckernel::pack_reconfig_data_format(cb_exp);
    ckernel::pack_reconfig_l1_acc(0);

    for (uint32_t i = 0; i < sq_valid; ++i) {
        ckernel::tile_regs_acquire();
        for (uint32_t j = 0; j < skv_valid; ++j) {
            // DEST[j] = scores[i,j] - m[i]  (per-row max broadcast across columns)
            ckernel::sub_tiles_bcast_cols(cb_scores, cb_row_max, i * skv_valid + j, i, j);
            ckernel::exp_tile<true>(j);
        }
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        // (a) normal pack: exp -> cb_exp[i*skv_valid + j] (l1_acc still 0).
        for (uint32_t j = 0; j < skv_valid; ++j) {
            ckernel::pack_tile<true>(j, cb_exp, i * skv_valid + j);
        }
        // (b) L1-accumulate the row's skv_valid column tiles into cb_sum_chunk[i]:
        // seed with the first column (l1_acc=0 => overwrite), accumulate the rest
        // (l1_acc=1). DEST is unchanged by the (a) packs, so it is re-read here.
        ckernel::pack_tile<true>(0, cb_sum_chunk, i);
        ckernel::pack_reconfig_l1_acc(1);
        for (uint32_t j = 1; j < skv_valid; ++j) {
            ckernel::pack_tile<true>(j, cb_sum_chunk, i);
        }
        ckernel::pack_reconfig_l1_acc(0);  // reset before next row / downstream ops
        ckernel::tile_regs_release();
    }
    cb_push_back(cb_exp, sq_skv);
    cb_push_back(cb_sum_chunk, sq_valid);
    cb_pop_front(cb_scores, sq_skv);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Skv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t n_kv_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t has_mask_v = get_compile_time_arg_val(4);
    constexpr bool has_mask = has_mask_v != 0;
    constexpr uint32_t skv_partial = get_compile_time_arg_val(5);  // valid cols in last S_kv tile (0 => aligned)
    constexpr bool has_kv_pad = skv_partial != 0;
    constexpr uint32_t Sq_t = get_compile_time_arg_val(6);
    constexpr uint32_t n_q_chunks = get_compile_time_arg_val(7);
    constexpr uint32_t Skv_t = get_compile_time_arg_val(8);
    constexpr uint32_t dest_limit = get_compile_time_arg_val(9);
    // R3c (perf): fast/approximate SFPU exp for the dominant P=exp phase, gated by
    // the host to the fp32_dest_acc_en=False throughput regime only (loose tolerance
    // absorbs the approximation; the max-precision regime stays exact -> no
    // regression). The exp's Approx template is a compile-time value.
    constexpr bool use_fast_exp = get_compile_time_arg_val(10) != 0;
    constexpr ckl::Approx exp_approx = use_fast_exp ? ckl::Approx::Fast : ckl::Approx::Exact;
    // Perf: in the throughput regime (use_fast_exp == !fp32_dest_acc_en) the whole compute
    // chain is uniform bf16 — inputs, intermediates, AND accumulators (the accumulators now
    // follow interm_format) — so no data-format reconfig is ever needed. Drop them (None).
    // The fp32-DEST regime is mixed-format (bf16 in, fp32 interm/accum) so it keeps Input/Output.
    // NOTE: the LLK already runtime-skips a reconfig whose formats match, so this mainly removes
    // the per-call equality check; measured to quantify that residual.
    constexpr auto RCFG = use_fast_exp ? ckl::BinaryDataFormatReconfig::None : ckl::BinaryDataFormatReconfig::Input;
    constexpr auto PRCFG = use_fast_exp ? ckl::PackTileReconfig::None : ckl::PackTileReconfig::Output;
    // R3e (perf): fuse the per-chunk row-sum reduce into the exp pack via packer L1
    // accumulation (raw-LLK dual-pack — see file-head comment). Host-gated to the
    // fp32_dest_acc_en=False throughput regime (== use_fast_exp), so the softmax
    // intermediates are bf16 and the max-precision path stays byte-identical.
    constexpr bool fuse_rowsum = get_compile_time_arg_val(11) != 0;
    // R4 (causal): block-skip whole future KV chunks + apply the on-device triangular
    // mask on straddling chunks. is_causal SUBSUMES the R1 KV-padding mask (padding
    // keys are always in the future), so has_kv_pad's phase-3b add is disabled when
    // causal is set (below) to avoid a double mask.
    constexpr bool is_causal = get_compile_time_arg_val(12) != 0;
    // R5 (perf): PV matmul output-subblock HEIGHT knob (see decomp_h). MEASURED flat on the
    // flagged shape — filling the full half-sync DEST section per subblock defeats the
    // intra-DEST math/pack pipeline that h=1 (4-tile subblocks) enables — so it is PARKED at
    // its trivial default (0 => h=1, byte-identical to R4). The host sets this to 1 only when
    // SDPA_PV_SB_H=1 (same-session A/B re-measurement); unset => 0 (parked). decomp_h also
    // self-gates (h=1 for fp32-DEST), so even enabled it is inert outside the throughput regime.
    constexpr bool grow_subblock_h = get_compile_time_arg_val(13) != 0;
    // R5a (perf, MEASUREMENT-ONLY ablation): bound the PV-matmul + O-rescale/accumulate
    // headroom (the max any PV-batching lever could remove) by stubbing that payload while
    // keeping every CB reserve/wait/pop/push intact (/perf-measure ablation method).
    //   0 = off (normal, shipped path).
    //   1 = stub the PV matmul FMA+pack only (drain cb_exp/cb_v like the real matmul,
    //       push a garbage cb_pv). Rescale/accumulate/normalize run normally. -> PV-matmul cost.
    //   2 = also stub the per-chunk O rescale (phase 8) + accumulate (phase 10): drain
    //       cb_corr(alpha) + cb_pv, leave cb_out_accum resident from the first-chunk copy.
    //       -> PV-matmul + rescale + accumulate cost (the whole "PV+accum" zone).
    //   3 = also stub the QK^T matmul (phase 2) -> total matmul + accum headroom (both FPU
    //       matmuls off; softmax/reduces run on garbage). Bounds the whole matmul-efficiency
    //       lever class (R5/R5a) against the SFPU-softmax residual.
    // Gated by env SDPA_ABLATE_PV in the descriptor; 0 for every shipped build.
    constexpr uint32_t ablate_pv = get_compile_time_arg_val(14);
    // Perf 2 (MEASUREMENT-ONLY ablation): stub the SOFTMAX payloads — the per-chunk
    // row-max reduce (phase 4) and the fused exp dual-pack (phase 6) — while keeping
    // every CB reserve/wait/pop/push intact. Combined with ablate_pv=3 (matmuls+accum
    // stubbed) this isolates the PURE per-phase overhead floor (init/reconfig/fill-drain/
    // CB-sync of every phase + the tiny scalar phases 5/7 + q-scale + normalize) from the
    // softmax PAYLOAD (the SFPU exp + the row-max reduce math). Answers the question the
    // prior refinements estimated but never freshly ablated: is the compute-bound residual
    // overhead-bound (only coarsening helps -> L1/divisor-blocked) or payload-bound
    // (attacking exp/reduce could help)? Gated by env SDPA_ABLATE_SOFTMAX; 0 for every
    // shipped build (compile-time-elided at the default).
    constexpr uint32_t ablate_softmax = get_compile_time_arg_val(15);
    const uint32_t num_wu = get_arg_val<uint32_t>(0);
    const uint32_t start_wu = get_arg_val<uint32_t>(1);

    // PV pack width (N = Dt, constant across the loop) — use the optimal decomposition.
    uint32_t pv_in1_sb = 0, pv_out_sb_w = 0;
    decomp_n(Dt, dest_limit, pv_in1_sb, pv_out_sb_w);
    // QKᵀ pack width (out_subblock_w) must stay CONSTANT across the KV loop:
    // mm_block_init_short (the per-call InitMode::Short) does not fully re-issue the
    // packer config, so changing out_subblock_w mid-loop to a non-power-of-2 partial
    // width (e.g. 4 -> 3) wedges the packer (are_packers_configured_correctly). When
    // Skv_t divides Skv_chunk_t there is no partial chunk (incl. the perf-flagged
    // shape, Skv_t%4==0) -> keep the optimal full-width decomposition; otherwise force
    // out_subblock_w = 1 (safe constant; only the small generality shapes with a
    // partial KV chunk pay the narrower subblock — the perf path is untouched). Only
    // the subblock COUNT (in1_num_subblocks) then varies per chunk, which is a loop
    // bound, not a packer reconfig.
    constexpr bool has_partial_kv = (Skv_t % Skv_chunk_t) != 0;
    uint32_t qk_full_in1_sb = 0, qk_full_out_sb_w = 0;
    decomp_n(Skv_chunk_t, dest_limit, qk_full_in1_sb, qk_full_out_sb_w);
    const uint32_t qk_out_sb_w = has_partial_kv ? 1u : qk_full_out_sb_w;

    using ckl::BroadcastDim;
    using ckl::Dst;
    using ckl::EltwiseShape;
    using ckl::InputLifecycle;
    using ckl::LastBlockTarget;
    using ckl::MatmulBlockShape;
    using ckl::OperandKind;
    using ckl::OutputCBLayout;
    using ckl::OutputLifecycle;
    using ckl::ReduceInputBlockShape;
    using ckl::ReduceInputPolicy;

    // R6 (perf): fuse the online-softmax O-accumulate (former phase 10) into the PV matmul
    // (phase 9) via packer L1-accumulation onto cb_out_accum, eliminating a whole serial phase
    // and the cb_pv CB. Gated to the throughput regime (fuse_rowsum) AND no-partial-q-chunk
    // (Sq_t % Sq_chunk_t == 0, so sq_valid == Sq_chunk_t for every work unit => cb_out_accum is a
    // FULL ring, wr_ptr wraps to the block base each push, so after phase 8's in-place rescale the
    // packer write pointer lands exactly on the resident alpha*O for the matmul to L1-accumulate
    // P*V onto in place). The fp32-DEST/max-precision path AND the rare prime-Sq_t partial-q
    // throughput path stay on the current P08/P09/P10 + cb_pv structure (byte-identical). See the
    // phase 8/9/10 blocks below and the CB-choreography note there.
    constexpr bool fuse_oaccum = fuse_rowsum && ((Sq_t % Sq_chunk_t) == 0);

    // CircularBuffer wrappers for the two matmuls (+ the fused-accumulate O target).
    ::CircularBuffer q_scaled_buf(cb_q_scaled), k_buf(cb_k_in), scores_buf(cb_scores);
    ::CircularBuffer exp_buf(cb_exp), v_buf(cb_v_in), pv_buf(cb_pv), out_accum_buf(cb_out_accum);

    // Boot: matmul-order hw config + one full matmul_block_init (helper's Short
    // init restores state after intervening eltwise/reduce ops per call).
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_q_scaled, cb_k_in, cb_scores);
    matmul_block_init(cb_q_scaled, cb_k_in, /*transpose=*/1, qk_out_sb_w, /*out_subblock_h=*/1, Dt);

    for (uint32_t wu = 0; wu < num_wu; ++wu) {
        // R1b: partial q-chunk. Decode qc for this work unit (qc = w % n_q_chunks,
        // since n_q_chunks divides H*n_q_chunks) and take only the valid tile-rows
        // of the last q-chunk. sq_valid is the M extent for every phase below.
        const uint32_t w = start_wu + wu;
        const uint32_t qc = w % n_q_chunks;
        const uint32_t sq_off = qc * Sq_chunk_t;
        const uint32_t sq_valid = (Sq_chunk_t < Sq_t - sq_off) ? Sq_chunk_t : (Sq_t - sq_off);
        const uint32_t sq_dt = sq_valid * Dt;

        // R5 (perf): PV matmul output-subblock decomposition. out_subblock_w (= pv_out_sb_w,
        // loop-invariant) is decomp_n(Dt); the HEIGHT grows to fill the DEST budget when the
        // output is single-N-subblock (row-major-safe) — see decomp_h. M = sq_valid varies
        // per q-chunk (partial last chunk), so the height is re-derived per work unit. When
        // the lever is off (grow_subblock_h=0) or decomp_h finds no room this is h=1,
        // in0_num_subblocks=sq_valid (the pre-R5 decomposition).
        uint32_t pv_in0_sb = sq_valid, pv_sb_h = 1;
        if constexpr (grow_subblock_h) {
            decomp_h(sq_valid, pv_out_sb_w, Dt, dest_limit, pv_in0_sb, pv_sb_h);
        }

        // R4 causal block-skip: mirror the reader — process only KV chunks not fully
        // in the future of this Q chunk (skv_off < sq_off + sq_valid). The first
        // processed chunk is always j=0 (in the past or on the diagonal), so the
        // online-softmax `first` init below is unaffected. Same n_kv_active formula
        // as the reader keeps the cb_k_in/cb_v_in/cb_kv_mask counts matched.
        uint32_t n_kv_active = n_kv_chunks;
        if constexpr (is_causal) {
            const uint32_t kv_limit = sq_off + sq_valid;  // tiles
            n_kv_active = (kv_limit + Skv_chunk_t - 1) / Skv_chunk_t;
            if (n_kv_active > n_kv_chunks) {
                n_kv_active = n_kv_chunks;
            }
        }

        // Phase 1: pre-scale Q (folds attention scale) -> cb_q_scaled (resident).
        {
            SDPA_ZONE("P01_QSCALE");
            ckl::mul<
                cb_q_in,
                cb_scale,
                cb_q_scaled,
                BroadcastDim::Scalar,
                InputLifecycle::Streaming,
                InputLifecycle::HeldBulk,
                OutputLifecycle::Streaming,
                RCFG,
                PRCFG,
                OperandKind::Scalar,
                OperandKind::Scalar>(EltwiseShape::tiles(sq_dt));
        }

        for (uint32_t j = 0; j < n_kv_active; ++j) {
            const bool first = (j == 0);
            // R1b: partial KV chunk. skv_valid is the QKᵀ N extent, PV K extent, and
            // score-block width for this chunk (only the last chunk is < Skv_chunk_t).
            const uint32_t skv_off = j * Skv_chunk_t;
            const uint32_t skv_valid = (Skv_chunk_t < Skv_t - skv_off) ? Skv_chunk_t : (Skv_t - skv_off);
            const uint32_t sq_skv = sq_valid * skv_valid;
            // QKᵀ N subblock COUNT for this chunk (out_subblock_w is the loop-constant
            // qk_out_sb_w). qk_out_sb_w divides skv_valid: it is 1 when a partial chunk
            // exists, else the full-width divisor of Skv_chunk_t (all chunks full).
            const uint32_t qk_in1_sb = skv_valid / qk_out_sb_w;

            // Phase 2: scores = Q_scaled . K^T  (transpose B; cb_q_scaled retained).
            {
                SDPA_ZONE("P02_QKT");
                if constexpr (ablate_pv >= 3) {
                    // Ablation (mode 3): QK^T matmul FMA+pack stubbed — wait cb_q_scaled (retained,
                    // not popped), drain cb_k_in like the real matmul, push a garbage cb_scores.
                    // Downstream mask/rowmax/exp run on garbage (perf-representative). Bounds total
                    // matmul headroom together with the PV+accum stub (modes >=1, >=2).
                    cb_wait_front(cb_q_scaled, sq_dt);
                    cb_wait_front(cb_k_in, skv_valid * Dt);
                    cb_pop_front(cb_k_in, skv_valid * Dt);
                    cb_reserve_back(cb_scores, sq_skv);
                    cb_push_back(cb_scores, sq_skv);
                } else {
                    ckl::matmul_block<
                        /*transpose=*/true,
                        /*packer_l1_acc=*/false,
                        LastBlockTarget::Out,
                        OutputCBLayout::SubblockMajor,
                        ckl::matmul_config::InitMode::Short,
                        ckl::InputPolicy::WaitAndRetainOnLastBlock,
                        ckl::InputPolicy::WaitAndPopPerKBlock>(
                        q_scaled_buf,
                        k_buf,
                        scores_buf,
                        scores_buf,
                        MatmulBlockShape::of(sq_valid, qk_in1_sb, 1, qk_out_sb_w, Dt, 1));
                }
            }  // P02_QKT

            // Phase 3: additive mask (custom mode), in place, before the row-max.
            if constexpr (has_mask) {
                ckl::add<cb_scores, cb_mask_in, cb_scores>(EltwiseShape::tiles(sq_skv));
            }

            // Phase 3b: generated additive mask (on-device), before the row-max — same
            // additive path as the custom mask. R4 causal takes precedence over R1's
            // KV-padding mask (causal subsumes it): on a straddling KV chunk (some key
            // tile-col >= some query tile-row) add the reader-generated triangular −∞
            // bias; fully-past chunks add nothing (matched with the reader's identical
            // predicate, so cb_kv_mask counts stay balanced). Otherwise (non-causal
            // h_non_aligned) apply R1's last-chunk vertical padding mask.
            if constexpr (is_causal) {
                if (skv_off + skv_valid > sq_off) {
                    ckl::add<cb_scores, cb_kv_mask, cb_scores>(EltwiseShape::tiles(sq_skv));
                }
            } else if constexpr (has_kv_pad) {
                if (j == n_kv_chunks - 1) {
                    ckl::add<cb_scores, cb_kv_mask, cb_scores>(EltwiseShape::tiles(sq_skv));
                }
            }

            // Phase 4: chunk row-max -> cb_corr (cb_scores held for the exp below).
            {
                SDPA_ZONE("P04_ROWMAX");
                if constexpr (ablate_softmax != 0) {
                    // Ablation: row-max reduce payload stubbed — wait cb_scores (no pop, as
                    // WaitUpfrontNoPop) and reserve/push a garbage cb_corr (sq_valid). Keeps
                    // CB balance byte-identical; downstream runs on garbage (perf-only).
                    cb_wait_front(cb_scores, sq_skv);
                    cb_reserve_back(cb_corr, sq_valid);
                    cb_push_back(cb_corr, sq_valid);
                } else {
                    ckl::reduce<
                        ckernel::PoolType::MAX,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_scores,
                        cb_scaler,
                        cb_corr,
                        ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, skv_valid, sq_valid));
                }
            }  // P04_ROWMAX

            // Phase 5: update running max m, form correction alpha.
            {
                SDPA_ZONE("P05_MAXUPD");
                if constexpr (ablate_softmax >= 2) {
                    // FULL-STUB (measurement-only): replicate P05's net CB ops, no math. Net:
                    // cb_corr pop+push, cb_row_max pop+push, cb_m_new push+pop. No in-place CB.
                    if (first) {
                        cb_wait_front(cb_corr, sq_valid);
                        cb_reserve_back(cb_row_max, sq_valid);
                        cb_push_back(cb_row_max, sq_valid);
                        cb_pop_front(cb_corr, sq_valid);
                    } else {
                        // binary_sfpu: corr(pop) x row_max(held) -> m_new(push)
                        cb_wait_front(cb_corr, sq_valid);
                        cb_wait_front(cb_row_max, sq_valid);
                        cb_reserve_back(cb_m_new, sq_valid);
                        cb_push_back(cb_m_new, sq_valid);
                        cb_pop_front(cb_corr, sq_valid);
                        // eltwise_chain: row_max(pop) x m_new(held) -> corr(push)
                        cb_wait_front(cb_row_max, sq_valid);
                        cb_wait_front(cb_m_new, sq_valid);
                        cb_reserve_back(cb_corr, sq_valid);
                        cb_push_back(cb_corr, sq_valid);
                        cb_pop_front(cb_row_max, sq_valid);
                        // copy: m_new(pop) -> row_max(push)
                        cb_wait_front(cb_m_new, sq_valid);
                        cb_reserve_back(cb_row_max, sq_valid);
                        cb_push_back(cb_row_max, sq_valid);
                        cb_pop_front(cb_m_new, sq_valid);
                    }
                } else if (first) {
                    ckl::copy<cb_corr, cb_row_max>(EltwiseShape::tiles(sq_valid));  // m = chunk_max
                } else {
                    // m_new = max(chunk_max, m_old) -> cb_m_new (m_old held for alpha).
                    // The held operand is walked with Block indexing (absolute base+i):
                    // Scalar+HeldStream never advances the front (no pop), so it would
                    // re-read tile 0 for every sq_valid>1 iteration.
                    ckl::binary_sfpu<
                        ckl::BinaryMax<>,
                        cb_corr,
                        cb_row_max,
                        cb_m_new,
                        InputLifecycle::Streaming,
                        InputLifecycle::HeldBulk,
                        OutputLifecycle::Streaming,
                        PRCFG,
                        OperandKind::Scalar,
                        OperandKind::Block>(EltwiseShape::tiles(sq_valid));
                    // alpha = exp(m_old - m_new) -> cb_corr (m_old popped, m_new held).
                    ckl::eltwise_chain(
                        EltwiseShape::tiles(sq_valid),
                        ckl::BinaryFpu<
                            cb_row_max,
                            cb_m_new,
                            ckl::BinaryFpuOp::Sub,
                            BroadcastDim::None,
                            InputLifecycle::Streaming,
                            InputLifecycle::HeldBulk,
                            RCFG,
                            Dst::D0,
                            OperandKind::Scalar,
                            OperandKind::Block>{},
                        ckl::Exp<>{},
                        ckl::PackTile<cb_corr, OutputLifecycle::Streaming>{});
                    // m = m_new.
                    ckl::copy<cb_m_new, cb_row_max>(EltwiseShape::tiles(sq_valid));
                }
            }  // P05_MAXUPD

            // Phase 6: P = exp(scores - m) -> cb_exp (Col bcast; scores popped, m held).
            // R3c: fast SFPU exp for the dominant softmax P=exp phase (measured ~54%
            // of per-chunk compute as the exact exp; the fast exp is ~75% cheaper ->
            // flagged shape 9.01->5.80 ms = 1.55x). `exp_approx` is Fast only in the
            // fp32_dest_acc_en=False throughput regime (host-gated); the max-precision
            // regime stays Exact (byte-identical, no regression). The alpha-correction
            // exp (phase 5) stays exact always to protect the running (m, l, O).
            {
                SDPA_ZONE("P06_EXP");
                if constexpr (ablate_softmax != 0) {
                    // Ablation: exp payload stubbed — replicate fused_exp_dual_pack's CB ops
                    // exactly (wait cb_scores + cb_row_max[no-pop], reserve+push cb_exp +
                    // cb_sum_chunk, pop cb_scores) with no sub/exp/pack math. Isolates the
                    // exp SFPU + dual-pack payload from its phase overhead (perf-only path;
                    // requires fuse_rowsum, which holds in the throughput regime this gate
                    // targets).
                    cb_wait_front(cb_scores, sq_skv);
                    cb_wait_front(cb_row_max, sq_valid);
                    cb_reserve_back(cb_exp, sq_skv);
                    cb_reserve_back(cb_sum_chunk, sq_valid);
                    cb_push_back(cb_exp, sq_skv);
                    cb_push_back(cb_sum_chunk, sq_valid);
                    cb_pop_front(cb_scores, sq_skv);
                } else if constexpr (fuse_rowsum) {
                    // R3e: raw-LLK dual-pack — cb_exp AND the partial row-sum (cb_sum_chunk)
                    // from ONE exp DEST window (fast exp; scores popped, m held). Replaces
                    // the phase-7 per-chunk reduce<SUM> below.
                    fused_exp_dual_pack(sq_valid, skv_valid);
                } else {
                    ckl::eltwise_chain(
                        EltwiseShape::grid(sq_valid, skv_valid),
                        ckl::BinaryFpu<
                            cb_scores,
                            cb_row_max,
                            ckl::BinaryFpuOp::Sub,
                            BroadcastDim::Col,
                            InputLifecycle::Bulk,
                            InputLifecycle::HeldBulk,
                            RCFG,
                            Dst::D0,
                            OperandKind::Block,
                            OperandKind::Col>{},
                        ckl::Exp<exp_approx>{},
                        ckl::PackTile<cb_exp, OutputLifecycle::Bulk>{});
                }
            }  // P06_EXP

            // Phase 7: chunk row-sum + running sum l update.
            {
                SDPA_ZONE("P07_ROWSUM");
                if constexpr (ablate_softmax >= 2) {
                    // FULL-STUB (measurement-only, throughput/fuse_rowsum regime): replicate P07's
                    // net CB ops, no math. cb_row_sum is written IN-PLACE (mul, add) so pop before
                    // reserve. cb_corr (alpha) is HELD here — P08's rescale pops it.
                    if (first) {
                        cb_wait_front(cb_sum_chunk, sq_valid);
                        cb_reserve_back(cb_row_sum, sq_valid);
                        cb_push_back(cb_row_sum, sq_valid);
                        cb_pop_front(cb_sum_chunk, sq_valid);
                    } else {
                        // mul: row_sum *= alpha (in-place; corr held)
                        cb_wait_front(cb_corr, sq_valid);
                        cb_wait_front(cb_row_sum, sq_valid);
                        cb_pop_front(cb_row_sum, sq_valid);
                        cb_reserve_back(cb_row_sum, sq_valid);
                        cb_push_back(cb_row_sum, sq_valid);
                        // add: row_sum += chunk_sum (in-place)
                        cb_wait_front(cb_row_sum, sq_valid);
                        cb_wait_front(cb_sum_chunk, sq_valid);
                        cb_pop_front(cb_row_sum, sq_valid);
                        cb_pop_front(cb_sum_chunk, sq_valid);
                        cb_reserve_back(cb_row_sum, sq_valid);
                        cb_push_back(cb_row_sum, sq_valid);
                    }
                } else if constexpr (fuse_rowsum) {
                    // R3e: the partial (rows x 1, 32-col, un-reduced) row-sum for this chunk
                    // is already in cb_sum_chunk (from the dual-pack above). Keep the running
                    // sum l in the SAME partial form across the KV loop: alpha-rescale +
                    // accumulate here, collapse to a scalar ONCE after the loop (phase 11).
                    if (first) {
                        // l_partial = chunk_partial_sum (bf16 -> fp32).
                        ckl::copy<cb_sum_chunk, cb_row_sum>(EltwiseShape::tiles(sq_valid));
                    } else {
                        // l_partial = alpha * l_partial (alpha col-vector bcast across the 32
                        // cols; alpha held for phase 8's O rescale).
                        ckl::mul<
                            cb_row_sum,
                            cb_corr,
                            cb_row_sum,
                            BroadcastDim::Col,
                            InputLifecycle::Streaming,
                            InputLifecycle::HeldBulk,
                            OutputLifecycle::Streaming,
                            RCFG,
                            PRCFG,
                            OperandKind::Scalar,
                            OperandKind::Col>(EltwiseShape::grid(sq_valid, 1));
                        // l_partial += chunk_partial_sum.
                        ckl::add<cb_row_sum, cb_sum_chunk, cb_row_sum>(EltwiseShape::tiles(sq_valid));
                    }
                } else if (first) {
                    // l = chunk_sum (cb_exp held for PV).
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_exp,
                        cb_scaler,
                        cb_row_sum,
                        ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, skv_valid, sq_valid));
                } else {
                    // l = alpha * l (alpha held for phase 8; Block indexing walks the
                    // held alpha tiles without popping — see phase-5 note).
                    ckl::mul<
                        cb_row_sum,
                        cb_corr,
                        cb_row_sum,
                        BroadcastDim::None,
                        InputLifecycle::Streaming,
                        InputLifecycle::HeldBulk,
                        OutputLifecycle::Streaming,
                        RCFG,
                        PRCFG,
                        OperandKind::Scalar,
                        OperandKind::Block>(EltwiseShape::tiles(sq_valid));
                    // chunk_sum -> cb_sum_chunk (cb_exp held for PV).
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_exp,
                        cb_scaler,
                        cb_sum_chunk,
                        ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, skv_valid, sq_valid));
                    // l += chunk_sum.
                    ckl::add<cb_row_sum, cb_sum_chunk, cb_row_sum>(EltwiseShape::tiles(sq_valid));
                }
            }  // P07_ROWSUM

            // Phase 8: rescale running output O by alpha (Col bcast; pops alpha).
            // In the fused-O-accumulate regime this in-place rescale is UNCHANGED, but it now
            // also positions cb_out_accum for phase 9's L1-accumulate: the in-place mul pops the
            // running O and pushes alpha*O, so (cb_out_accum being a full ring in this regime) the
            // packer write pointer wraps back onto the just-written alpha*O — exactly where the
            // phase-9 PV matmul L1-accumulates P*V. (ablate_pv is measurement-only and meaningful
            // only on the non-fused path; the fused path always runs the real rescale.)
            {
                SDPA_ZONE("P08_ORESCALE");
                if (!first) {
                    if constexpr (!fuse_oaccum && ablate_pv >= 2) {
                        // Ablation: rescale stubbed — just drain alpha (kept CB balance).
                        cb_wait_front(cb_corr, sq_valid);
                        cb_pop_front(cb_corr, sq_valid);
                    } else {
                        ckl::mul<
                            cb_out_accum,
                            cb_corr,
                            cb_out_accum,
                            BroadcastDim::Col,
                            InputLifecycle::Streaming,
                            InputLifecycle::Bulk,
                            OutputLifecycle::Streaming,
                            RCFG,
                            PRCFG,
                            OperandKind::Scalar,
                            OperandKind::Col>(EltwiseShape::grid(sq_valid, Dt));
                    }
                }
            }  // P08_ORESCALE

            // Phase 9: PV = P . V (cb_exp + cb_v popped).
            //
            // R6 FUSED regime (fuse_oaccum): the PV matmul packs DIRECTLY onto the running O in
            // cb_out_accum via packer L1-accumulation (caller_owns_pack_target + TileRowMajor +
            // Interm + packer_l1_acc + the new accumulate_output toggle), so the former phase 10
            // (O += PV) disappears and cb_pv is not needed.
            //   * First KV chunk: accumulate_output=false => block 0 SEEDS (l1_acc=0), O = P*V.
            //     No phase-8 rescale ran, so cb_out_accum is empty: reserve it, pack the seed, push.
            //   * Non-first: accumulate_output=true => block 0 L1-accumulates (l1_acc=1) onto the
            //     alpha*O phase 8 left resident. cb_out_accum is CALLER-OWNED (the matmul does NO
            //     reserve/push/wait/pop on it), so we own its sync: pop the alpha*O phase 8 pushed
            //     (undoing that push — pop only advances the read pointer, the tiles stay resident
            //     in L1), reserve the block (full ring => the write pointer lands back exactly on
            //     that resident alpha*O), let the matmul L1-accumulate P*V onto it, then push the
            //     new O. The pop+reserve keeps the pack inside a reserved region (CB-sanitizer
            //     clean) while the L1-accumulate is genuinely in place.
            // accumulate_output is a compile-time template arg, hence the runtime first/non-first
            // split into two instantiations. pack_reconfig_l1_acc(0) after restores the packer to
            // overwrite mode for the next chunk's QKᵀ pack / the post-loop normalize.
            //
            // NON-FUSED regime (fp32-DEST max-precision, or rare prime-Sq_t partial-q throughput):
            // the original PV -> cb_pv matmul + phase-10 accumulate, byte-identical.
            {
                SDPA_ZONE("P09_PV");
                if constexpr (fuse_oaccum) {
                    const auto pv_shape =
                        MatmulBlockShape::of(pv_in0_sb, pv_in1_sb, pv_sb_h, pv_out_sb_w, skv_valid, 1);
                    if (first) {
                        cb_reserve_back(cb_out_accum, sq_dt);
                        ckl::matmul_block<
                            /*transpose=*/false,
                            /*packer_l1_acc=*/true,
                            LastBlockTarget::Interm,
                            OutputCBLayout::TileRowMajor,
                            ckl::matmul_config::InitMode::Short,
                            ckl::InputPolicy::WaitAndPopPerKBlock,
                            ckl::InputPolicy::WaitAndPopPerKBlock,
                            ckl::NoPostCompute,
                            ckl::NoPreKBlock,
                            ckl::NoPostKBlock,
                            /*untilize_block_ct_dim=*/0,
                            ckl::NoKBlockInnerDimFn,
                            ckl::NoIn0Source,
                            ckl::NoIn1BaseOffset,
                            /*caller_owns_pack_target=*/true,
                            /*accumulate_output=*/false,
                            ckl::NoneActivation,
                            ckl::matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT>(
                            exp_buf,
                            v_buf,
                            out_accum_buf,
                            out_accum_buf,
                            pv_shape,
                            {},
                            {},
                            /*in1_per_core_w=*/0,
                            /*out_row_width=*/Dt);
                        cb_push_back(cb_out_accum, sq_dt);
                    } else {
                        cb_wait_front(cb_out_accum, sq_dt);    // alpha*O phase 8 pushed
                        cb_pop_front(cb_out_accum, sq_dt);     // undo that push (alpha*O stays in L1)
                        cb_reserve_back(cb_out_accum, sq_dt);  // full ring => wr_ptr back onto alpha*O
                        ckl::matmul_block<
                            /*transpose=*/false,
                            /*packer_l1_acc=*/true,
                            LastBlockTarget::Interm,
                            OutputCBLayout::TileRowMajor,
                            ckl::matmul_config::InitMode::Short,
                            ckl::InputPolicy::WaitAndPopPerKBlock,
                            ckl::InputPolicy::WaitAndPopPerKBlock,
                            ckl::NoPostCompute,
                            ckl::NoPreKBlock,
                            ckl::NoPostKBlock,
                            /*untilize_block_ct_dim=*/0,
                            ckl::NoKBlockInnerDimFn,
                            ckl::NoIn0Source,
                            ckl::NoIn1BaseOffset,
                            /*caller_owns_pack_target=*/true,
                            /*accumulate_output=*/true,
                            ckl::NoneActivation,
                            ckl::matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT>(
                            exp_buf,
                            v_buf,
                            out_accum_buf,
                            out_accum_buf,
                            pv_shape,
                            {},
                            {},
                            /*in1_per_core_w=*/0,
                            /*out_row_width=*/Dt);
                        cb_push_back(cb_out_accum, sq_dt);  // publish the updated running O
                    }
                    ckernel::pack_reconfig_l1_acc(0);  // restore overwrite mode for downstream packs.
                } else if constexpr (ablate_pv >= 1) {
                    // Ablation: PV matmul FMA+pack stubbed — drain the operands exactly like
                    // the real matmul (cb_exp sq_skv, cb_v skv_valid*Dt) and push a garbage
                    // cb_pv (sq_dt) so downstream CB balance is byte-identical.
                    cb_wait_front(cb_exp, sq_skv);
                    cb_pop_front(cb_exp, sq_skv);
                    cb_wait_front(cb_v_in, skv_valid * Dt);
                    cb_pop_front(cb_v_in, skv_valid * Dt);
                    cb_reserve_back(cb_pv, sq_dt);
                    cb_push_back(cb_pv, sq_dt);
                } else {
                    ckl::matmul_block<
                        /*transpose=*/false,
                        /*packer_l1_acc=*/false,
                        LastBlockTarget::Out,
                        OutputCBLayout::SubblockMajor,
                        ckl::matmul_config::InitMode::Short,
                        ckl::InputPolicy::WaitAndPopPerKBlock,
                        ckl::InputPolicy::WaitAndPopPerKBlock>(
                        exp_buf,
                        v_buf,
                        pv_buf,
                        pv_buf,
                        MatmulBlockShape::of(pv_in0_sb, pv_in1_sb, pv_sb_h, pv_out_sb_w, skv_valid, 1));
                }
            }  // P09_PV

            // Phase 10: accumulate O += PV. Only in the NON-FUSED regime — the fused PV matmul
            // above already L1-accumulated P*V onto cb_out_accum in place.
            if constexpr (!fuse_oaccum) {
                {
                    SDPA_ZONE("P10_OACCUM");
                    if (first) {
                        ckl::copy<cb_pv, cb_out_accum>(EltwiseShape::tiles(sq_dt));  // O = PV
                    } else if constexpr (ablate_pv >= 2) {
                        // Ablation: accumulate stubbed — drain cb_pv; cb_out_accum stays resident.
                        cb_wait_front(cb_pv, sq_dt);
                        cb_pop_front(cb_pv, sq_dt);
                    } else {
                        ckl::add<cb_out_accum, cb_pv, cb_out_accum>(EltwiseShape::tiles(sq_dt));
                    }
                }  // P10_OACCUM
            }
        }

        // Phase 11: normalize O = O * (1/l), pack to bf16 -> cb_out.
        {
            SDPA_ZONE("P11_NORM");
            if constexpr (fuse_rowsum) {
                // R3e: collapse the partial (32-col) running sum to the scalar denominator
                // l ONCE per q-chunk — a single reduce<SUM,REDUCE_ROW> (the FPU
                // matmul-with-ones), amortized over the whole KV loop instead of one
                // reduce per chunk. -> cb_corr (free after the loop; pops cb_row_sum).
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_row_sum,
                    cb_scaler,
                    cb_corr,
                    ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(1, 1, sq_valid));
                ckl::unary<ckl::Recip<>, cb_corr, cb_corr>(EltwiseShape::tiles(sq_valid));
                ckl::mul<
                    cb_out_accum,
                    cb_corr,
                    cb_out,
                    BroadcastDim::Col,
                    InputLifecycle::Streaming,
                    InputLifecycle::Bulk,
                    OutputLifecycle::Streaming,
                    RCFG,
                    PRCFG,
                    OperandKind::Scalar,
                    OperandKind::Col>(EltwiseShape::grid(sq_valid, Dt));
            } else {
                ckl::unary<ckl::Recip<>, cb_row_sum, cb_row_sum>(EltwiseShape::tiles(sq_valid));
                ckl::mul<
                    cb_out_accum,
                    cb_row_sum,
                    cb_out,
                    BroadcastDim::Col,
                    InputLifecycle::Streaming,
                    InputLifecycle::Bulk,
                    OutputLifecycle::Streaming,
                    RCFG,
                    PRCFG,
                    OperandKind::Scalar,
                    OperandKind::Col>(EltwiseShape::grid(sq_valid, Dt));
            }
        }  // P11_NORM

        // Release the retained Q-scaled block and the running max for the next q-chunk.
        cb_pop_front(cb_q_scaled, sq_dt);
        cb_pop_front(cb_row_max, sq_valid);
    }
}
