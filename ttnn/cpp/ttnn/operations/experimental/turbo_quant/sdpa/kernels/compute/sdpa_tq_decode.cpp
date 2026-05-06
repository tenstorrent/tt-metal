// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TurboQuant SDPA decode compute kernel — interleaved version.
//
// Dequantizes K/V from BFP4 indices+norms to BF16 one chunk at a time,
// interleaved with the SDPA attention math. This avoids needing CBs large
// enough to hold the entire dequantized cache.
//
// Per K/V chunk:
//   1. Dequantize K chunk  (cb_k_idx, cb_k_norms → cb_k_in)
//   2. Q × K^T  matmul + online softmax update
//   3. Dequantize V chunk  (cb_v_idx, cb_v_norms → cb_v_in)
//   4. Attn × V  matmul + output accumulation (ping-pong)
//
// Decode simplifications:
//   - Sq_chunk_t = 1 (single query token)
//   - q_num_chunks = 1
//   - is_causal = false (attend to all prior tokens)
//   - No mask, no attention sink, no sliding window

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define EXP_APPROX_MODE 1

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/bcast.h"
#include "tools/profiler/kernel_profiler.hpp"

// K-split FP32 merge-boundary DPRINTs (2026-05-02). Gated so they only
// activate when TQ_DPRINT_KSPLIT is set in the kernel defines (program
// factory passes this through). Pattern: dump 4 FP32 values from the
// first row of the target tile at the worker pack-out, the reducer
// wait_front, and after each max_block in the merge loop.
#ifdef TQ_DPRINT_KSPLIT
#include "api/debug/dprint.h"
#include "api/debug/dprint_tile.h"
#endif

// Include existing SDPA compute building blocks (matmul_blocks, softmax helpers, etc.)
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

// Empirical fix for multi-chunk online softmax: a non-inlined UNPACK-thread volatile L1 read
// of the given CB's first tile. Without this call at end-of-iteration, chunks 1+ produce
// wrong softmax state (cos≈0.1-0.3 vs expected 0.999+). The pack→unpack path needs this
// to flush cb_max/cb_sum/cb_out_im pack writes from the current iteration before the
// unpacker reads them as prev_max/prev_sum/prev_out in the next iteration.
// Reading only 8 words fails, 64 works — threshold seems to be a minimum data volume.
// Matches the sync TSLICE(cb, range::hw0_32_4) provides inside DPRINT_UNPACK.
__attribute__((noinline)) inline void sync_unpack_cb_read(uint32_t cb_id) {
    volatile tt_l1_ptr uint32_t* ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(cb_id).fifo_rd_ptr << cb_addr_shift);
    for (uint32_t i = 0; i < 64; ++i) {
        [[maybe_unused]] volatile uint32_t dummy = ptr[i];
    }
}

// ── Inline dequantize: BFP4 index tile → BF16 centroid×norm tile ──
// Reads 1 tile from cb_idx, 1 tile from cb_norms (already waiting).
// Writes 1 dequantized tile to cb_out via cb_temp for bcast multiply.
template <uint32_t NumLevels>
inline void dequantize_one_tile(
    uint32_t cb_idx,
    uint32_t cb_norms,
    uint32_t cb_temp,
    uint32_t cb_out,
    const float* centroids,
    const uint32_t* level_bits) {
    // Phase 1: Centroid gather → cb_temp
    cb_wait_front(cb_idx, 1);
    cb_reserve_back(cb_temp, 1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(cb_idx);  // configure unpacker for BFP4 cb
    copy_tile(cb_idx, 0, 0);  // DST[0] = indices (BFP4 auto-unpack to f32)
    fill_tile_init();
    fill_tile(1, centroids[0]);

    for (uint32_t lev = 1; lev < NumLevels; lev++) {
        copy_tile(cb_idx, 0, 2);
        unary_ge_tile_init();
        unary_ge_tile(2, level_bits[lev]);
        fill_tile_init();
        fill_tile(3, centroids[lev]);
        sub_binary_tile_init();
        sub_binary_tile(3, 1, 3);
        mul_binary_tile_init();
        mul_binary_tile(2, 3, 3);
        add_binary_tile_init();
        add_binary_tile(1, 3, 1);
    }

    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_temp);  // ensure packer matches temp CB format
    pack_tile(1, cb_temp);
    tile_regs_release();
    cb_pop_front(cb_idx, 1);
    cb_push_back(cb_temp, 1);

    // Phase 2: Multiply by norm (broadcast column 0 across all columns)
    cb_wait_front(cb_temp, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    mul_bcast_cols_init_short(cb_temp, cb_norms);
    mul_tiles_bcast_cols(cb_temp, cb_norms, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);  // ensure packer matches output CB format
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_pop_front(cb_temp, 1);
    cb_push_back(cb_out, 1);
}

// Dequantize a full chunk (Sk_chunk_t rows × Wt tiles per row).
// Norms: 1 tile per row (shared across Wt head-dim tiles).
template <uint32_t Sk_chunk_t, uint32_t Wt, uint32_t NumLevels>
inline void dequantize_chunk(
    uint32_t cb_idx,
    uint32_t cb_norms,
    uint32_t cb_temp,
    uint32_t cb_out,
    const float* centroids,
    const uint32_t* level_bits) {
    for (uint32_t row = 0; row < Sk_chunk_t; row++) {
        cb_wait_front(cb_norms, 1);
        for (uint32_t col = 0; col < Wt; col++) {
            dequantize_one_tile<NumLevels>(cb_idx, cb_norms, cb_temp, cb_out, centroids, level_bits);
        }
        cb_pop_front(cb_norms, 1);
    }
}

// ──────────────────────────────────────────────────────────────────────
// K / V chunk dequant — matches original pre-fill layout for matmul compat.
//
// K:  two-pass (typecast → cb_dq_temp, gather+norm → cb_out) with transpose
//     of both tile CONTENT (pack_tile<true>) and tile GRID (index swap to
//     col*Sk + row). Produces cb_out in transposed layout [DHt × Sk] which
//     is what sdpa's matmul_blocks(..., transpose=true) expects for Q×K^T.
//
// V:  two-pass but without the output transpose, keeping natural [Sk × vDHt]
//     layout needed for softmax × V matmul (transpose=false).
// ──────────────────────────────────────────────────────────────────────

// DataFormat constants for typecast (BFP4=7, BFP8=4, BF16=5)
constexpr uint32_t TQ_DF_BFP4 = 7;
constexpr uint32_t TQ_DF_BFP8 = 4;
constexpr uint32_t TQ_DF_BF16 = 5;

// Typecast Sk_chunk_t tiles of BFP8 norms from cb_in to BF16 in cb_out. The
// dequant_*_chunk paths require BF16 norms because mul_tiles_bcast_cols does
// not natively unpack BFP8 input. Cost: 1 copy + 1 typecast + 1 pack per K
// chunk row (Sk_chunk_t tiles total) — negligible vs the dequant body.
template <uint32_t Sk_chunk_t>
inline void typecast_norms_bfp8_to_bf16(uint32_t cb_in, uint32_t cb_out) {
    init_sfpu(cb_in, cb_out);
    cb_wait_front(cb_in, Sk_chunk_t);
    cb_reserve_back(cb_out, Sk_chunk_t);
    for (uint32_t t = 0; t < Sk_chunk_t; ++t) {
        tile_regs_acquire();
        copy_tile(cb_in, t, 0);
        typecast_tile_init<TQ_DF_BFP8, TQ_DF_BF16>();
        typecast_tile<TQ_DF_BFP8, TQ_DF_BF16>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    cb_pop_front(cb_in, Sk_chunk_t);
    cb_push_back(cb_out, Sk_chunk_t);
}

template <uint32_t Sk_chunk_t, uint32_t DHt, uint32_t NumLevels>
inline void dequant_k_chunk(
    uint32_t cb_k_idx,
    uint32_t cb_k_norms,
    uint32_t cb_dq_temp,
    uint32_t cb_k_in,
    const float* centroids,
    const uint32_t* level_bits,
    const uint32_t* delta_bits) {
    constexpr uint32_t chunk_tiles = Sk_chunk_t * DHt;

    // Pass 1: typecast BFP4→BF16 into cb_dq_temp (row-major layout).
    {
        // DeviceZoneScopedN("TQ_K_TYPECAST");
        init_sfpu(cb_k_idx, cb_dq_temp);
        cb_reserve_back(cb_dq_temp, chunk_tiles);
        for (uint32_t t = 0; t < chunk_tiles; t++) {
            tile_regs_acquire();
            cb_wait_front(cb_k_idx, 1);
            copy_tile(cb_k_idx, 0, 0);
            typecast_tile_init<TQ_DF_BFP4, TQ_DF_BF16>();
            typecast_tile<TQ_DF_BFP4, TQ_DF_BF16>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_dq_temp);
            cb_pop_front(cb_k_idx, 1);
            tile_regs_release();
        }
        cb_push_back(cb_dq_temp, chunk_tiles);
    }

    // Pass 2a: Centroid gather, in-place in cb_dq_temp.
    //
    // For each input idx (BF16, integer 0..N-1), produce centroids[idx] using
    // the telescoping cascade
    //   result = c[0] + Σ_{lev=1..N-1} (idx >= lev) * (c[lev] - c[lev-1])
    //
    // The deltas (c[lev] - c[lev-1]) are precomputed once at kernel start and
    // multiplied via mul_unary_tile (replaces fill+mul_binary). We also keep
    // idx in DST 0 across the loop and use copy_dest_values for the per-level
    // refresh into DST 2 (replaces re-reading idx from cb_dq_temp every level).
    // Net per level: 4 SFPU ops (was 6), saving ~33% of the cascade overhead.
    //
    // Tier 3A: hoist all SFPU *_init calls out of the per-tile and per-level
    // loops. The init macros configure the SFPU dispatcher for an op type;
    // once configured, subsequent calls of the same op don't need re-init.
    // This drops ~480 init calls per chunk down to 6 (one per op type used).
    {
        // DeviceZoneScopedN("TQ_K_GATHER");
        mm_init(cb_dq_temp, cb_k_norms, cb_k_in);
        cb_wait_front(cb_dq_temp, chunk_tiles);
        cb_wait_front(cb_k_norms, Sk_chunk_t);
        copy_tile_to_dst_init_short(cb_dq_temp);
        fill_tile_init();
        copy_dest_values_init();
        unary_ge_tile_init();
        binop_with_scalar_tile_init();
        add_binary_tile_init();
        for (uint32_t t = 0; t < chunk_tiles; t++) {
            tile_regs_acquire();
            copy_tile(cb_dq_temp, t, 0);  // DST 0 = idx (preserved across cascade)
            fill_tile(1, centroids[0]);  // DST 1 = result, initialised to c[0]
            for (uint32_t lev = 1; lev < NumLevels; lev++) {
                copy_dest_values<DataFormat::Float32>(0, 2);    // DST 2 = idx (FP32 dst mode)
                unary_ge_tile(2, level_bits[lev]);              // DST 2 = (idx >= lev)
                mul_unary_tile(2, delta_bits[lev]);             // DST 2 *= (c[lev] - c[lev-1])
                add_binary_tile(1, 2, 1);  // DST 1 += contribution
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_dq_temp);
            pack_tile<true>(1, cb_dq_temp, t);
            tile_regs_release();
        }
    }

    // Pass 2b: Norm bcast multiply + tile-grid transpose into cb_k_in.
    // Tier 3A: hoist mul_bcast_cols_init_short outside the per-tile loops.
    {
        // DeviceZoneScopedN("TQ_K_NORM_TRANSPOSE");
        cb_reserve_back(cb_k_in, chunk_tiles);
        mul_bcast_cols_init_short(cb_dq_temp, cb_k_norms);
        for (uint32_t row = 0; row < Sk_chunk_t; row++) {
            for (uint32_t col = 0; col < DHt; col++) {
                uint32_t src_tile = row * DHt + col;
                tile_regs_acquire();
                mul_tiles_bcast_cols(cb_dq_temp, cb_k_norms, src_tile, row, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_k_in);
                pack_tile<true>(0, cb_k_in, col * Sk_chunk_t + row);
                tile_regs_release();
            }
        }
        cb_pop_front(cb_dq_temp, chunk_tiles);
        cb_pop_front(cb_k_norms, Sk_chunk_t);
        cb_push_back(cb_k_in, chunk_tiles);
    }
}

template <uint32_t Sk_chunk_t, uint32_t vDHt, uint32_t NumLevels>
inline void dequant_v_chunk(
    uint32_t cb_v_idx,
    uint32_t cb_v_norms,
    uint32_t cb_dq_temp,
    uint32_t cb_v_in,
    const float* centroids,
    const uint32_t* level_bits,
    const uint32_t* delta_bits) {
    constexpr uint32_t chunk_tiles = Sk_chunk_t * vDHt;

    // Pass 1: Typecast BFP4→BF16 into cb_dq_temp.
    {
        // DeviceZoneScopedN("TQ_V_TYPECAST");
        init_sfpu(cb_v_idx, cb_dq_temp);
        cb_reserve_back(cb_dq_temp, chunk_tiles);
        for (uint32_t t = 0; t < chunk_tiles; t++) {
            tile_regs_acquire();
            cb_wait_front(cb_v_idx, 1);
            copy_tile(cb_v_idx, 0, 0);
            typecast_tile_init<TQ_DF_BFP4, TQ_DF_BF16>();
            typecast_tile<TQ_DF_BFP4, TQ_DF_BF16>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_dq_temp);
            cb_pop_front(cb_v_idx, 1);
            tile_regs_release();
        }
        cb_push_back(cb_dq_temp, chunk_tiles);
    }

    // Pass 2a: Centroid gather, in-place in cb_dq_temp. (See dequant_k_chunk
    // pass 2a for the rationale — same telescoping cascade with precomputed
    // deltas + copy_dest_values + mul_unary_tile, ~33% fewer SFPU ops/level.)
    // Tier 3A: hoist all SFPU *_init calls out of the per-tile/per-level loops.
    {
        // DeviceZoneScopedN("TQ_V_GATHER");
        mm_init(cb_dq_temp, cb_v_norms, cb_v_in);
        cb_wait_front(cb_dq_temp, chunk_tiles);
        cb_wait_front(cb_v_norms, Sk_chunk_t);
        copy_tile_to_dst_init_short(cb_dq_temp);
        fill_tile_init();
        copy_dest_values_init();
        unary_ge_tile_init();
        binop_with_scalar_tile_init();
        add_binary_tile_init();
        for (uint32_t t = 0; t < chunk_tiles; t++) {
            tile_regs_acquire();
            copy_tile(cb_dq_temp, t, 0);  // DST 0 = idx (preserved across cascade)
            fill_tile(1, centroids[0]);  // DST 1 = result
            for (uint32_t lev = 1; lev < NumLevels; lev++) {
                copy_dest_values<DataFormat::Float32>(0, 2);  // FP32 dst mode
                unary_ge_tile(2, level_bits[lev]);
                mul_unary_tile(2, delta_bits[lev]);
                add_binary_tile(1, 2, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_dq_temp);
            pack_tile<true>(1, cb_dq_temp, t);
            tile_regs_release();
        }
    }

    // Pass 2b: Norm bcast multiply into cb_v_in (no output transpose — natural [Sk × vDHt]).
    // Sequential pack matches the row-major src layout.
    // Tier 3A: hoist mul_bcast_cols_init_short outside the per-tile loops.
    {
        // DeviceZoneScopedN("TQ_V_NORM");
        cb_reserve_back(cb_v_in, chunk_tiles);
        mul_bcast_cols_init_short(cb_dq_temp, cb_v_norms);
        for (uint32_t row = 0; row < Sk_chunk_t; row++) {
            for (uint32_t col = 0; col < vDHt; col++) {
                uint32_t src_tile = row * vDHt + col;
                tile_regs_acquire();
                mul_tiles_bcast_cols(cb_dq_temp, cb_v_norms, src_tile, row, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_v_in);
                pack_tile(0, cb_v_in);
                tile_regs_release();
            }
        }
        cb_pop_front(cb_dq_temp, chunk_tiles);
        cb_pop_front(cb_v_norms, Sk_chunk_t);
        cb_push_back(cb_v_in, chunk_tiles);
    }
}

// Template helper to load centroid compile-time args (must be at namespace scope).
template <uint32_t Idx, uint32_t N, uint32_t Base>
struct LoadSDPACentroids {
    static void apply(float* c, uint32_t* lb) {
        constexpr uint32_t bits = get_compile_time_arg_val(Base + 1 + Idx);
        union {
            uint32_t u;
            float f;
        } conv;
        conv.u = bits;
        c[Idx] = conv.f;
        float fi = static_cast<float>(Idx);
        __builtin_memcpy(&lb[Idx], &fi, sizeof(uint32_t));
        LoadSDPACentroids<Idx + 1, N, Base>::apply(c, lb);
    }
};
template <uint32_t N, uint32_t Base>
struct LoadSDPACentroids<N, N, Base> {
    static void apply(float*, uint32_t*) {}
};

void kernel_main() {
    // ── Compile-time args: standard SDPA + TQ centroids ──
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t vDHt = get_compile_time_arg_val(5);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(7);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(10);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(12);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(17);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(18);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(21);
    constexpr uint32_t num_cores = get_compile_time_arg_val(22);

    constexpr bool is_causal = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(27);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(28);

    // TQ compile-time args
    constexpr uint32_t TQ_BASE = 33;
    constexpr uint32_t num_levels = get_compile_time_arg_val(TQ_BASE);
    // pre_rescaled=1: BFP4 values are already centroid×norm, skip gather+norm
    constexpr bool pre_rescaled = get_compile_time_arg_val(TQ_BASE + 1 + num_levels) == 1;
    // norms_are_bfp8=1: norms cache is BFP8_B (must typecast to BF16 before bcast_cols multiply)
    constexpr bool norms_are_bfp8 = get_compile_time_arg_val(TQ_BASE + 2 + num_levels) == 1;
    // return_lse=1: also pack LSE = max + log(sum) to cb_lse_out (c_3) before
    // the final divide, so a host-side combine can merge two SDPA outputs via
    // online softmax. See turbo_quant/LSE_COMBINE_DESIGN.md. Incompatible with
    // cores_per_head_arg > 1 (no combine of LSE across the Tier 2A reducer).
    constexpr bool return_lse = get_compile_time_arg_val(TQ_BASE + 3 + num_levels) == 1;

    // Runtime args
    const uint32_t local_batch_start = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(2);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(4);
    const uint32_t local_q_start = get_arg_val<uint32_t>(5);
    const uint32_t local_q_end = get_arg_val<uint32_t>(6);
    // (Tier 2A) Per-core chunk-slice routing. The program factory currently sends
    // (0, 1) per core, so the kernel processes the full chunk range exactly as
    // before. Worker/reducer split lights up when cores_per_head_arg > 1.
    const uint32_t core_idx_in_group_arg = get_arg_val<uint32_t>(7);
    const uint32_t cores_per_head_arg = get_arg_val<uint32_t>(8);
    // (Tier 2A Phase 2.3) Per-program semaphore for worker→reducer signal. The
    // worker `noc_semaphore_inc`s this when its partial state is packed to the
    // reducer's L1; the reducer `noc_semaphore_wait`s for K-1 increments before
    // pulling and merging. Currently unused (cores_per_head_arg == 1 means the
    // kernel runs the legacy single-core path with no cross-core sync needed).
    [[maybe_unused]] const uint32_t reducer_semaphore_id = get_arg_val<uint32_t>(9);
    const bool is_reducer = (core_idx_in_group_arg == 0);
    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    // Load centroid constants from compile-time args (template-unrolled,
    // defined at namespace scope above kernel_main)
    float centroids[16];
    uint32_t level_bits_arr[16];
    LoadSDPACentroids<0, num_levels, TQ_BASE>::apply(centroids, level_bits_arr);

    // Precompute centroid deltas (c[lev] - c[lev-1]) as FP32 bit patterns —
    // used by the gather cascade's mul_unary_tile to skip the per-level
    // fill_tile + sub_binary_tile pair. Computed once per kernel run.
    uint32_t delta_bits_arr[16];
    delta_bits_arr[0] = 0;  // unused
    for (uint32_t lev = 1; lev < num_levels; ++lev) {
        float delta = centroids[lev] - centroids[lev - 1];
        union {
            float f;
            uint32_t u;
        } conv;
        conv.f = delta;
        delta_bits_arr[lev] = conv.u;
    }

    // ── CB indices ──
    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;

    constexpr uint32_t cb_k_idx = tt::CBIndex::c_10;
    constexpr uint32_t cb_k_norms_in = tt::CBIndex::c_11;  // BFP8 (or BF16) from reader
    constexpr uint32_t cb_v_idx = tt::CBIndex::c_12;
    constexpr uint32_t cb_v_norms_in = tt::CBIndex::c_13;  // BFP8 (or BF16) from reader
    constexpr uint32_t cb_dq_temp = tt::CBIndex::c_14;
    constexpr uint32_t cb_k_norms_bf16 = tt::CBIndex::c_15;  // BFP8→BF16 typecast scratch (K)
    constexpr uint32_t cb_v_norms_bf16 = tt::CBIndex::c_17;  // BFP8→BF16 typecast scratch (V)
    // Tier 2A Phase 2.3 — cross-core partial-state CBs. Workers pack their final
    // (max, sum, out) here; the writer kernel NoC-sends to the reducer's L1.
    constexpr uint32_t cb_partial_max = tt::CBIndex::c_18;
    constexpr uint32_t cb_partial_sum = tt::CBIndex::c_19;
    constexpr uint32_t cb_partial_out = tt::CBIndex::c_20;
    // Reducer-side mirrors — the writer NoC-pushes peer partials into these.
    // Each holds K slots (one per worker including reducer slot 0). Reducer
    // compute discards slot 0 then merges slots 1..K-1 sequentially.
    constexpr uint32_t cb_remote_max = tt::CBIndex::c_21;
    constexpr uint32_t cb_remote_sum = tt::CBIndex::c_22;
    constexpr uint32_t cb_remote_out = tt::CBIndex::c_23;
    // Tier 2A merge scratch: cb_merge_new_max holds max(prev_max, peer_max);
    // cb_merge_peer_diff holds exp((peer_max - new_max) * scale). cb_exp_max_diff
    // (c_31) is reused for self_diff = exp((prev_max - new_max) * scale).
    constexpr uint32_t cb_merge_new_max = tt::CBIndex::c_3;
    constexpr uint32_t cb_merge_peer_diff = tt::CBIndex::c_4;
    constexpr uint32_t cb_cur_pos = tt::CBIndex::c_8;
    // After typecast (when BFP8), the dequant kernels consume the BF16 scratch CB.
    // When norms are already BF16, both aliases point at the same input CB.
    constexpr uint32_t cb_k_norms = norms_are_bfp8 ? cb_k_norms_bf16 : cb_k_norms_in;
    constexpr uint32_t cb_v_norms = norms_are_bfp8 ? cb_v_norms_bf16 : cb_v_norms_in;

    // ── Wait for reader to fill cur_pos CB ──
    // Reader does a single noc_async_read of the cur_pos tensor into cb_cur_pos
    // at kernel start; we use ckernel::read_tile_value (UNPACK→mailbox→MATH/PACK)
    // to read cur_pos[nb] per batch in the nb loop below.
    cb_wait_front(cb_cur_pos, 1);
    constexpr uint32_t k_chunk_size_tokens = Sk_chunk_t * 32;

    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_im_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    mm_init(cb_q_in, cb_k_in, cb_out);

    // DataFormat values: Bfp4_b=7, Float16_b=5
    constexpr uint32_t DF_BFP4 = 7;
    constexpr uint32_t DF_BF16 = 5;

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            // ══════════════════════════════════════════════════════════════
            // Pre-rescaled fast path: cb_k_in/cb_v_in are filled by reader
            // (BFP4 pre-rescaled values). Standard SDPA consumes directly.
            // Used when LSE export is NOT requested.
            //
            // When return_lse=True we fall through to the unified Flash
            // Attention loop below (with `if constexpr (!pre_rescaled)`
            // guards on the dequant steps), because sdpa_standard does not
            // expose internal max/sum and the writer would deadlock waiting
            // for cb_lse_out tiles that compute never pushes.
            // ══════════════════════════════════════════════════════════════
            if constexpr (pre_rescaled && !return_lse) {
                mm_init(cb_q_in, cb_k_in, cb_out);
                sdpa_standard<
                    cb_qk_im,
                    cb_identity_scale_in,
                    (uint32_t)tt::CBIndex::c_4,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    DHt,
                    vDHt,
                    false /*use_attention_sink*/,
                    false /*is_causal*/,
                    false /*use_provided_mask*/,
                    false /*use_padded_mask*/,
                    false /*is_chunked*/,
                    scale_fp32,
                    (uint32_t)0 /*sliding_window*/>(
                    Skt,
                    qk_in0_block_w,
                    qk_subblock_w,
                    qk_subblock_h,
                    qk_in0_num_subblocks,
                    qk_in1_num_subblocks,
                    qk_num_blocks,
                    out_in0_block_w,
                    out_subblock_w,
                    out_subblock_h,
                    out_in0_num_subblocks,
                    out_in1_num_subblocks,
                    out_num_blocks,
                    (uint32_t)0,  // iter_q_start
                    (uint32_t)1,  // iter_q_end
                    (uint32_t)1,  // q_num_chunks
                    (uint32_t)0,  // local_q_start
                    (uint32_t)0,  // chunked_q_chunk_offset
                    k_num_chunks,
                    q_chunk_tiles,
                    k_chunk_tiles,
                    v_chunk_tiles,
                    qk_chunk_tiles,
                    out_chunk_tiles,
                    cb_q_in,
                    cb_k_in,
                    cb_v_in,
                    (uint32_t)tt::CBIndex::c_3 /*cb_mask_in unused*/,
                    cb_col_identity,
                    cb_out_im_A,
                    cb_out_im_B,
                    cb_max_A,
                    cb_max_B,
                    cb_sum_A,
                    cb_sum_B,
                    cb_exp_max_diff,
                    cb_out);
            } else {
                DeviceZoneScopedN("TQ_FULL_DEQUANT_HEAD");
                // ══════════════════════════════════════════════════════════
                // Full dequant path: interleaved dequant + Flash Attention.
                //
                // Per chunk:
                //   1. Dequantize K chunk (BFP4 indices + BF16 norms → BF16 K)
                //   2. Q × K^T matmul → cb_qk_im
                //   3. Online softmax (cur_max update, exp, partial cur_sum)
                //   4. Dequantize V chunk
                //   5. softmax × V → alias_mm2_cur_out
                //   6. Lazy softmax correction (rescale prev_sum/prev_out)
                //   7. Swap prev↔cur aliases
                // After loop: final row reduction + recip + normalize → cb_out.
                //
                // CBs only hold 1 chunk at a time (double-buffered), so this
                // scales to 128K+ context without L1 blow-up.
                // ══════════════════════════════════════════════════════════

                // Softmax state aliases, swapped each chunk (Flash Attention pattern).
                uint32_t alias_prev_sum = cb_sum_A;
                uint32_t alias_cur_sum = cb_sum_B;
                uint32_t alias_prev_max = cb_max_A;
                uint32_t alias_cur_max = cb_max_B;
                uint32_t alias_mm2_prev_out = cb_out_im_A;
                uint32_t alias_mm2_cur_out = cb_out_im_B;

                {
                    DeviceZoneScopedN("TQ_HEAD_INIT");
                    mm_init(cb_q_in, cb_k_in, cb_out);
                }

                // Limit k_chunks to those that actually contain valid (filled) data.
                // Mirrors reader_tq_decode.cpp's bound — must agree to avoid CB deadlock.
                // Hoisted out of TQ_CHUNK_LOOP scope so the post-loop softmax-denom
                // correction (below) can use cur_pos_nb and valid_k_chunks.
                const uint32_t cur_pos_nb = read_tile_value(cb_cur_pos, 0, nb);
                const uint32_t valid_k_chunks_raw = (cur_pos_nb + k_chunk_size_tokens) / k_chunk_size_tokens;
                const uint32_t valid_k_chunks = valid_k_chunks_raw < k_num_chunks ? valid_k_chunks_raw : k_num_chunks;

                // ── Tier 2A scaffolding: per-core chunk slice (Phase 2.1) ──
                // Runtime args (slots 7 / 8) route this core to a slice of the
                // chunk range. Currently the program factory always sends (0, 1)
                // so this core processes every chunk just as before. Phase 2.2
                // will start sending non-trivial values; the cross-core reduce
                // of partial (max, sum, out) state lands in Phase 2.3-2.5.
                // See turbo_quant/TIER_2A_DESIGN.md for the full plan.
                const uint32_t chunks_per_worker = (valid_k_chunks + cores_per_head_arg - 1) / cores_per_head_arg;
                const uint32_t k_chunk_start_for_core = core_idx_in_group_arg * chunks_per_worker;
                const uint32_t k_chunk_end_for_core = (k_chunk_start_for_core + chunks_per_worker < valid_k_chunks)
                                                          ? k_chunk_start_for_core + chunks_per_worker
                                                          : valid_k_chunks;

                {
                    DeviceZoneScopedN("TQ_CHUNK_LOOP");
                    for (uint32_t k_chunk = k_chunk_start_for_core; k_chunk < k_chunk_end_for_core; ++k_chunk) {
                        DeviceZoneScopedN("TQ_K_CHUNK_TOTAL");
                        // ── Step 0: Typecast BFP8 norms → BF16 (when stored as BFP8) ──
                        // Skipped in pre_rescaled mode (no norms — values already include them).
                        if constexpr (!pre_rescaled && norms_are_bfp8) {
                            typecast_norms_bfp8_to_bf16<Sk_chunk_t>(cb_k_norms_in, cb_k_norms_bf16);
                        }

                        // ── Step 1: Dequantize K chunk → cb_k_in (transposed layout) ──
                        // Skipped in pre_rescaled mode (reader fills cb_k_in directly with
                        // pre-rescaled BFP4 values in the transposed layout matmul expects).
                        if constexpr (!pre_rescaled) {
                            // DeviceZoneScopedN("TQ_DEQUANT_K");
                            dequant_k_chunk<Sk_chunk_t, DHt, num_levels>(
                                cb_k_idx, cb_k_norms, cb_dq_temp, cb_k_in, centroids, level_bits_arr, delta_bits_arr);
                        }

                        // ── Step 2: QK = Q × K^T ──
                        {
                            // DeviceZoneScopedN("TQ_QK_MATMUL");
                            reconfig_data_format(cb_k_in, cb_q_in);
                            pack_reconfig_data_format(cb_qk_im);
                            matmul_blocks(
                                cb_q_in,
                                cb_k_in,
                                cb_qk_im,
                                Sq_chunk_t,
                                Sk_chunk_t,
                                DHt,
                                qk_num_blocks,
                                qk_in0_num_subblocks,
                                qk_in1_num_subblocks,
                                qk_in0_block_w,
                                qk_subblock_h,
                                qk_subblock_w,
                                true /*transpose K*/);
                        }

                        // ── Step 3: Online softmax ──
                        {
                            // DeviceZoneScopedN("TQ_SOFTMAX");
                            // cur_max = max(QK, dim=-1) (with eltwise max vs prev on chunk>0)
                            reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                            reduce_c<
                                PoolType::MAX,
                                ReduceDim::REDUCE_ROW,
                                cb_qk_im,
                                cb_identity_scale_in,
                                Sq_chunk_t,
                                Sk_chunk_t>(alias_cur_max, alias_prev_max, k_chunk > k_chunk_start_for_core);

                            // QK = exp((QK - cur_max) * scale); partial reduce_sum into cur_sum
                            sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true>(
                                alias_cur_max, alias_cur_sum, Sk_chunk_t);
                        }

                        // ── Step 4: Dequantize V chunk → cb_v_in (natural layout) ──
                        // Skipped in pre_rescaled mode (reader fills cb_v_in directly).
                        if constexpr (!pre_rescaled && norms_are_bfp8) {
                            typecast_norms_bfp8_to_bf16<Sk_chunk_t>(cb_v_norms_in, cb_v_norms_bf16);
                        }
                        if constexpr (!pre_rescaled) {
                            // DeviceZoneScopedN("TQ_DEQUANT_V");
                            dequant_v_chunk<Sk_chunk_t, vDHt, num_levels>(
                                cb_v_idx, cb_v_norms, cb_dq_temp, cb_v_in, centroids, level_bits_arr, delta_bits_arr);
                        }

                        // ── Step 5: OUT_IM = softmax × V ──
                        {
                            // DeviceZoneScopedN("TQ_OUT_MATMUL");
                            reconfig_data_format(cb_v_in, cb_qk_im);
                            pack_reconfig_data_format(alias_mm2_cur_out);
                            matmul_blocks(
                                cb_qk_im,
                                cb_v_in,
                                alias_mm2_cur_out,
                                Sq_chunk_t,
                                vDHt,
                                Sk_chunk_t,
                                out_num_blocks,
                                out_in0_num_subblocks,
                                out_in1_num_subblocks,
                                out_in0_block_w,
                                out_subblock_h,
                                out_subblock_w,
                                false /*no transpose*/);

                            cb_pop_front(cb_qk_im, qk_chunk_tiles);
                            reconfig_data_format(alias_prev_max, alias_cur_max);
                        }

                        // ── Step 6: Lazy softmax correction (from second chunk in this core's slice onward) ──
                        if (k_chunk > k_chunk_start_for_core) {
                            // DeviceZoneScopedN("TQ_SOFTMAX_CORRECTION");
                            // exp_max_diff = exp((prev_max - cur_max) * scale)
                            sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
                            cb_pop_front(alias_prev_max, Sq_chunk_t);
                            // prev_sum *= exp_max_diff
                            mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
                            // cur_sum += prev_sum
                            add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);
                            // mm2_cur_out += mm2_prev_out * exp_max_diff (via L1 accum)
                            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, true>(
                                alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out);
                        }

                        // Empirical fix: multiple volatile L1 reads via the CB tile pointer on UNPACK.
                        {
                            // DeviceZoneScopedN("TQ_SYNC_UNPACK");
                            UNPACK(sync_unpack_cb_read(alias_cur_max););
                        }

                        // ── Step 7: Swap aliases for next iteration ──
                        uint32_t tmp;
                        tmp = alias_prev_sum;
                        alias_prev_sum = alias_cur_sum;
                        alias_cur_sum = tmp;
                        tmp = alias_prev_max;
                        alias_prev_max = alias_cur_max;
                        alias_cur_max = tmp;
                        tmp = alias_mm2_prev_out;
                        alias_mm2_prev_out = alias_mm2_cur_out;
                        alias_mm2_cur_out = tmp;
                    }  // end k_chunk loop
                }  // end TQ_CHUNK_LOOP zone

                // ── Final: row-reduce partial sum, recip, normalize output ──
                {
                    DeviceZoneScopedN("TQ_FINAL_NORMALIZE");

                    // ── Tier 2A: empty-slice worker fast path ──
                    // When chunks_per_worker doesn't divide valid_k_chunks evenly
                    // (or valid_k_chunks < cores_per_head_arg), one or more workers
                    // see [chunk_start, chunk_end) = [N, N) — an empty slice. The
                    // chunk loop iterated 0 times so alias_prev_max/sum/out CBs are
                    // empty; calling matmul_reduce here would block forever on
                    // cb_wait_front(alias_prev_sum, Sq_chunk_t).
                    //
                    // Solution: detect empty slice on the worker side, skip
                    // matmul_reduce, and push *neutral* tiles to cb_partial_*:
                    //   peer_max = -10000.0  → exp((peer_max - new_max) * scale) ≈ 0
                    //   peer_sum = 0
                    //   peer_out = 0
                    // The writer still NoC-sends these and sema_inc's the reducer,
                    // so the reducer's wait completes and its merge for this peer
                    // becomes a no-op (peer_diff ≈ 0 zeroes out the contribution
                    // regardless of peer_sum/out).
                    //
                    // The reducer (idx == 0) always has slot 0 = chunks
                    // [0, chunks_per_worker) which is non-empty when valid_k_chunks
                    // ≥ 1 (always true), so this branch never fires there.
                    const bool has_local_data = (k_chunk_start_for_core < k_chunk_end_for_core);
                    if (cores_per_head_arg > 1 && !is_reducer && !has_local_data) {
                        // peer_max ← -10000 (large-negative sentinel)
                        cb_reserve_back(cb_partial_max, Sq_chunk_t);
                        for (uint32_t i = 0; i < Sq_chunk_t; ++i) {
                            tile_regs_acquire();
                            fill_tile_init();
                            fill_tile(0, -10000.0f);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_reconfig_data_format(cb_partial_max);
                            pack_tile(0, cb_partial_max);
                            tile_regs_release();
                        }
                        cb_push_back(cb_partial_max, Sq_chunk_t);

                        // peer_sum ← 0
                        cb_reserve_back(cb_partial_sum, Sq_chunk_t);
                        for (uint32_t i = 0; i < Sq_chunk_t; ++i) {
                            tile_regs_acquire();
                            fill_tile_init();
                            fill_tile(0, 0.0f);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_reconfig_data_format(cb_partial_sum);
                            pack_tile(0, cb_partial_sum);
                            tile_regs_release();
                        }
                        cb_push_back(cb_partial_sum, Sq_chunk_t);

                        // peer_out ← 0 (out_chunk_tiles tiles)
                        cb_reserve_back(cb_partial_out, out_chunk_tiles);
                        for (uint32_t i = 0; i < out_chunk_tiles; ++i) {
                            tile_regs_acquire();
                            fill_tile_init();
                            fill_tile(0, 0.0f);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_reconfig_data_format(cb_partial_out);
                            pack_tile(0, cb_partial_out);
                            tile_regs_release();
                        }
                        cb_push_back(cb_partial_out, out_chunk_tiles);

                        cb_pop_front(cb_q_in, q_chunk_tiles);
                        continue;  // empty slice — done with this (nb, nq)
                    }

                    matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);

                    // ── Tier 2A Phase 2.3 step 2c: worker pack-and-skip path ──
                    // When this core is a worker (idx > 0 in a K > 1 group), it must NOT
                    // finalize the output. Instead pack the local (max, sum, out) into
                    // cb_partial_max/sum/out — the writer kernel NoC-sends these to the
                    // reducer's L1 (Phase 2.3 step 3) and the reducer's compute kernel
                    // merges them via online-softmax correction (Phase 2.3 step 4).
                    //
                    // Skipped at K=1 since `cores_per_head_arg > 1` is false.
                    if (cores_per_head_arg > 1 && !is_reducer) {
                        cb_wait_front(alias_prev_max, Sq_chunk_t);
                        cb_wait_front(alias_prev_sum, Sq_chunk_t);
                        cb_wait_front(alias_mm2_prev_out, out_chunk_tiles);

                        // alias_prev_max → cb_partial_max
                        cb_reserve_back(cb_partial_max, Sq_chunk_t);
                        for (uint32_t i = 0; i < Sq_chunk_t; ++i) {
                            tile_regs_acquire();
                            copy_tile_to_dst_init_short(alias_prev_max);
                            copy_tile(alias_prev_max, i, 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_reconfig_data_format(cb_partial_max);
                            pack_tile(0, cb_partial_max);
                            tile_regs_release();
                        }
#ifdef TQ_DPRINT_KSPLIT
                        // Worker side: dump just-packed cb_partial_max contents so we
                        // can confirm the FP32 bytes leaving the worker are finite.
                        // On TRISC, TileSlice takes only (cb, tile_idx, slice, endl_rows, untilize).
                        DPRINT_PACK(
                            DPRINT << "[TQ_DBG] WRK nq=" << (uint32_t)nq << " idx=" << core_idx_in_group_arg << " PMAX="
                                   << TileSlice(
                                          cb_partial_max,
                                          0,
                                          SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1},
                                          true,
                                          true)
                                   << ENDL());
#endif
                        cb_push_back(cb_partial_max, Sq_chunk_t);

                        // alias_prev_sum → cb_partial_sum
                        cb_reserve_back(cb_partial_sum, Sq_chunk_t);
                        for (uint32_t i = 0; i < Sq_chunk_t; ++i) {
                            tile_regs_acquire();
                            copy_tile_to_dst_init_short(alias_prev_sum);
                            copy_tile(alias_prev_sum, i, 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_reconfig_data_format(cb_partial_sum);
                            pack_tile(0, cb_partial_sum);
                            tile_regs_release();
                        }
                        cb_push_back(cb_partial_sum, Sq_chunk_t);

                        // alias_mm2_prev_out → cb_partial_out (out_chunk_tiles tiles)
                        cb_reserve_back(cb_partial_out, out_chunk_tiles);
                        for (uint32_t i = 0; i < out_chunk_tiles; ++i) {
                            tile_regs_acquire();
                            copy_tile_to_dst_init_short(alias_mm2_prev_out);
                            copy_tile(alias_mm2_prev_out, i, 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_reconfig_data_format(cb_partial_out);
                            pack_tile(0, cb_partial_out);
                            tile_regs_release();
                        }
                        cb_push_back(cb_partial_out, out_chunk_tiles);

                        cb_pop_front(alias_prev_max, Sq_chunk_t);
                        cb_pop_front(alias_prev_sum, Sq_chunk_t);
                        cb_pop_front(alias_mm2_prev_out, out_chunk_tiles);
                        cb_pop_front(cb_q_in, q_chunk_tiles);
                        continue;  // skip dilution / recip / normalize / write
                    }

                    // ── Tier 2A Phase 2.3 step 4: reducer cross-core merge ──
                    // The writer kernel has waited on the per-program semaphore
                    // until all K-1 workers `noc_semaphore_inc`'d, then pushed
                    // K slots' worth of tiles onto cb_remote_*. Slot 0 is the
                    // reducer's own (unused — its state is in alias_prev_*),
                    // slots 1..K-1 hold peer partials.
                    //
                    // For each peer, perform the symmetric online-softmax
                    // merge:
                    //   new_max  = max(prev_max, peer_max)
                    //   self_d   = exp((prev_max - new_max) * scale)
                    //   peer_d   = exp((peer_max - new_max) * scale)
                    //   prev_sum = self_d * prev_sum + peer_d * peer_sum
                    //   prev_out = self_d * prev_out + peer_d * peer_out
                    //   prev_max = new_max
                    if (cores_per_head_arg > 1 && is_reducer) {
                        // Wait for all K worth of slots (writer pushes K at once).
                        cb_wait_front(cb_remote_max, cores_per_head_arg * Sq_chunk_t);
                        cb_wait_front(cb_remote_sum, cores_per_head_arg * Sq_chunk_t);
                        cb_wait_front(cb_remote_out, cores_per_head_arg * out_chunk_tiles);

#ifdef TQ_DPRINT_KSPLIT
                        // Reducer side: dump cb_remote_max for every slot 0..K-1 so we
                        // can confirm the worker bytes actually arrived intact via NoC.
                        // Slot 0 is reducer's own (never written), slots 1..K-1 are
                        // peer partials. TRISC TileSlice always uses unpack-side metadata.
                        for (uint32_t s = 0; s < cores_per_head_arg; ++s) {
                            DPRINT_UNPACK(
                                DPRINT << "[TQ_DBG] RED nq=" << (uint32_t)nq << " RMAX[" << s << "]="
                                       << TileSlice(
                                              cb_remote_max,
                                              s,
                                              SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1},
                                              true,
                                              true)
                                       << ENDL());
                        }
#endif

                        // Discard slot 0 (reducer's own slot, never written).
                        cb_pop_front(cb_remote_max, Sq_chunk_t);
                        cb_pop_front(cb_remote_sum, Sq_chunk_t);
                        cb_pop_front(cb_remote_out, out_chunk_tiles);

                        for (uint32_t w = 1; w < cores_per_head_arg; ++w) {
                            // 1. cb_merge_new_max = max(alias_prev_max, cb_remote_max[front])
                            //    max_block does NOT pop either input.
                            max_block<>(alias_prev_max, cb_remote_max, cb_merge_new_max, Sq_chunk_t);
#ifdef TQ_DPRINT_KSPLIT
                            // After max_block: dump cb_merge_new_max — first op in the
                            // merge cluster; if this goes non-finite the bug is upstream
                            // (max_block reading cross-format inputs incorrectly), if
                            // it stays finite the bug is in a later merge op.
                            DPRINT_UNPACK(
                                DPRINT << "[TQ_DBG] RED nq=" << (uint32_t)nq << " w=" << w << " NEWMAX="
                                       << TileSlice(
                                              cb_merge_new_max,
                                              0,
                                              SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1},
                                              true,
                                              true)
                                       << ENDL());
#endif

                            // 2. cb_exp_max_diff = exp((alias_prev_max - cb_merge_new_max) * scale)
                            sub_exp_block<scale_fp32>(alias_prev_max, cb_merge_new_max, cb_exp_max_diff, Sq_chunk_t);

                            // 3. alias_prev_sum *= cb_exp_max_diff (col-bcast, no pop of either)
                            //    For Sq_chunk_t=1 the in-place pop/re-push of alias_prev_sum is a
                            //    queue rotation by 1 slot — harmless because the CB only holds
                            //    1 useful slot. cb_exp_max_diff retains its front for the out scale.
                            mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                            // 4. cb_merge_peer_diff = exp((cb_remote_max - cb_merge_new_max) * scale)
                            sub_exp_block<scale_fp32>(cb_remote_max, cb_merge_new_max, cb_merge_peer_diff, Sq_chunk_t);

                            // 5. cb_partial_sum = cb_remote_sum * cb_merge_peer_diff
                            //    Use immediate_pop=true so cb_remote_sum is popped FROM THE FRONT
                            //    by exactly Sq_chunk_t tiles (advances to the next peer's slot).
                            //    A naive in-place version would pop+push and rotate the queue,
                            //    breaking subsequent K>2 iterations.
                            mul_block_bcast_cols<Sq_chunk_t, 1, true, false>(
                                cb_remote_sum, cb_merge_peer_diff, cb_partial_sum);

                            // 6. alias_prev_sum += cb_partial_sum (pops both, re-pushes alias_prev_sum)
                            add_block_inplace(alias_prev_sum, cb_partial_sum, Sq_chunk_t);

                            // 7. alias_mm2_prev_out *= cb_exp_max_diff (pops cb_exp_max_diff)
                            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(alias_mm2_prev_out, cb_exp_max_diff);

                            // 8. Recompute peer_diff (step 5 popped it via mul_block_bcast_cols).
                            sub_exp_block<scale_fp32>(cb_remote_max, cb_merge_new_max, cb_merge_peer_diff, Sq_chunk_t);

                            // 9. cb_partial_out = cb_remote_out * cb_merge_peer_diff
                            //    immediate_pop=true again — pops cb_remote_out's front slot
                            //    (out_chunk_tiles tiles) and cb_merge_peer_diff (Sq_chunk_t tiles).
                            mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false>(
                                cb_remote_out, cb_merge_peer_diff, cb_partial_out);

                            // 10. alias_mm2_prev_out += cb_partial_out (pops both, re-pushes alias_mm2_prev_out)
                            add_block_inplace(alias_mm2_prev_out, cb_partial_out, out_chunk_tiles);

                            // 11. alias_prev_max := cb_merge_new_max
                            cb_pop_front(alias_prev_max, Sq_chunk_t);
                            move_block<true>(cb_merge_new_max, alias_prev_max, Sq_chunk_t);

                            // Advance cb_remote_max to the next peer's slot. cb_remote_sum/out
                            // were already advanced by the immediate_pop mul_block_bcast_cols.
                            cb_pop_front(cb_remote_max, Sq_chunk_t);
                        }
                    }

                    // ── Softmax-denominator correction for unfilled K positions ──
                    // The chunk loop iterates fixed k_chunk_size_tokens (=128) positions
                    // per chunk, but only (cur_pos+1) positions have real data; the rest
                    // have K=0 (cache zero-init). Each zero K position contributes
                    // exp((0 - max) * scale) = exp(-max * scale) to the softmax denominator,
                    // diluting real attention weights to ~30% of correct.
                    //
                    // V[zero] = 0 contributes nothing to the numerator (mm2_prev_out), so
                    // direction is correct, but magnitude is wrong. LayerNorm partially
                    // compensates per-layer, but the bias compounds across 32 layers and
                    // produces garbage tokens at e2e (root-cause-finding 2026-04-28).
                    //
                    // Fix: subtract zero_count * exp(-max * scale) from prev_sum so the
                    // denominator only accounts for real positions.
                    const uint32_t total_iterated = valid_k_chunks * k_chunk_size_tokens;
                    const uint32_t real_count = cur_pos_nb + 1;
                    if (total_iterated > real_count) {
                        const uint32_t zero_count = total_iterated - real_count;
                        union {
                            uint32_t u;
                            float f;
                        } zc_conv;
                        zc_conv.f = (float)zero_count;
                        // Negative scale (flip sign bit of FP32 bit pattern) for exp(-max*scale).
                        constexpr uint32_t neg_scale_fp32 = scale_fp32 ^ 0x80000000u;

                        // Compute correction = zero_count * exp(-prev_max * scale) into cb_exp_max_diff.
                        cb_wait_front(alias_prev_max, Sq_chunk_t);
                        cb_reserve_back(cb_exp_max_diff, Sq_chunk_t);
                        for (uint32_t i = 0; i < Sq_chunk_t; ++i) {
                            tile_regs_acquire();
                            copy_tile_to_dst_init_short(alias_prev_max);
                            copy_tile(alias_prev_max, i, 0);
                            // exp(prev_max * (-scale)) = exp(-max * scale)
                            exp_tile_init<true, neg_scale_fp32, InputClamping::None>();
                            exp_tile<true, false, InputClamping::None>(0);
                            // Multiply by zero_count (scalar) using fill+mul_binary.
                            fill_tile_init();
                            fill_tile(1, zc_conv.f);
                            mul_binary_tile_init();
                            mul_binary_tile(0, 1, 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_reconfig_data_format(cb_exp_max_diff);
                            pack_tile(0, cb_exp_max_diff);
                            tile_regs_release();
                        }
                        cb_push_back(cb_exp_max_diff, Sq_chunk_t);

                        // Subtract correction from alias_prev_sum (in-place).
                        sub_tiles_init(alias_prev_sum, cb_exp_max_diff);
                        cb_wait_front(cb_exp_max_diff, Sq_chunk_t);
                        cb_wait_front(alias_prev_sum, Sq_chunk_t);
                        for (uint32_t i = 0; i < Sq_chunk_t; ++i) {
                            tile_regs_acquire();
                            sub_tiles(alias_prev_sum, cb_exp_max_diff, i, i, 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_reconfig_data_format(alias_prev_sum);
                            pack_tile(0, alias_prev_sum);
                            tile_regs_release();
                        }
                        cb_pop_front(alias_prev_sum, Sq_chunk_t);
                        cb_pop_front(cb_exp_max_diff, Sq_chunk_t);
                        cb_reserve_back(alias_prev_sum, Sq_chunk_t);
                        cb_push_back(alias_prev_sum, Sq_chunk_t);
                    }

                    // ── LSE export (sliding-window hybrid) ──
                    // Pack LSE = prev_max·scale + log(prev_sum) to cb_lse_out
                    // (c_3) BEFORE the recip+normalize. The kernel's softmax
                    // computes exp((s − max)·scale), so prev_sum holds
                    // Σ exp((s − max)·scale) and the proper LSE for the *scaled*
                    // scores is max·scale + log(sum). Used by the host-side
                    // hybrid combine (LSE_COMBINE_DESIGN.md). c_3 is otherwise
                    // the Tier 2A reducer's cb_merge_new_max — return_lse and
                    // cores_per_head_arg > 1 must not be set at the same time.
                    if constexpr (return_lse) {
                        constexpr uint32_t cb_lse_out = tt::CBIndex::c_3;
                        cb_reserve_back(cb_lse_out, Sq_chunk_t);
                        cb_wait_front(alias_prev_max, Sq_chunk_t);
                        for (uint32_t i = 0; i < Sq_chunk_t; ++i) {
                            tile_regs_acquire();
                            // DST 0 = sum
                            copy_tile_to_dst_init_short(alias_prev_sum);
                            copy_tile(alias_prev_sum, i, 0);
                            // DST 0 = log(sum)
                            log_tile_init();
                            log_tile(0);
                            // DST 1 = max
                            copy_tile_to_dst_init_short(alias_prev_max);
                            copy_tile(alias_prev_max, i, 1);
                            // DST 1 = max · scale  (matches the kernel's softmax scaling)
                            binop_with_scalar_tile_init();
                            mul_unary_tile(1, scale_fp32);
                            // DST 0 = log(sum) + max·scale = LSE
                            add_binary_tile_init();
                            add_binary_tile(0, 1, 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_reconfig_data_format(cb_lse_out);
                            pack_tile(0, cb_lse_out);
                            tile_regs_release();
                        }
                        cb_push_back(cb_lse_out, Sq_chunk_t);
                    }

                    recip_block_inplace(alias_prev_sum, Sq_chunk_t);
                    pack_reconfig_data_format(cb_out);
                    mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(alias_mm2_prev_out, alias_prev_sum, cb_out);
                    cb_pop_front(alias_prev_max, Sq_chunk_t);
                    cb_pop_front(cb_q_in, q_chunk_tiles);
                }
            }  // end !pre_rescaled branch
        }
    }

    // Release the cur_pos CB so the reader can refill it on the next trace replay.
    cb_pop_front(cb_cur_pos, 1);
}
