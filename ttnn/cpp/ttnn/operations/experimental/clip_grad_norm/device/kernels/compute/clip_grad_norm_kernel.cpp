// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_binary.h"

// Define reduce defaults BEFORE including reduce.h (will be overridden in template calls)
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "debug/dprint.h"
#include "debug/waypoint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    uint32_t umax_norm = get_compile_time_arg_val(2);
    uint32_t up = get_compile_time_arg_val(3);
    uint32_t ueps = get_compile_time_arg_val(4);

    // Reinterpret compile-time args as floats
    union {
        uint32_t u;
        float f;
    } u_max_norm, u_p, u_eps;
    u_max_norm.u = umax_norm;
    u_p.u = up;
    u_eps.u = ueps;
    float max_norm = u_max_norm.f;
    float p = u_p.f;
    float eps = u_eps.f;

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_2;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;         // For reduce scaler (all 1s)
    constexpr uint32_t cb_norm_partial = tt::CBIndex::c_3;   // Per-core partial norm (compute → reader)
    constexpr uint32_t cb_norm_global = tt::CBIndex::c_5;    // Global norm (sender → all cores)
    constexpr uint32_t cb_norm_external = tt::CBIndex::c_6;  // All partials for reduction (sender only)
    constexpr uint32_t cb_temp = tt::CBIndex::c_4;           // Temporary CB

    // Read runtime args to determine if we're the sender core
    uint32_t num_cores = get_arg_val<uint32_t>(4);  // num_cores
    uint32_t core_id = get_arg_val<uint32_t>(5);    // core_id (0 = sender)
    bool is_sender = (core_id == 0);

    WAYPOINT("INI");
    // Note: We don't call init_sfpu() here because we use reduce_init() which handles
    // the necessary initialization. init_sfpu() includes a tensix_sync() that can deadlock.
    binop_with_scalar_tile_init();
    WAYPOINT("IN2");

    // Initialize scaler CB with tile of all 1s
    tile_regs_acquire();
    cb_reserve_back(cb_scaler, 1);
    fill_tile(0, 1.0f);
    pack_tile(0, cb_scaler);
    cb_push_back(cb_scaler, 1);
    tile_regs_release();
    WAYPOINT("SCL");

    // Initialize reduce for scalar reduction
    if (p > 1e38f) {
        // L-inf: use MAX reduction
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_in0, cb_scaler, cb_norm_partial);
    } else {
        // L2 or general p: use SUM reduction
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_in0, cb_scaler, cb_norm_partial);
    }
    cb_wait_front(cb_scaler, 1);
    WAYPOINT("RDI");

    // ============================================================================
    // Phase 1: Compute per-core norm contribution using reduce_tile
    // ============================================================================
    WAYPOINT("P1S");
    DPRINT << "COMPUTE: Phase 1 start, per_core_block_cnt=" << per_core_block_cnt
           << " per_core_block_dim=" << per_core_block_dim << ENDL();
    acquire_dst();
    constexpr uint32_t reduce_dst_idx = 0;

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        WAYPOINT("P1B");
        DPRINT << "COMPUTE: block " << block_index << " start" << ENDL();
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            WAYPOINT("P1T");
            DPRINT << "COMPUTE: waiting for tile " << tile_index << ENDL();
            cb_wait_front(cb_in0, 1);
            WAYPOINT("P1W");
            DPRINT << "COMPUTE: got tile " << tile_index << ENDL();
            tile_regs_acquire();

            copy_tile(cb_in0, 0, 0);

            // Compute |x|^p for each element
            if (p == 2.0f) {
                // L2 norm: square each element
                mul_tiles_init(cb_in0, cb_in0);
                mul_tiles(cb_in0, cb_in0, 0, 0, 0);  // reg[0] = x^2
            } else if (p > 1e38f) {
                // L-inf norm: compute abs = max(x, -x)
                copy_tile(cb_in0, 0, 1);
                negative_tile_init();
                negative_tile(1);
                pack_tile(0, cb_temp);
                cb_push_back(cb_temp, 1);
                pack_tile(1, cb_temp);
                cb_push_back(cb_temp, 1);
                cb_wait_front(cb_temp, 1);
                cb_wait_front(cb_temp, 1);
                binary_max_tile_init();
                binary_max_tile(0, 1, 0);  // reg[0] = abs(x)
                cb_pop_front(cb_temp, 1);
                cb_pop_front(cb_temp, 1);
            } else {
                // General p-norm: |x|^p
                // Compute abs = max(x, -x)
                copy_tile(cb_in0, 0, 1);
                negative_tile_init();
                negative_tile(1);
                pack_tile(0, cb_temp);
                cb_push_back(cb_temp, 1);
                pack_tile(1, cb_temp);
                cb_push_back(cb_temp, 1);
                cb_wait_front(cb_temp, 1);
                cb_wait_front(cb_temp, 1);
                binary_max_tile_init();
                binary_max_tile(0, 1, 0);  // reg[0] = abs(x)
                cb_pop_front(cb_temp, 1);
                cb_pop_front(cb_temp, 1);

                // Compute |x|^p
                pack_tile(0, cb_temp);
                cb_push_back(cb_temp, 1);
                fill_tile(1, p);
                cb_wait_front(cb_temp, 1);
                copy_tile(cb_temp, 0, 0);
                power_binary_tile_init();
                power_binary_tile(0, 1, 0);  // reg[0] = |x|^p
                cb_pop_front(cb_temp, 1);
            }

            // Pack to temp CB for reduce_tile
            // Ensure tile operations complete before packing
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_temp);
            tile_regs_release();
            cb_push_back(cb_temp, 1);
            // Wait for pack to complete before using the tile
            cb_wait_front(cb_temp, 1);

            // Reduce tile to scalar and accumulate (reduce_tile accumulates automatically)
            WAYPOINT("P1R");
            if (p > 1e38f) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_temp, cb_scaler, 0, 0, reduce_dst_idx);
            } else {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_temp, cb_scaler, 0, 0, reduce_dst_idx);
            }
            WAYPOINT("P1D");

            cb_pop_front(cb_temp, 1);
            cb_pop_front(cb_in0, 1);
        }
    }

    // Pack accumulated per-core result (output to reader kernel for cross-core reduction)
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_norm_partial, 1);
    WAYPOINT("P1P");
    pack_tile(reduce_dst_idx, cb_norm_partial);
    tile_regs_commit();
    tile_regs_release();
    cb_push_back(cb_norm_partial, 1);
    WAYPOINT("P1E");
    DPRINT << "COMPUTE: pushed partial norm to cb_norm_partial, per_core_block_cnt=" << per_core_block_cnt
           << " per_core_block_dim=" << per_core_block_dim << ENDL();
    release_dst();

    // ============================================================================
    // Phase 2: Get global norm (combine partials if sender, receive if receiver) and compute scale
    // ============================================================================
    WAYPOINT("P2S");
    tile_regs_acquire();

    if (is_sender && num_cores > 1) {
        // Sender core: combine all partials from cb_norm_external using reduce_tile
        WAYPOINT("P2W");
        cb_wait_front(cb_norm_external, num_cores);
        WAYPOINT("P2G");

        // Determine reduce type once (L-inf uses MAX, others use SUM)
        bool use_max_reduce = (p > 1e38f);

        // Initialize reduce for combining all partials
        if (use_max_reduce) {
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_norm_external, cb_scaler, cb_norm_global);
        } else {
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_norm_external, cb_scaler, cb_norm_global);
        }

        acquire_dst();
        constexpr uint32_t reduce_dst_idx = 0;

        // Reduce all partials
        for (uint32_t i = 0; i < num_cores; ++i) {
            if (use_max_reduce) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_norm_external, cb_scaler, i, 0, reduce_dst_idx);
            } else {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_norm_external, cb_scaler, i, 0, reduce_dst_idx);
            }
        }

        // Pack combined result to cb_norm_global (for reader to multicast)
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_norm_global, 1);
        pack_tile(reduce_dst_idx, cb_norm_global);
        tile_regs_release();
        cb_push_back(cb_norm_global, 1);
        release_dst();
        reduce_uninit();

        cb_pop_front(cb_norm_external, num_cores);

        // Global norm is already in reg[reduce_dst_idx] (which is 0) from reduce_tile
        // No need to wait for cb_norm_global or copy it - we already have it in reg[0]
        // Reader will read cb_norm_global and multicast it to all cores
    } else {
        // Receiver core (or single core): wait for global norm from multicast (or use local norm)
        cb_wait_front(cb_norm_global, 1);
        copy_tile(cb_norm_global, 0, 0);  // Load global norm to reg[0]
        // DO NOT pop cb_norm_global - no need, data already copied to reg[0]
    }

    // Compute norm from sum
    if (p == 2.0f) {
        sqrt_tile_init();
        sqrt_tile(0);  // reg[0] = sqrt(sum_sq)
    } else if (p > 1e38f) {
        // L-inf: already max, no transformation
    } else {
        // General p: (sum(|x|^p))^(1/p)
        pack_tile(0, cb_temp);
        cb_push_back(cb_temp, 1);
        float inv_p = 1.0f / p;
        fill_tile(1, inv_p);
        cb_wait_front(cb_temp, 1);
        copy_tile(cb_temp, 0, 0);
        power_binary_tile_init();
        power_binary_tile(0, 1, 0);  // reg[0] = reg[0]^(1/p)
        cb_pop_front(cb_temp, 1);
    }

    // Compute scale = min(1, max_norm / (norm + eps))
    pack_tile(0, cb_temp);
    cb_push_back(cb_temp, 1);
    union {
        uint32_t u;
        float f;
    } u_eps_bits;
    u_eps_bits.f = eps;
    uint32_t eps_bits = u_eps_bits.u;
    cb_wait_front(cb_temp, 1);
    copy_tile(cb_temp, 0, 0);
    add_unary_tile(0, eps_bits);  // reg[0] = norm + eps

    pack_tile(0, cb_temp);
    cb_push_back(cb_temp, 1);
    fill_tile(1, max_norm);
    cb_wait_front(cb_temp, 1);
    copy_tile(cb_temp, 0, 0);
    div_binary_tile_init();
    div_binary_tile(1, 0, 1);  // reg[1] = max_norm / (norm + eps)

    fill_tile(0, 1.0f);
    binary_min_tile_init();
    binary_min_tile(1, 0, 1);  // reg[1] = min(scale, 1.0)

    // Store scale factor in cb_norm_partial (reuse after it's been consumed by reader)
    constexpr uint32_t cb_scale = tt::CBIndex::c_3;  // Reuse cb_norm_partial for scale
    pack_tile(1, cb_scale);
    cb_push_back(cb_scale, 1);
    reduce_uninit();
    tile_regs_release();

    // ============================================================================
    // Phase 3: Scale all input tiles and output
    // ============================================================================
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_out0, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();
            cb_wait_front(cb_in0, 1);
            copy_tile(cb_in0, 0, 0);

            // Load scale factor
            cb_wait_front(cb_scale, 1);
            copy_tile(cb_scale, 0, 1);

            // Scale: reg[0] = reg[0] * reg[1]
            pack_tile(0, cb_temp);
            cb_push_back(cb_temp, 1);
            pack_tile(1, cb_temp);
            cb_push_back(cb_temp, 1);
            cb_wait_front(cb_temp, 1);
            cb_wait_front(cb_temp, 1);
            mul_tiles_init(cb_temp, cb_temp);
            mul_tiles(cb_temp, cb_temp, 0, 0, 0);
            cb_pop_front(cb_temp, 1);
            cb_pop_front(cb_temp, 1);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out0);
            cb_pop_front(cb_in0, 1);
            tile_regs_release();
        }
        cb_push_back(cb_out0, per_core_block_dim);
    }

    cb_pop_front(cb_scale, 1);
}
}  // namespace NAMESPACE
