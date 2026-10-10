// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/debug/dprint.h"
#include "internal/circular_buffer_interface.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/bcast.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/identity.h"
#include "api/compute/compute_kernel_api.h"

// Scaled FP8 combine helpers (same math as per_token_cast_to_fp8: power-of-two scale, RNE to e4m3).
#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <int ITERATIONS = 8>
inline void combine_ceil_power_of_two() {
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::vInt exponent = sfpi::exexp(value, sfpi::ExponentMode::Biased);
        sfpi::vFloat mantissa = sfpi::setexp(value, 127);
        sfpi::vFloat result = sfpi::setexp(sfpi::vFloat(1.0f), exponent);
        v_if(mantissa != 1.0f) { result = sfpi::setexp(sfpi::vFloat(1.0f), exponent + 1); }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
template <int ITERATIONS = 8>
inline void combine_round_to_e4m3_rne() {
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(value);
        sfpi::vUInt lsb = (bits >> 20) & 1;
        bits = bits + 0x7ffffU + lsb;
        bits = bits & 0xfff00000U;
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(bits);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
inline void combine_ceil_power_of_two_tile(uint32_t idst) {
    SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, combine_ceil_power_of_two, (8), idst, VectorMode::RC);
}
inline void combine_round_to_e4m3_rne_tile(uint32_t idst) {
    SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, combine_round_to_e4m3_rne, (8), idst, VectorMode::RC);
}
#endif

#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_COMBINE(...)
#endif

// Compile-time args (shared across untilizer cores):
//   0: cb_untilize_id                  - CB for compute's untilized output (c_2)
//   1: cb_in_id                        - CB for untilize input tile data (c_0)
//   2: cb_experts_tok_counter_id       - CB holding multicasted per-expert token counts (c_1)
//   3: experts_tok_counter_pages       - number of pages in the counter CB
//   4: experts_per_chip                - count of experts mapped to this chip
//   5: counter_offset                  - uint32 offset into the counter buffer for this chip
//   6: max_dispatch_buffer_token_size  - per-chip dispatch capacity (overflow clamp)
//   7: read_batch_size                 - rows per untilize batch (== tile_height for this op)
//   8: full_ct_dim                     - hidden_size / tile_width (tiles per batch)
//   9: block_ct_dim                    - tiles per pack call (largest divisor of full_ct_dim <= 8)
//  10: cb_counter_total_pages          - full page capacity of c_1 (counter + trailer); used
//                                        for cb_wait_front on the multicasted CB
//
// Runtime args (per untilizer core):
//   0: expert_start_idx                - first expert handled (now 0; every group handles all experts)
//   1: expert_end_idx                  - one past the last expert handled (now experts_per_chip)
//   2: core_id                         - local index in the owning sender's untilizer group (0..k_s-1)
//   3: num_untilizer_cores                  - k_s, size of the owning sender's untilizer group
//   4: untilizer_global_pos            - this core's position in the global interleaved untilizer
//                                        ordering; its batches are global_pos, +G, +2G, … per expert
//   5: total_untilizers                - G, total untilizer cores across all senders (global stride)
void kernel_main() {
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(0);
    CircularBuffer cb_untilize(cb_untilize_id);
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(1);
    CircularBuffer cb_in(cb_in_id);
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(2);
    CircularBuffer cb_experts_tok_counter(cb_experts_tok_counter_id);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(3);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(4);
    constexpr uint32_t counter_offset = get_compile_time_arg_val(5);
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(6);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(7);
    constexpr uint32_t full_ct_dim = get_compile_time_arg_val(8);
    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(9);
    constexpr uint32_t cb_counter_total_pages = get_compile_time_arg_val(10);
    constexpr uint32_t num_blocks = full_ct_dim / block_ct_dim;
    constexpr bool fp8_scaled = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t cb_abs_id = get_compile_time_arg_val(12);
    constexpr uint32_t cb_scaler_id = get_compile_time_arg_val(13);
    constexpr uint32_t cb_scale_id = get_compile_time_arg_val(14);
    constexpr uint32_t cb_inv_scale_id = get_compile_time_arg_val(15);
    constexpr uint32_t clamp_min_bits = get_compile_time_arg_val(16);
    constexpr uint32_t inv_448_bits = get_compile_time_arg_val(17);
    CircularBuffer cb_abs(cb_abs_id);
    CircularBuffer cb_scaler(cb_scaler_id);
    CircularBuffer cb_scale(cb_scale_id);
    CircularBuffer cb_inv_scale(cb_inv_scale_id);
    // read_batch_size doubles as tile_height: one tile-row of input -> read_batch_size element rows.
    constexpr uint32_t tile_height = read_batch_size;
    constexpr uint32_t tiles_per_batch = full_ct_dim;

    uint32_t rt_idx = 0;
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_idx++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_idx++);
    uint32_t core_id = get_arg_val<uint32_t>(rt_idx++);
    uint32_t num_untilizer_cores = get_arg_val<uint32_t>(rt_idx++);
    uint32_t untilizer_global_pos = get_arg_val<uint32_t>(rt_idx++);
    uint32_t total_untilizers = get_arg_val<uint32_t>(rt_idx++);
    // core_id / num_untilizer_cores no longer drive batch assignment under the global round-robin.
    (void)core_id;
    (void)num_untilizer_cores;

    if constexpr (fp8_scaled) {
        // Configure on the fp32 reduce operand so the unpacker sees full 4-face tiles (see
        // per_token_cast_to_fp8); every phase below reconfigures its own operands.
        compute_kernel_hw_startup(cb_abs_id, cb_untilize_id);
        cb_scaler.wait_front(1);  // writer_untilize fills the 1.0 reduce scaler once
    } else {
        compute_kernel_hw_startup(cb_in_id, cb_untilize_id);
        pack_untilize_init<block_ct_dim, full_ct_dim>(cb_in_id, cb_untilize_id);
    }

    // Wait for the owning sender's expert-token-count multicast.  reader_untilize on this same
    // core pushes the counter CB once after counter_ready_sem fires, so cb_wait_front here
    // doubles as the gate for "counter data is live in L1".  The CB is never popped — both
    // this kernel and writer_untilize rely on the data staying resident.
    cb_experts_tok_counter.wait_front(cb_counter_total_pages);

    // Snapshot per-expert token counts.  read_tile_value has UNPACK read from L1 and broadcast
    // to MATH / PACK via mailbox, so all three TRISCs end up with identical counts and walk the
    // same iteration sequence.
    uint32_t local_expert_counts[experts_per_chip];
    for (uint32_t e = 0; e < experts_per_chip; e++) {
        local_expert_counts[e] = read_tile_value(cb_experts_tok_counter_id, 0, counter_offset + e);
    }

    // Accumulate dispatch-buffer offset for experts below this core's range using raw counts
    // (host laid the dispatch buffer out with tile-aligned per-expert stride; the same applies
    // to skipped experts).
    uint32_t start_page_tiled = 0;
    for (uint32_t e = 0; e < expert_start_idx; e++) {
        start_page_tiled += ((local_expert_counts[e] + tile_height - 1) / tile_height) * tiles_per_batch;
    }

    for (uint32_t local_expert = expert_start_idx; local_expert < expert_end_idx; local_expert++) {
        uint32_t expert_tokens = local_expert_counts[local_expert];
        uint32_t start_token = (start_page_tiled / tiles_per_batch) * tile_height;
        // Mirror reader_dispatch's overflow guard so we never untilize past the dispatch buffer.
        if (start_token >= max_dispatch_buffer_token_size) {
            expert_tokens = 0;
        } else if (start_token + expert_tokens > max_dispatch_buffer_token_size) {
            expert_tokens = max_dispatch_buffer_token_size - start_token;
        }

        uint32_t actual_batches = (expert_tokens + read_batch_size - 1) / read_batch_size;

        // Global round-robin: this core handles batches untilizer_global_pos, +G, +2G, … of every
        // expert (G = total_untilizers across all senders).  Must match reader_untilize /
        // writer_untilize exactly so the c_0 / c_2 producer-consumer protocol stays in lockstep.
        for (uint32_t batch_idx = untilizer_global_pos; batch_idx < actual_batches; batch_idx += total_untilizers) {
            cb_untilize.reserve_back(read_batch_size);
            for (uint32_t block = 0; block < num_blocks; block++) {
                cb_in.wait_front(block_ct_dim);
                if constexpr (fp8_scaled) {
                    // 1x32 block = one tile row: scale[r] = pow2_up(max(amax_r, 1e-4) / 448) per tile.
                    // Phase A: |x| of each tile (fp32) -> cb_abs.
                    reconfig_full_operand_srca(cb_in_id);
                    pack_reconfig_data_format(cb_abs_id);
                    copy_init(cb_in_id);
                    abs_tile_init();
                    cb_abs.reserve_back(block_ct_dim);
                    for (uint32_t k = 0; k < block_ct_dim; ++k) {
                        tile_regs_acquire();
                        copy_tile(cb_in_id, k, 0);
                        abs_tile(0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, cb_abs_id);
                        tile_regs_release();
                    }
                    cb_abs.push_back(block_ct_dim);
                    // Phase B: row max -> scale (slot 0, to the writer) and 1/scale (slot 1).
                    cb_abs.wait_front(block_ct_dim);
                    cb_scale.reserve_back(block_ct_dim);
                    cb_inv_scale.reserve_back(block_ct_dim);
                    reconfig_data_format(cb_abs_id, cb_scaler_id);
                    pack_reconfig_data_format(cb_scale_id);
                    for (uint32_t k = 0; k < block_ct_dim; ++k) {
                        tile_regs_acquire();
                        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, cb_scale_id);
                        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, k, 0, 0);
                        reduce_uninit();
                        clamp_tile_init();
                        clamp_tile(0, clamp_min_bits, 0x7f7fffffu);
                        binop_with_scalar_tile_init();
                        mul_unary_tile(0, inv_448_bits);
                        MATH((combine_ceil_power_of_two_tile(0)));
                        copy_dest_values_init();
                        copy_dest_values<DataFormat::Float32>(0, 1);
                        recip_tile_init();
                        recip_tile(1);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, cb_scale_id);
                        pack_tile(1, cb_inv_scale_id);
                        tile_regs_release();
                    }
                    cb_scale.push_back(block_ct_dim);
                    cb_inv_scale.push_back(block_ct_dim);
                    cb_abs.pop_front(block_ct_dim);
                    // Phase C: x * (1/scale) per row, RNE onto the e4m3 grid, untilize into the row stride.
                    cb_inv_scale.wait_front(block_ct_dim);
                    reconfig_data_format(cb_in_id, cb_inv_scale_id);
                    mul_bcast_cols_init(cb_in_id, cb_inv_scale_id);
                    pack_reconfig_data_format(cb_untilize_id);
                    pack_untilize_dest_init<block_ct_dim, full_ct_dim>(cb_untilize_id);
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < block_ct_dim; ++k) {
                        mul_tiles_bcast_cols(cb_in_id, cb_inv_scale_id, k, k, k);
                    }
                    MATH((SFPU_UNARY_INIT(unused)));
                    for (uint32_t k = 0; k < block_ct_dim; ++k) {
                        MATH((combine_round_to_e4m3_rne_tile(k)));
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_untilize_dest<block_ct_dim, full_ct_dim>(cb_untilize_id, 1, block);
                    tile_regs_release();
                    pack_untilize_uninit(cb_untilize_id);
                    cb_inv_scale.pop_front(block_ct_dim);
                } else {
                    // DeviceZoneScopedN("UNTILIZING");
                    pack_untilize_block<block_ct_dim, full_ct_dim>(cb_in_id, 1, cb_untilize_id, block);
                }
                cb_in.pop_front(block_ct_dim);
            }
            cb_untilize.push_back(read_batch_size);
        }
        start_page_tiled += ((expert_tokens + tile_height - 1) / tile_height) * tiles_per_batch;
    }

    if constexpr (!fp8_scaled) {
        pack_untilize_uninit(cb_untilize_id);
    }
}
