// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/numeric/bfloat16.h"
#include <stdint.h>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
/* This kernel does:
Top-p Cumulative Probability Filtering:
Iteratively accumulates probabilities, comparing them against the nucleus threshold p to determine the smallest set of
tokens satisfying cumulative probability > p condition.

Top-k Sampling:
Samples from the top-k subset by comparing cumulative sums of probabilities with a random threshold to select the
appropriate index.
*/

constexpr uint32_t FACE_WIDTH = 16;
constexpr uint32_t FACE_HEIGHT = 16;

// Widen bf16 to float32 — exact since bf16 is a subset of float32.
// Uses soft-float on the data-movement RISC-V core.
FORCE_INLINE float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t temp_addr = get_arg_val<uint32_t>(1);
    uint32_t k_addr = get_arg_val<uint32_t>(2);
    uint32_t p_addr = get_arg_val<uint32_t>(3);

    uint32_t arg_id = 0;
    constexpr auto dst_args = TensorAccessorArgs<0>();
    constexpr auto temp_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<temp_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    constexpr uint32_t args_base = p_args.next_compile_time_args_offset();
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(args_base + 0);
    constexpr uint32_t cb_id_mask = get_compile_time_arg_val(args_base + 1);
    constexpr uint32_t scaler_max_cb_id = get_compile_time_arg_val(args_base + 2);
    constexpr uint32_t scaler_sum_cb_id = get_compile_time_arg_val(args_base + 3);
    constexpr uint32_t output_final_indices_rm_cb_index = get_compile_time_arg_val(args_base + 4);
    constexpr uint32_t output_local_values_cb_index = get_compile_time_arg_val(args_base + 5);
    constexpr uint32_t output_local_indices_cb_index = get_compile_time_arg_val(args_base + 6);
    constexpr uint32_t final_indices_stick_size = get_compile_time_arg_val(args_base + 7);
    // args_base + 8: out_stick_size (passed from factory, unused in kernel)
    constexpr uint32_t rand_tile_index = get_compile_time_arg_val(args_base + 9);
    constexpr uint32_t cb_id_k = get_compile_time_arg_val(args_base + 10);
    constexpr uint32_t cb_id_p = get_compile_time_arg_val(args_base + 11);
    constexpr uint32_t cb_id_temp = get_compile_time_arg_val(args_base + 12);
    constexpr uint32_t core_id = get_compile_time_arg_val(args_base + 13);
    constexpr uint32_t ids_per_batch = get_compile_time_arg_val(args_base + 14);
    constexpr uint32_t num_cores = get_compile_time_arg_val(args_base + 15);
    // Local sort-index width must match the index CB format / fp32_dest_acc_en chosen by the host:
    // 32-bit (Int32) on Quasar, 16-bit (UInt16) on WH/BH.
    constexpr bool use_32bit_index = get_compile_time_arg_val(args_base + 16) == 1;
    // Number of running cores / users. The final-indices CB holds one stick per user (no longer
    // hard-coded to 32), so this kernel waits/pops exactly `num_users` sticks.
    constexpr uint32_t num_users = get_compile_time_arg_val(args_base + 17);
    // WAR-hazard signal: when enabled, increment `war_sem_addr` on the gather's drain core once at
    // the very end of the op so the next decode step's SAMPLING_VALUES all-gather can safely reuse
    // the persistent buffer. Closes a cross-sub-device Write-After-Read race that only shows up under
    // trace (see models/common/sampling and the llama3_70b_galaxy TT_CCL war_sem).
    constexpr uint32_t signal_war_sem = get_compile_time_arg_val(args_base + 18);
    constexpr uint32_t war_sem_addr = get_compile_time_arg_val(args_base + 19);
    constexpr uint32_t war_drain_noc_x = get_compile_time_arg_val(args_base + 20);
    constexpr uint32_t war_drain_noc_y = get_compile_time_arg_val(args_base + 21);
    constexpr uint32_t k_chunk_size = num_cores * sizeof(uint32_t);     // 4 bytes per uint32_t
    constexpr uint32_t p_chunk_size = num_cores * sizeof(uint16_t);     // 2 bytes per uint16_t
    constexpr uint32_t temp_chunk_size = num_cores * sizeof(uint16_t);  // 2 bytes per uint16_t
    constexpr uint32_t out_chunk_size = num_cores * sizeof(uint32_t);   // 4 bytes per uint32_t
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<scaler_max_cb_id, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<scaler_sum_cb_id, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    // read k, p, temp

    Noc noc;
    CircularBuffer cb_k(cb_id_k);
    CircularBuffer cb_p(cb_id_p);
    CircularBuffer cb_temp(cb_id_temp);
    CircularBuffer cb_rand(rand_tile_index);
    CircularBuffer cb_final_indices(output_final_indices_rm_cb_index);
    CircularBuffer cb_local_values(output_local_values_cb_index);
    CircularBuffer cb_local_indices(output_local_indices_cb_index);
    CircularBuffer cb_out(cb_id_out);

    const auto addrg_k = TensorAccessor(k_args, k_addr);
    cb_k.reserve_back(1);
    uint32_t cb_id_k_ptr = cb_k.get_write_ptr();
    // Read the entire aligned chunk to avoid NOC alignment issues
    noc.async_read(addrg_k, cb_k, k_chunk_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_k.push_back(1);
    CoreLocalMem<volatile uint32_t> k_ptr(cb_id_k_ptr);
    // Index into the chunk to get this core's value
    uint32_t k = k_ptr[core_id];

    const auto addrg_p = TensorAccessor(p_args, p_addr);
    cb_p.reserve_back(1);
    uint32_t cb_id_p_ptr = cb_p.get_write_ptr();
    // Read the entire aligned chunk to avoid NOC alignment issues
    noc.async_read(addrg_p, cb_p, p_chunk_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_p.push_back(1);
    CoreLocalMem<volatile uint16_t> p_ptr(cb_id_p_ptr);
    // Index into the chunk to get this core's value
    uint32_t p = p_ptr[core_id];

    const auto addrg_temp = TensorAccessor(temp_args, temp_addr);
    // cb_temp.reserve_back(1);
    uint32_t cb_id_temp_ptr = cb_temp.get_write_ptr();
    // Read the entire aligned chunk to avoid NOC alignment issues
    noc.async_read(addrg_temp, cb_temp, temp_chunk_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    // cb_temp.push_back(1);

    CoreLocalMem<volatile uint16_t> temp_ptr(cb_id_temp_ptr);
    // Index into the chunk to get this core's value
    uint16_t temp = temp_ptr[core_id];
    uint32_t temp_packed = (static_cast<uint32_t>(temp) << 16) + static_cast<uint32_t>(temp);
    generate_bcast_unary_scalar(CircularBuffer(cb_id_temp), temp_packed);
    // generate the top-k mask
    constexpr uint32_t one = 1;
    generate_mask<cb_id_mask, one>(one, ids_per_batch / 32, k - 1);
    // get random number
    cb_rand.wait_front(1);
    CoreLocalMem<volatile uint16_t> rand_values(cb_rand.get_read_ptr());
    uint16_t rand = rand_values[0];
    // wait for compute kernel
    cb_final_indices.wait_front(num_users);
    cb_local_values.wait_front(1);
    cb_local_indices.wait_front(1);
    // Read producer-written compute outputs from these CBs in L1.
    CoreLocalMem<volatile uint16_t> local_values(cb_local_values.get_read_ptr());

    using local_index_t = std::conditional_t<use_32bit_index, uint32_t, uint16_t>;
    CoreLocalMem<volatile local_index_t> local_indices(cb_local_indices.get_read_ptr());

    CoreLocalMem<volatile uint32_t> final_indices(cb_final_indices.get_read_ptr() + core_id * final_indices_stick_size);

    uint32_t out_addr = cb_out.get_write_ptr();
    CoreLocalMem<volatile uint32_t> index_out(out_addr);

    uint32_t start_id_local_phase_0 = core_id * FACE_WIDTH;
    // each user is on 1 core, so core_id = user_id
    // users 0-16 have their data on first 2 faces (2 * FACE_WIDTH * FACE_HEIGHT = 2*16*16 = 512 values)
    // skip the first 2 faces for users >= FACE_WIDTH users (16 users)
    if (core_id >= FACE_WIDTH) {
        start_id_local_phase_0 = 2 * FACE_WIDTH * FACE_HEIGHT + (core_id - FACE_WIDTH) * FACE_WIDTH;
    }
    uint32_t end_id_local_phase_0 = start_id_local_phase_0 + FACE_WIDTH;
    uint32_t start_id_local_phase_1 = FACE_WIDTH * FACE_HEIGHT + start_id_local_phase_0;
    uint32_t end_id_local_phase_1 = start_id_local_phase_1 + (k - FACE_WIDTH);
    if (k <= FACE_WIDTH) {
        end_id_local_phase_0 = start_id_local_phase_0 + k;
        start_id_local_phase_1 = end_id_local_phase_0;
        end_id_local_phase_1 = end_id_local_phase_0;
    }

    // Top-p filtering in float32 for precision
    float p_f = bf16_to_f32(static_cast<uint16_t>(p & 0xFFFF));
    float cum_prob_f = 0.0f;
    uint32_t kept_tokens = 0;
    bool cutoff_found_in_phase_0 = false;
    bool cutoff_found_in_phase_1 = false;
    uint32_t top_p_cutoff = end_id_local_phase_1;  // Default to all tokens
    for (uint32_t i = start_id_local_phase_0; i < end_id_local_phase_0; ++i) {
        cum_prob_f += bf16_to_f32(local_values[i]);
        if (cum_prob_f > p_f) {
            top_p_cutoff = i + 1;
            cutoff_found_in_phase_0 = true;
            kept_tokens = top_p_cutoff - start_id_local_phase_0;
            break;
        }
    }
    if (!cutoff_found_in_phase_0) {
        kept_tokens = FACE_WIDTH;
        for (uint32_t i = start_id_local_phase_1; i < end_id_local_phase_1; ++i) {
            // cum sum of local values
            cum_prob_f += bf16_to_f32(local_values[i]);
            if (cum_prob_f > p_f) {
                top_p_cutoff = i + 1;
                kept_tokens += top_p_cutoff - start_id_local_phase_1;
                cutoff_found_in_phase_1 = true;
                break;
            }
        }
    }
    // adjust phase indices
    if (cutoff_found_in_phase_0) {
        // skip last FACE_WIDTH tokens since cutoff found in phase 0
        start_id_local_phase_1 = end_id_local_phase_0;
        end_id_local_phase_1 = end_id_local_phase_0;
        // adjust phase 0 to only keep the tokens that are in the top-p set
        end_id_local_phase_0 = start_id_local_phase_0 + kept_tokens;
    } else if (cutoff_found_in_phase_1) {
        // in case cutoff not found in phase 0, but in phase 1,
        // keep all tokens in phase 0 and part of tokens in phase 1 which is (kept_tokens - FACE_WIDTH)
        end_id_local_phase_1 = start_id_local_phase_1 + (kept_tokens - FACE_WIDTH);
    }

    // Stochastic sampling in float32
    float rand_f = bf16_to_f32(rand);
    float cum_sum_f = 0.0f;
    index_out[core_id] = final_indices[local_indices[start_id_local_phase_0]];
    bool index_found = false;

    for (uint32_t i = start_id_local_phase_0; i < end_id_local_phase_0; ++i) {
        cum_sum_f += bf16_to_f32(local_values[i]) / cum_prob_f;
        if (cum_sum_f > rand_f) {
            index_out[core_id] = final_indices[local_indices[i]];
            index_found = true;
            break;
        }
    }
    if (!index_found) {
        for (uint32_t i = start_id_local_phase_1; i < end_id_local_phase_1; ++i) {
            cum_sum_f += bf16_to_f32(local_values[i]) / cum_prob_f;
            if (cum_sum_f > rand_f) {
                index_out[core_id] = final_indices[local_indices[i]];
                index_found = true;
                break;
            }
        }
    }

    // Release consumed CBs
    cb_rand.pop_front(1);
    cb_local_values.pop_front(1);
    cb_local_indices.pop_front(1);
    cb_final_indices.pop_front(num_users);

    const auto s_out = TensorAccessor(dst_args, dst_addr);
    // Write individual core result - output buffer should handle alignment
    noc.async_write(
        use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out),
        s_out,
        4,
        {.offset_bytes = core_id * 4},
        {.page_id = 0, .offset_bytes = core_id * 4});
    noc.async_write_barrier();

    // Signal WAR completion: this core is done reading the gathered SAMPLING_VALUES/INDICES for this
    // step. Increment the war semaphore on the gather's drain core; the next step's SAMPLING_VALUES
    // all-gather waits until all `num_users` sampling cores have signalled before overwriting the
    // reused persistent buffer.
    if constexpr (signal_war_sem != 0) {
        const uint64_t war_noc_addr =
            get_noc_addr(war_drain_noc_x, war_drain_noc_y, war_sem_addr, noc.get_noc_id());
        noc_semaphore_inc(war_noc_addr, 1, noc.get_noc_id());
        noc.async_atomic_barrier();
    }
}
