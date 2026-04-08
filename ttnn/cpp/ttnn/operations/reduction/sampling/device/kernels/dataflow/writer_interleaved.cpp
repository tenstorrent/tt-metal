// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/numeric/bfloat16.h"
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
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
    constexpr uint32_t scale_cb_index = get_compile_time_arg_val(args_base + 2);
    constexpr uint32_t packed_identity_scalar = get_compile_time_arg_val(args_base + 3);
    constexpr uint32_t output_final_indices_rm_cb_index = get_compile_time_arg_val(args_base + 4);
    constexpr uint32_t output_local_values_cb_index = get_compile_time_arg_val(args_base + 5);
    constexpr uint32_t output_local_indices_cb_index = get_compile_time_arg_val(args_base + 6);
    constexpr uint32_t final_indices_stick_size = get_compile_time_arg_val(args_base + 7);
    constexpr uint32_t out_stick_size = get_compile_time_arg_val(args_base + 8);
    constexpr uint32_t rand_tile_index = get_compile_time_arg_val(args_base + 9);
    constexpr uint32_t cb_id_k = get_compile_time_arg_val(args_base + 10);
    constexpr uint32_t cb_id_p = get_compile_time_arg_val(args_base + 11);
    constexpr uint32_t cb_id_temp = get_compile_time_arg_val(args_base + 12);
    constexpr uint32_t core_id = get_compile_time_arg_val(args_base + 13);
    constexpr uint32_t ids_per_batch = get_compile_time_arg_val(args_base + 14);
    constexpr uint32_t num_cores = get_compile_time_arg_val(args_base + 15);
    constexpr uint32_t k_chunk_size = num_cores * sizeof(uint32_t);     // 4 bytes per uint32_t
    constexpr uint32_t p_chunk_size = num_cores * sizeof(uint16_t);     // 2 bytes per uint16_t
    constexpr uint32_t temp_chunk_size = num_cores * sizeof(uint16_t);  // 2 bytes per uint16_t
    constexpr uint32_t out_chunk_size = num_cores * sizeof(uint32_t);   // 4 bytes per uint32_t
    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    generate_reduce_scaler(scale_cb_index, packed_identity_scalar);
    // read k, p, temp

    const auto addrg_k = TensorAccessor(k_args, k_addr, 128);
    cb_reserve_back(cb_id_k, 1);
    uint32_t cb_id_k_ptr = get_write_ptr(cb_id_k);
    uint64_t k_noc_addr = get_noc_addr(0, addrg_k);
    // Read the entire aligned chunk to avoid NOC alignment issues
    noc_async_read(k_noc_addr, cb_id_k_ptr, k_chunk_size);
    noc_async_read_barrier();
    cb_push_back(cb_id_k, 1);
    volatile tt_l1_ptr uint32_t* k_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_id_k_ptr);
    // Index into the chunk to get this core's value
    uint32_t k = k_ptr[core_id];

    const auto addrg_p = TensorAccessor(p_args, p_addr, 64);
    cb_reserve_back(cb_id_p, 1);
    uint32_t cb_id_p_ptr = get_write_ptr(cb_id_p);
    uint64_t p_noc_addr = get_noc_addr(0, addrg_p);
    // Read the entire aligned chunk to avoid NOC alignment issues
    noc_async_read(p_noc_addr, cb_id_p_ptr, p_chunk_size);
    noc_async_read_barrier();
    cb_push_back(cb_id_p, 1);
    volatile tt_l1_ptr uint16_t* p_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_id_p_ptr);
    // Index into the chunk to get this core's value
    uint32_t p = p_ptr[core_id];

    const auto addrg_temp = TensorAccessor(temp_args, temp_addr, 64);
    // cb_reserve_back(cb_id_temp, 1);
    uint32_t cb_id_temp_ptr = get_write_ptr(cb_id_temp);
    uint64_t temp_noc_addr = get_noc_addr(0, addrg_temp);
    // Read the entire aligned chunk to avoid NOC alignment issues
    noc_async_read(temp_noc_addr, cb_id_temp_ptr, temp_chunk_size);
    noc_async_read_barrier();
    // cb_push_back(cb_id_temp, 1);

    volatile tt_l1_ptr uint16_t* temp_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_id_temp_ptr);
    // Index into the chunk to get this core's value
    uint16_t temp = temp_ptr[core_id];
    uint32_t temp_packed = (static_cast<uint32_t>(temp) << 16) + static_cast<uint32_t>(temp);
    generate_bcast_unary_scalar(cb_id_temp, temp_packed);
    // generate the top-k mask
    constexpr uint32_t one = 1;
    generate_mask<cb_id_mask, one>(one, ids_per_batch / 32, k - 1);
    // get random number
    cb_wait_front(rand_tile_index, 1);
    uint32_t cb_rand_addr = get_read_ptr(rand_tile_index);
    volatile tt_l1_ptr uint16_t* rand_values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_rand_addr);
    uint16_t rand = rand_values[0];
    // wait for compute kernel
    cb_wait_front(output_final_indices_rm_cb_index, 32);
    cb_wait_front(output_local_values_cb_index, 1);
    cb_wait_front(output_local_indices_cb_index, 1);
    // Read producer-written compute outputs from these CBs in L1.
    uint32_t cb_local_values_addr = get_read_ptr(output_local_values_cb_index);
    volatile tt_l1_ptr uint16_t* local_values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_local_values_addr);

    uint32_t cb_local_indices_addr = get_read_ptr(output_local_indices_cb_index);
    volatile tt_l1_ptr uint16_t* local_indices = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_local_indices_addr);

    uint32_t cb_final_indices_addr = get_read_ptr(output_final_indices_rm_cb_index);
    volatile tt_l1_ptr uint32_t* final_indices =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_final_indices_addr + core_id * final_indices_stick_size);

    uint32_t out_addr = get_write_ptr(cb_id_out);
    volatile tt_l1_ptr uint32_t* index_out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);

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
    cb_pop_front(rand_tile_index, 1);
    cb_pop_front(output_local_values_cb_index, 1);
    cb_pop_front(output_local_indices_cb_index, 1);
    cb_pop_front(output_final_indices_rm_cb_index, 32);

    const auto s_out = TensorAccessor(dst_args, dst_addr, out_stick_size);
    uint64_t dst_noc_addr = get_noc_addr(0, s_out);
    // Write individual core result - output buffer should handle alignment
    noc_async_write(out_addr + core_id * 4, dst_noc_addr + core_id * 4, 4);
    noc_async_write_barrier();
}
