// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/bfloat16.h"
#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

/* This kernel does:
Top-p Cumulative Probability Filtering:
Iteratively accumulates probabilities, comparing them against the nucleus threshold p to determine the smallest set of
tokens satisfying cumulative probabilty > p condition.

Top-k Sampling:
Samples from the top-k subset by comparing cumulative sums of probabilities with a random threshold to select the
appropriate index.
*/

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    uint32_t arg_id = 0;
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(0);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_mask = get_compile_time_arg_val(2);
    constexpr uint32_t scale_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t packed_identity_scalar = get_compile_time_arg_val(4);
    constexpr uint32_t output_final_indices_rm_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t output_local_values_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t output_local_indices_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t final_indices_stick_size = get_compile_time_arg_val(8);
    constexpr uint32_t out_stick_size = get_compile_time_arg_val(9);
    constexpr uint32_t rand_tile_index = get_compile_time_arg_val(10);
    constexpr uint32_t k = get_compile_time_arg_val(11);
    constexpr uint32_t p = get_compile_time_arg_val(12);
    constexpr uint32_t core_id = get_compile_time_arg_val(13);
    constexpr uint32_t ids_per_batch = get_compile_time_arg_val(14);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    generate_reduce_scaler(scale_cb_index, packed_identity_scalar);

    // generate the top-k mask
    constexpr uint32_t one = 1;
    generate_mask<cb_id_mask, one, ids_per_batch / 32>(one, k - 1);

    // get random number
    cb_wait_front(rand_tile_index, 1);
    uint32_t cb_rand_addr = get_write_ptr(rand_tile_index);
    volatile tt_l1_ptr uint16_t* rand_values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_rand_addr);
    uint16_t rand = rand_values[0];

    // wait for compute kernel
    cb_wait_front(output_final_indices_rm_cb_index, 32);
    cb_wait_front(output_local_values_cb_index, 1);
    cb_wait_front(output_local_indices_cb_index, 1);

    // Use cb as L1 scratch memory
    uint32_t cb_local_values_addr = get_write_ptr(output_local_values_cb_index);
    volatile tt_l1_ptr uint16_t* local_values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_local_values_addr);

    uint32_t cb_local_indices_addr = get_write_ptr(output_local_indices_cb_index);
    volatile tt_l1_ptr uint16_t* local_indices = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_local_indices_addr);

    uint32_t cb_final_indices_addr = get_write_ptr(output_final_indices_rm_cb_index);
    volatile tt_l1_ptr uint32_t* final_indices =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_final_indices_addr + core_id * final_indices_stick_size);

    uint32_t out_addr = get_write_ptr(cb_id_out);
    volatile tt_l1_ptr uint32_t* index_out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);

    uint32_t start_id_local_phase_0 = core_id * 16;
    // each user is on 1 core, so core_id = user_id
    // users 0-16 have their data on first 2 faces (2 * 16 * 16 values)
    // skip the first 2 faces for users>=16
    if (core_id >= 16) {
        start_id_local_phase_0 = 32 * 16 + (core_id - 16) * 16;
    }
    uint32_t end_id_local_phase_0 = start_id_local_phase_0 + 16;
    uint32_t start_id_local_phase_1 = 16 * 16 + start_id_local_phase_0;
    uint32_t end_id_local_phase_1 = start_id_local_phase_1 + (k - 16);
    if (k <= 16) {
        end_id_local_phase_0 = start_id_local_phase_0 + k;
        start_id_local_phase_1 = end_id_local_phase_0;
        end_id_local_phase_1 = end_id_local_phase_0;
    }

    if (p != 0) {
        uint16_t bf16_p = static_cast<uint16_t>(p & 0xFFFF);
        uint32_t cum_prob = 0;
        bool cutoff_found = false;
        uint32_t top_p_cutoff = end_id_local_phase_1;  // Default to all tokens
        for (uint32_t i = start_id_local_phase_0; i < end_id_local_phase_0; ++i) {
            bfloat16_add(cum_prob, local_values[i]);
            if (bfloat16_greater(cum_prob, bf16_p)) {
                top_p_cutoff = i + 1;  // Include this token in the top-p set
                cutoff_found = true;
                break;
            }
        }
        if (!cutoff_found) {
            for (uint32_t i = start_id_local_phase_1; i < end_id_local_phase_1; ++i) {
                // cum sum of local values
                cum_prob = bfloat16_add(cum_prob, local_values[i]);
                if (bfloat16_greater(cum_prob, bf16_p)) {
                    top_p_cutoff = i + 1;
                    break;
                }
            }
        }

        // adjust phase indices
        end_id_local_phase_1 = start_id_local_phase_1 + (top_p_cutoff - 16);
        if (top_p_cutoff <= 16) {
            end_id_local_phase_0 = start_id_local_phase_0 + top_p_cutoff;
            start_id_local_phase_1 = end_id_local_phase_0;
            end_id_local_phase_1 = end_id_local_phase_0;
        }
    }

    uint32_t cum_sum = 0;
    index_out[core_id] = final_indices[local_indices[start_id_local_phase_0]];
    bool index_found = false;

    // Sample from the top-k values
    for (uint32_t i = start_id_local_phase_0; i < end_id_local_phase_0; ++i) {
        // cum sum of local values
        cum_sum = bfloat16_add(cum_sum, local_values[i]);
        if (bfloat16_greater(cum_sum, rand)) {
            index_out[core_id] = final_indices[local_indices[i]];
            index_found = true;
            break;
        }
    }
    if (!index_found) {
        for (uint32_t i = start_id_local_phase_1; i < end_id_local_phase_1; ++i) {
            // cum sum of local values
            cum_sum = bfloat16_add(cum_sum, local_values[i]);
            if (bfloat16_greater(cum_sum, rand)) {
                index_out[core_id] = final_indices[local_indices[i]];
                index_found = true;
                break;
            }
        }
    }

    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_size};
    uint64_t dst_noc_addr = get_noc_addr(0, s_out);
    noc_async_write(out_addr + core_id * 4, dst_noc_addr + core_id * 4, 4);
    noc_async_write_barrier();
}
