// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/numeric/bfloat16.h"
#include <stdint.h>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
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
    constexpr auto final_indices_stick_size = get_arg(args::final_indices_stick_size);
    // The host also supplies out_stick_size; this kernel does not read it.
    constexpr auto core_id = get_arg(args::core_id);
    constexpr auto ids_per_batch = get_arg(args::ids_per_batch);
    constexpr auto num_cores = get_arg(args::num_cores);
    // Local sort-index width must match the index buffer format / fp32 dest accumulation chosen by
    // the host: 32-bit (Int32) on Quasar, 16-bit (UInt16) on WH/BH.
    constexpr bool use_32bit_index = get_arg(args::use_32bit_index) == 1;
    // Number of running cores / users. The final-indices buffer holds one stick per user (no longer
    // hard-coded to 32), so this kernel waits/pops exactly `num_users` sticks.
    constexpr auto num_users = get_arg(args::num_users);
    constexpr uint32_t k_chunk_size = num_cores * sizeof(uint32_t);     // 4 bytes per uint32_t
    constexpr uint32_t p_chunk_size = num_cores * sizeof(uint16_t);     // 2 bytes per uint16_t
    constexpr uint32_t temp_chunk_size = num_cores * sizeof(uint16_t);  // 2 bytes per uint16_t
    constexpr uint32_t out_chunk_size = num_cores * sizeof(uint32_t);   // 4 bytes per uint32_t
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler_max, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler_sum, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    // read k, p, temp

    Noc noc;
    DataflowBuffer dfb_k(dfb::k);
    DataflowBuffer dfb_p(dfb::p);
    DataflowBuffer dfb_temp(dfb::temp);
    DataflowBuffer dfb_rand(dfb::rand);
    DataflowBuffer dfb_final_indices(dfb::final_indices);
    DataflowBuffer dfb_local_values(dfb::local_values);
    DataflowBuffer dfb_local_indices(dfb::local_indices);
    DataflowBuffer dfb_out(dfb::out);

    const auto addrg_k = TensorAccessor(tensor::k);
    dfb_k.reserve_back(1);
    uint32_t dfb_k_ptr = dfb_k.get_write_ptr();
    // Read the entire aligned chunk to avoid NOC alignment issues
    noc.async_read(addrg_k, dfb_k, k_chunk_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    dfb_k.push_back(1);
    CoreLocalMem<volatile uint32_t> k_ptr(dfb_k_ptr);
    // Index into the chunk to get this core's value
    uint32_t k = k_ptr[core_id];

    const auto addrg_p = TensorAccessor(tensor::p);
    dfb_p.reserve_back(1);
    uint32_t dfb_p_ptr = dfb_p.get_write_ptr();
    // Read the entire aligned chunk to avoid NOC alignment issues
    noc.async_read(addrg_p, dfb_p, p_chunk_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    dfb_p.push_back(1);
    CoreLocalMem<volatile uint16_t> p_ptr(dfb_p_ptr);
    // Index into the chunk to get this core's value
    uint32_t p = p_ptr[core_id];

    const auto addrg_temp = TensorAccessor(tensor::temp);
    // dfb_temp.reserve_back(1);
    uint32_t dfb_temp_ptr = dfb_temp.get_write_ptr();
    // Read the entire aligned chunk to avoid NOC alignment issues
    noc.async_read(addrg_temp, dfb_temp, temp_chunk_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    // dfb_temp.push_back(1);

    CoreLocalMem<volatile uint16_t> temp_ptr(dfb_temp_ptr);
    // Index into the chunk to get this core's value
    uint16_t temp = temp_ptr[core_id];
    uint32_t temp_packed = (static_cast<uint32_t>(temp) << 16) + static_cast<uint32_t>(temp);
    // This donor takes a CircularBuffer by value, so the buffer handle is wrapped at the call site
    // rather than passed straight through. The header is shared by several op families, so its
    // signature is not this op's to change.
    generate_bcast_unary_scalar(CircularBuffer(dfb::temp), temp_packed);
    // generate the top-k mask
    constexpr uint32_t one = 1;
    generate_mask<dfb::mask, one>(one, ids_per_batch / 32, k - 1);
    // get random number
    dfb_rand.wait_front(1);
    CoreLocalMem<volatile uint16_t> rand_values(dfb_rand.get_read_ptr());
    uint16_t rand = rand_values[0];
    // wait for compute kernel
    dfb_final_indices.wait_front(num_users);
    dfb_local_values.wait_front(1);
    dfb_local_indices.wait_front(1);
    // Read producer-written compute outputs from these buffers in SRAM.
    CoreLocalMem<volatile uint16_t> local_values(dfb_local_values.get_read_ptr());

    using local_index_t = std::conditional_t<use_32bit_index, uint32_t, uint16_t>;
    CoreLocalMem<volatile local_index_t> local_indices(dfb_local_indices.get_read_ptr());

    CoreLocalMem<volatile uint32_t> final_indices(
        dfb_final_indices.get_read_ptr() + core_id * final_indices_stick_size);

    uint32_t out_addr = dfb_out.get_write_ptr();
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

    // Release consumed buffers
    dfb_rand.pop_front(1);
    dfb_local_values.pop_front(1);
    dfb_local_indices.pop_front(1);
    dfb_final_indices.pop_front(num_users);

    const auto s_out = TensorAccessor(tensor::output);
    // Write individual core result - output buffer should handle alignment
    noc.async_write(dfb_out, s_out, 4, {.offset_bytes = core_id * 4}, {.page_id = 0, .offset_bytes = core_id * 4});
    noc.async_write_barrier();
}
