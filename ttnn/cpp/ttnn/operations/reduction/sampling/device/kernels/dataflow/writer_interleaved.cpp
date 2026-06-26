// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/numeric/bfloat16.h"
#include <stdint.h>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
/* This kernel does:
Top-p Cumulative Probability Filtering + Top-k Sampling. */

constexpr uint32_t FACE_WIDTH = 16;
constexpr uint32_t FACE_HEIGHT = 16;

// Widen bf16 to float32 — exact since bf16 is a subset of float32.
FORCE_INLINE float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

void kernel_main() {
    const uint32_t core_id = get_arg(args::core_id);

    constexpr uint32_t final_indices_stick_size = get_arg(args::final_indices_stick_size);
    constexpr uint32_t ids_per_batch = get_arg(args::ids_per_batch);
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    // Local sort-index width: 32-bit (Int32) on Quasar, 16-bit (UInt16) on WH/BH.
    constexpr bool use_32bit_index = get_arg(args::use_32bit_index) == 1;
    constexpr uint32_t num_users = get_arg(args::num_users);

    constexpr uint32_t cb_id_out = dfb::output;
    constexpr uint32_t cb_id_mask = dfb::topk_mask;
    constexpr uint32_t scaler_max_cb_id = dfb::scaler_max;
    constexpr uint32_t scaler_sum_cb_id = dfb::scaler_sum;
    constexpr uint32_t cb_id_temp = dfb::temp;

    constexpr auto output_final_indices_rm_cb_index = dfb::final_indices;
    constexpr auto output_local_values_cb_index = dfb::local_vals;
    constexpr auto output_local_indices_cb_index = dfb::output_ind;
    constexpr auto rand_tile_index = dfb::rand;
    constexpr auto cb_id_k = dfb::k;
    constexpr auto cb_id_p = dfb::p;

    constexpr uint32_t temp_chunk_size = num_cores * sizeof(uint16_t);  // 2 bytes per uint16_t

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<scaler_max_cb_id, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<scaler_sum_cb_id, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    Noc noc;
    DataflowBuffer cb_k(cb_id_k);
    DataflowBuffer cb_p(cb_id_p);
    DataflowBuffer cb_temp(cb_id_temp);
    DataflowBuffer cb_rand(rand_tile_index);
    DataflowBuffer cb_final_indices(output_final_indices_rm_cb_index);
    DataflowBuffer cb_local_values(output_local_values_cb_index);
    DataflowBuffer cb_local_indices(output_local_indices_cb_index);
    // cb_out uses CircularBuffer for the use<AddrSelector::WRITE_PTR> noc-source helper (DataflowBuffer
    // has no AddrSelector); same physical cb id, FIFO ops drive the cross-kernel bridge to compute.
    CircularBuffer cb_out(cb_id_out);

    // k / p are produced by the reader (cross-kernel bridge): consume + index this core's value.
    cb_k.wait_front(1);
    CoreLocalMem<volatile uint32_t> k_ptr(cb_k.get_read_ptr());
    uint32_t k = k_ptr[core_id];
    cb_k.pop_front(1);

    cb_p.wait_front(1);
    CoreLocalMem<volatile uint16_t> p_ptr(cb_p.get_read_ptr());
    uint32_t p = p_ptr[core_id];
    cb_p.pop_front(1);

    // temp: read the chunk into the temp CB's L1, index this core's value, then produce the
    // broadcast-scalar tile that the compute kernel consumes.
    const auto addrg_temp = TensorAccessor(tensor::temp);
    uint32_t cb_id_temp_ptr = cb_temp.get_write_ptr();
    noc.async_read(addrg_temp, cb_temp, temp_chunk_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    CoreLocalMem<volatile uint16_t> temp_ptr(cb_id_temp_ptr);
    uint16_t temp = temp_ptr[core_id];
    uint32_t temp_packed = (static_cast<uint32_t>(temp) << 16) + static_cast<uint32_t>(temp);
    generate_bcast_unary_scalar(cb_id_temp, temp_packed);

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

    // Output staging CB (cross-kernel bridge: producer here, terminal no-op consumer on compute).
    cb_out.reserve_back(1);
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

    const auto s_out = TensorAccessor(tensor::output);
    // Write individual core result - output buffer should handle alignment
    noc.async_write(
        use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out),
        s_out,
        4,
        {.offset_bytes = core_id * 4},
        {.page_id = 0, .offset_bytes = core_id * 4});
    noc.async_write_barrier();
    // Publish the staged output so the compute kernel's terminal no-op consumer can drain it.
    cb_out.push_back(1);
}
