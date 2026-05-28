// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/numeric/bfloat16.h"
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

constexpr uint32_t FACE_WIDTH = 16;
constexpr uint32_t FACE_HEIGHT = 16;

FORCE_INLINE float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

void kernel_main() {
    constexpr uint32_t final_indices_stick_size = get_arg(args::final_indices_stick_size);
    constexpr uint32_t ids_per_batch = get_arg(args::ids_per_batch);
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr uint32_t num_users = get_arg(args::num_users);
    const uint32_t core_id = get_arg(args::core_id);  // per-node RTA

    constexpr uint32_t cb_id_out = dfb::cb_out;
    constexpr uint32_t cb_id_mask = dfb::cb_mask;
    constexpr uint32_t scaler_max_cb_id = dfb::scaler_max;
    constexpr uint32_t scaler_sum_cb_id = dfb::scaler_sum;
    constexpr uint32_t output_final_indices_rm_cb_index = dfb::final_indices_rm;
    constexpr uint32_t output_local_values_cb_index = dfb::local_vals;
    constexpr uint32_t output_local_indices_cb_index = dfb::local_indices;
    constexpr uint32_t rand_tile_index = dfb::rand_tile;
    constexpr uint32_t cb_id_k = dfb::cb_k;
    constexpr uint32_t cb_id_p = dfb::cb_p;
    constexpr uint32_t cb_id_temp = dfb::cb_temp;

    constexpr uint32_t k_chunk_size = num_cores * sizeof(uint32_t);
    constexpr uint32_t p_chunk_size = num_cores * sizeof(uint16_t);
    constexpr uint32_t temp_chunk_size = num_cores * sizeof(uint16_t);

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
    DataflowBuffer cb_out(cb_id_out);

    const auto addrg_k = TensorAccessor(ta::k_tensor);
    cb_k.reserve_back(1);
    uint32_t cb_id_k_ptr = cb_k.get_write_ptr();
    noc.async_read(addrg_k, cb_k, k_chunk_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_k.push_back(1);
    CoreLocalMem<volatile uint32_t> k_ptr(cb_id_k_ptr);
    uint32_t k = k_ptr[core_id];

    const auto addrg_p = TensorAccessor(ta::p_tensor);
    cb_p.reserve_back(1);
    uint32_t cb_id_p_ptr = cb_p.get_write_ptr();
    noc.async_read(addrg_p, cb_p, p_chunk_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_p.push_back(1);
    CoreLocalMem<volatile uint16_t> p_ptr(cb_id_p_ptr);
    uint32_t p = p_ptr[core_id];

    const auto addrg_temp = TensorAccessor(ta::temp);
    uint32_t cb_id_temp_ptr = cb_temp.get_write_ptr();
    noc.async_read(addrg_temp, cb_temp, temp_chunk_size, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    CoreLocalMem<volatile uint16_t> temp_ptr(cb_id_temp_ptr);
    uint16_t temp = temp_ptr[core_id];
    uint32_t temp_packed = (static_cast<uint32_t>(temp) << 16) + static_cast<uint32_t>(temp);
    generate_bcast_unary_scalar(cb_id_temp, temp_packed);

    constexpr uint32_t one = 1;
    generate_mask<cb_id_mask, one>(one, ids_per_batch / 32, k - 1);

    cb_rand.wait_front(1);
    CoreLocalMem<volatile uint16_t> rand_values(cb_rand.get_read_ptr());
    uint16_t rand = rand_values[0];

    // cb_final_indices.wait_front(32);
    cb_final_indices.wait_front(num_users);
    cb_local_values.wait_front(1);
    cb_local_indices.wait_front(1);

    CoreLocalMem<volatile uint16_t> local_values(cb_local_values.get_read_ptr());
    CoreLocalMem<volatile uint32_t> local_indices(cb_local_indices.get_read_ptr());
    CoreLocalMem<volatile uint32_t> final_indices(cb_final_indices.get_read_ptr() + core_id * final_indices_stick_size);

    uint32_t out_addr = cb_out.get_write_ptr();
    CoreLocalMem<volatile uint32_t> index_out(out_addr);

    uint32_t start_id_local_phase_0 = core_id * FACE_WIDTH;
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
    uint32_t top_p_cutoff = end_id_local_phase_1;
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
            cum_prob_f += bf16_to_f32(local_values[i]);
            if (cum_prob_f > p_f) {
                top_p_cutoff = i + 1;
                kept_tokens += top_p_cutoff - start_id_local_phase_1;
                cutoff_found_in_phase_1 = true;
                break;
            }
        }
    }
    if (cutoff_found_in_phase_0) {
        start_id_local_phase_1 = end_id_local_phase_0;
        end_id_local_phase_1 = end_id_local_phase_0;
        end_id_local_phase_0 = start_id_local_phase_0 + kept_tokens;
    } else if (cutoff_found_in_phase_1) {
        end_id_local_phase_1 = start_id_local_phase_1 + (kept_tokens - FACE_WIDTH);
    }

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

    cb_rand.pop_front(1);
    cb_local_values.pop_front(1);
    cb_local_indices.pop_front(1);
    // cb_final_indices.pop_front(32);
    cb_final_indices.pop_front(num_users);

    const auto s_out = TensorAccessor(ta::output);
    // `use<CircularBuffer::AddrSelector::WRITE_PTR>` takes a CircularBuffer&; construct a
    // temporary CircularBuffer over the same backing CB id (dfb::cb_out → uint32_t) for
    // this single call site.
    CircularBuffer cb_out_legacy(dfb::cb_out);
    noc.async_write(
        use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out_legacy),
        s_out,
        4,
        {.offset_bytes = core_id * 4},
        {.page_id = 0, .offset_bytes = core_id * 4});
    noc.async_write_barrier();
}
