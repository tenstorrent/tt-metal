// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/bfloat16.h"
#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
/* This kernel does:
Top-p Cumulative Probability Filtering:
Iteratively accumulates probabilities, comparing them against the nucleus threshold p to determine the smallest set of
tokens satisfying cumulative probabilty > p condition.

Top-k Sampling:
Samples from the top-k subset by comparing cumulative sums of probabilities with a random threshold to select the
appropriate index.
*/

constexpr uint32_t FACE_WIDTH = 16;
constexpr uint32_t FACE_HEIGHT = 16;

uint16_t bfloat16_add(uint16_t bf16_a, uint16_t bf16_b) {
    // Extract the sign, exponent, and mantissa from both values
    uint16_t sign_a = bf16_a & 0x8000;
    uint16_t sign_b = bf16_b & 0x8000;
    int16_t exp_a = (bf16_a & 0x7F80) >> 7;
    int16_t exp_b = (bf16_b & 0x7F80) >> 7;
    uint16_t mant_a = (bf16_a & 0x007F) | 0x0080;  // Add implicit leading 1
    uint16_t mant_b = (bf16_b & 0x007F) | 0x0080;  // Add implicit leading 1

    // Handle subnormal numbers (exponent is zero)
    if (exp_a == 0) {
        mant_a &= 0x007F;  // Remove implicit leading 1
    }
    if (exp_b == 0) {
        mant_b &= 0x007F;
    }

    // Align the mantissas by shifting the smaller one
    if (exp_a > exp_b) {
        mant_b >>= (exp_a - exp_b);
    } else if (exp_b > exp_a) {
        mant_a >>= (exp_b - exp_a);
        exp_a = exp_b;
    }

    // Add or subtract mantissas based on signs
    uint16_t mant_res;
    uint16_t sign_res;
    if (sign_a == sign_b) {
        mant_res = mant_a + mant_b;
        sign_res = sign_a;  // Result keeps the same sign
    } else {
        if (mant_a >= mant_b) {
            mant_res = mant_a - mant_b;
            sign_res = sign_a;  // Result keeps the sign of the larger magnitude
        } else {
            mant_res = mant_b - mant_a;
            sign_res = sign_b;
        }
    }

    // Handle zero result
    if (mant_res == 0) {
        return 0;
    }

    // Normalize the result
    if (mant_res & 0x0100) {  // Mantissa overflow
        mant_res >>= 1;
        exp_a += 1;
    }
    while (mant_res && !(mant_res & 0x0080)) {  // Normalize mantissa (shift left)
        mant_res <<= 1;
        exp_a -= 1;
    }

    // Handle exponent overflow and underflow
    if (exp_a >= 0xFF) {  // Overflow to infinity
        return sign_res | 0x7F80;
    }
    if (exp_a <= 0) {              // Underflow to zero or subnormal
        mant_res >>= (1 - exp_a);  // Shift mantissa to make exponent zero
        exp_a = 0;
    }

    // Combine the result
    uint16_t result = sign_res | (exp_a << 7) | (mant_res & 0x007F);
    return result;
}

uint16_t bfloat16_div(uint16_t bf16_a, uint16_t bf16_b) {
    // Extract sign, exponent, mantissa
    uint16_t sign_a = bf16_a & 0x8000;
    uint16_t sign_b = bf16_b & 0x8000;
    int16_t exp_a = (bf16_a & 0x7F80) >> 7;
    int16_t exp_b = (bf16_b & 0x7F80) >> 7;
    uint16_t mant_a = bf16_a & 0x007F;
    uint16_t mant_b = bf16_b & 0x007F;

    // Handle special cases (NaN, inf, zero)
    int a_is_nan = (exp_a == 0xFF) && (mant_a != 0);
    int b_is_nan = (exp_b == 0xFF) && (mant_b != 0);
    int a_is_inf = (exp_a == 0xFF) && (mant_a == 0);
    int b_is_inf = (exp_b == 0xFF) && (mant_b == 0);
    int a_is_zero = (exp_a == 0) && (mant_a == 0);
    int b_is_zero = (exp_b == 0) && (mant_b == 0);

    // NaN propagation
    if (a_is_nan) {
        return bf16_a;
    }
    if (b_is_nan) {
        return bf16_b | 0x0040;  // Make sure it's a quiet NaN
    }

    // Inf/0 rules
    if (a_is_inf && b_is_inf) {
        return 0x7FC0;  // NaN
    }
    if (a_is_inf) {
        return (sign_a ^ sign_b) | 0x7F80;  // Inf with correct sign
    }
    if (b_is_inf) {
        return (sign_a ^ sign_b);  // Zero with correct sign
    }
    if (a_is_zero && b_is_zero) {
        return 0x7FC0;  // NaN
    }
    if (a_is_zero) {
        return (sign_a ^ sign_b);  // Zero
    }
    if (b_is_zero) {
        return (sign_a ^ sign_b) | 0x7F80;  // Inf
    }

    // Handle subnormal numbers (normalize them first)
    if (exp_a == 0) {
        // Subnormal number - normalize it
        exp_a = 1;  // Start with minimum normal exponent
        while (mant_a && !(mant_a & 0x80)) {
            mant_a <<= 1;
            exp_a -= 1;
        }
        // If mant_a became 0, it was actually zero (handled above)
        // Otherwise, mant_a now has implicit leading 1 in bit 7
        if (mant_a) {
            mant_a &= 0x7F;  // Remove the implicit leading 1
        }
    } else {
        // Normal number - add implicit leading 1
        mant_a = mant_a | 0x80;
    }

    if (exp_b == 0) {
        // Subnormal number - normalize it
        exp_b = 1;  // Start with minimum normal exponent
        while (mant_b && !(mant_b & 0x80)) {
            mant_b <<= 1;
            exp_b -= 1;
        }
        // If mant_b became 0, it was actually zero (handled above)
        if (mant_b) {
            mant_b &= 0x7F;  // Remove the implicit leading 1
        }
    } else {
        // Normal number - add implicit leading 1
        mant_b = mant_b | 0x80;
    }

    // Result sign
    uint16_t sign_res = sign_a ^ sign_b;

    // Result exponent (subtract exponents, add bias)
    int16_t exp_res = exp_a - exp_b + 127;

    // Division of mantissas
    // Use 16 bits for numerator to preserve precision
    uint32_t mant_res = ((uint32_t)mant_a << 7) / mant_b;

    // Normalize result mantissa
    // The result should have leading 1 in bit 7 position
    while (mant_res && !(mant_res & 0x80)) {
        mant_res <<= 1;
        exp_res -= 1;
    }

    // Handle exponent overflow
    if (exp_res >= 0xFF) {
        return sign_res | 0x7F80;  // Infinity
    }

    // Handle exponent underflow to subnormal or zero
    if (exp_res <= 0) {
        if (exp_res < -7) {
            // Too small, return zero
            return sign_res;
        }
        // Create subnormal result
        // Shift mantissa right by (1 - exp_res) positions
        mant_res >>= (1 - exp_res);
        exp_res = 0;
    } else {
        // Normal result - remove implicit leading 1
        mant_res &= 0x7F;
    }

    // Combine the result
    uint16_t result = sign_res | (exp_res << 7) | (mant_res & 0x7F);
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

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_mask = get_compile_time_arg_val(5);
    constexpr uint32_t scale_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t packed_identity_scalar = get_compile_time_arg_val(7);
    constexpr uint32_t output_final_indices_rm_cb_index = get_compile_time_arg_val(8);
    constexpr uint32_t output_local_values_cb_index = get_compile_time_arg_val(9);
    constexpr uint32_t output_local_indices_cb_index = get_compile_time_arg_val(10);
    constexpr uint32_t final_indices_stick_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_stick_size = get_compile_time_arg_val(12);
    constexpr uint32_t rand_tile_index = get_compile_time_arg_val(13);
    constexpr uint32_t cb_id_k = get_compile_time_arg_val(14);
    constexpr uint32_t cb_id_p = get_compile_time_arg_val(15);
    constexpr uint32_t cb_id_temp = get_compile_time_arg_val(16);
    constexpr uint32_t core_id = get_compile_time_arg_val(17);
    constexpr uint32_t ids_per_batch = get_compile_time_arg_val(18);
    constexpr uint32_t num_cores = get_compile_time_arg_val(19);
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

    uint16_t bf16_p = static_cast<uint16_t>(p & 0xFFFF);
    uint32_t cum_prob = 0;
    uint32_t kept_tokens = 0;
    bool cutoff_found_in_phase_0 = false;
    bool cutoff_found_in_phase_1 = false;
    uint32_t top_p_cutoff = end_id_local_phase_1;  // Default to all tokens
    for (uint32_t i = start_id_local_phase_0; i < end_id_local_phase_0; ++i) {
        cum_prob = bfloat16_add(cum_prob, local_values[i]);
        if (bfloat16_greater(cum_prob, bf16_p)) {
            top_p_cutoff = i + 1;  // Include this token in the top-p set
            cutoff_found_in_phase_0 = true;
            kept_tokens = top_p_cutoff - start_id_local_phase_0;
            break;
        }
    }
    if (!cutoff_found_in_phase_0) {
        kept_tokens = FACE_WIDTH;
        for (uint32_t i = start_id_local_phase_1; i < end_id_local_phase_1; ++i) {
            // cum sum of local values
            cum_prob = bfloat16_add(cum_prob, local_values[i]);
            if (bfloat16_greater(cum_prob, bf16_p)) {
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

    uint32_t cum_sum = 0;
    index_out[core_id] = final_indices[local_indices[start_id_local_phase_0]];
    bool index_found = false;

    // Sample from the top-k values
    for (uint32_t i = start_id_local_phase_0; i < end_id_local_phase_0; ++i) {
        // cum sum of local values
        cum_sum = bfloat16_add(cum_sum, bfloat16_div(local_values[i], cum_prob));
        if (bfloat16_greater(cum_sum, rand)) {
            index_out[core_id] = final_indices[local_indices[i]];
            index_found = true;
            break;
        }
    }
    if (!index_found) {
        for (uint32_t i = start_id_local_phase_1; i < end_id_local_phase_1; ++i) {
            // cum sum of local values
            cum_sum = bfloat16_add(cum_sum, bfloat16_div(local_values[i], cum_prob));
            if (bfloat16_greater(cum_sum, rand)) {
                index_out[core_id] = final_indices[local_indices[i]];
                index_found = true;
                break;
            }
        }
    }

    const auto s_out = TensorAccessor(dst_args, dst_addr, out_stick_size);
    uint64_t dst_noc_addr = get_noc_addr(0, s_out);
    // Write individual core result - output buffer should handle alignment
    noc_async_write(out_addr + core_id * 4, dst_noc_addr + core_id * 4, 4);
    noc_async_write_barrier();
}
