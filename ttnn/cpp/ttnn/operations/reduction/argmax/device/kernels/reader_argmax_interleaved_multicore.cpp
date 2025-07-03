// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>

#include "dataflow_api.h"
#include "utils/bfloat16.h"

/**
 * @brief Process inner dimension units for argmax reduction
 *
 * This function handles the core logic of processing inner dimension units,
 * including reading data, finding max values and indices, and storing results.
 *
 * @tparam src_is_dram Whether source is in DRAM
 * @tparam reduce_all Whether to reduce across all dimensions
 * @param inner_dim_units Number of units in the inner dimension
 * @param outer_idx Current outer dimension index
 * @param src_offset Source offset for reading data
 * @param s_src Source stream descriptor
 * @param src_cb_addr Source circular buffer address
 * @param src_read_size Size of data to read from source
 * @param red_dim_offset Reduction dimension offset
 * @param red_dim_units_this_core Number of reduction units for this core
 * @param in_vals Input values buffer
 * @param red_dim_units Total reduction dimension units
 * @param max_idx Maximum index found (output parameter)
 * @param max_val Maximum value found (output parameter)
 * @param red_idxs Buffer for reduction indices
 * @param red_vals Buffer for reduction values
 */
template <bool src_is_dram, bool reduce_all>
inline void find_argmax_for_core(
    const uint32_t inner_dim_units,
    const uint32_t outer_idx,
    const uint32_t src_offset,
    const InterleavedAddrGen<src_is_dram>& s_src,
    const uint32_t src_cb_addr,
    const uint32_t src_read_size,
    const uint32_t red_dim_offset,
    const uint32_t red_dim_units_this_core,
    volatile tt_l1_ptr uint16_t* in_vals,
    const uint32_t red_dim_units,
    uint32_t& max_idx,
    uint16_t& max_val,
    volatile tt_l1_ptr uint32_t* red_idxs,
    volatile tt_l1_ptr uint16_t* red_vals) {
    for (uint32_t j = 0; j < inner_dim_units; ++j) {
        noc_async_read(s_src.get_noc_addr(outer_idx * inner_dim_units + j, src_offset), src_cb_addr, src_read_size);
        noc_async_read_barrier();

        // Reset max_val for each new output
        if constexpr (not reduce_all) {
            max_idx = 0;
            max_val = NEG_INF_BFLOAT16;
        }

        for (uint32_t i = red_dim_offset; i < (red_dim_offset + red_dim_units_this_core); ++i) {
            uint16_t val = in_vals[i - red_dim_offset];
            if (bfloat16_greater(val, max_val)) {
                auto full_idx = outer_idx * inner_dim_units * red_dim_units + j * red_dim_units + i;
                max_idx = reduce_all ? full_idx : i;
                max_val = val;
            } else if (val == max_val) {
                auto full_idx = outer_idx * inner_dim_units * red_dim_units + j * red_dim_units + i;
                max_idx = reduce_all ? std::min(max_idx, full_idx) : std::min(max_idx, i);
            }
        }

        if constexpr (not reduce_all) {
            red_idxs[j] = max_idx;
            red_vals[j] = max_val;
        }
    }
}

/**
 * @brief Finds the argmax from intermediate outputs across all cores
 *
 * This function processes the intermediate outputs from all cores to find the final
 * argmax values. It compares values from each core and selects the maximum value,
 * with ties broken by selecting the smaller index.
 *
 * @tparam num_cores Total number of cores participating in the reduction
 * @param inner_idx Index of the inner dimension unit to find argmax for
 * @param red_val_cb_local_base_addr Base address of the circular buffer storing intermediate values
 * @param red_idx_cb_local_base_addr Base address of the circular buffer storing intermediate indices
 * @param red_val_size_per_core Size of the value buffer per core
 * @param red_idx_size_per_core Size of the index buffer per core
 * @param inner_dim_units Number of elements in the inner dimension
 */
template <uint32_t num_cores>
inline uint32_t find_argmax_from_intermediate_outputs(
    const uint32_t inner_idx,
    const uint32_t red_val_cb_local_base_addr,
    const uint32_t red_idx_cb_local_base_addr,
    const uint32_t red_val_size_per_core,
    const uint32_t red_idx_size_per_core) {
    // Reset max_val for each new output
    uint16_t max_val = NEG_INF_BFLOAT16;
    uint32_t max_idx = 0;

    for (uint32_t i = 0; i < num_cores; ++i) {
        volatile tt_l1_ptr auto i_red_vals =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(red_val_cb_local_base_addr + i * red_val_size_per_core);
        volatile tt_l1_ptr auto i_red_idxs =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_idx_cb_local_base_addr + i * red_idx_size_per_core);

        uint16_t val = i_red_vals[inner_idx];

        if (bfloat16_greater(val, max_val)) {
            max_idx = i_red_idxs[inner_idx];
            max_val = val;
        } else if ((val == max_val) && (i_red_idxs[inner_idx] < max_idx)) {
            max_idx = i_red_idxs[inner_idx];
        }
    }
    return max_idx;
}

void kernel_main() {
    // Runtime args
    // ------------
    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_id = get_arg_val<uint32_t>(2);

    // The offset of the input CB in the input tensor
    const uint32_t src_offset = get_arg_val<uint32_t>(3);

    // The offset of the reduction dim in the input CB
    const uint32_t red_dim_offset = get_arg_val<uint32_t>(4);

    // The bytes to read from the input CB for this core
    const uint32_t src_read_size = get_arg_val<uint32_t>(5);

    // This is the number of elements in the reduction dim processed by this core
    const uint32_t red_dim_units_this_core = get_arg_val<uint32_t>(6);

    // Compile time args
    // -----------------
    constexpr uint32_t src_cb_idx = get_compile_time_arg_val(0);

    // This CB is only used in the reduction core. It is used to store
    // final outputs (indices) after reduction of intermediate outputs.
    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(1);

    // This CB holds intermediate outputs (indices) in each core
    constexpr uint32_t red_idxs_cb_idx = get_compile_time_arg_val(2);

    // This CB holds intermediate outputs (values) in each core
    constexpr uint32_t red_vals_cb_idx = get_compile_time_arg_val(3);

    constexpr bool src_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(5);
    constexpr uint32_t src_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t dst_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t red_idx_size_per_core = get_compile_time_arg_val(8);
    constexpr uint32_t red_val_size_per_core = get_compile_time_arg_val(9);

    // This is the number of elements in the output, excluding the last two dimensions.
    // i.e. for an input tensor of shape (.., N, C, H, W), this is (.. * N * C)
    // It also depends on the `keepdim`
    constexpr uint32_t outer_dim_units = get_compile_time_arg_val(10);

    // This is the number of elements in the last dimension of the output
    // i.e. for an input tensor of shape (.., N, C, H, W), this is H.
    // This dictates the page size in the output cb
    constexpr uint32_t inner_dim_units = get_compile_time_arg_val(11);

    // This is the number of elements in the input tensor along the reduction dim (W)
    constexpr uint32_t red_dim_units = get_compile_time_arg_val(12);

    // Boolean to indicate if we reduce across _all_ dimensions or just on the reduction dim (last dim)
    constexpr bool reduce_all = (bool)get_compile_time_arg_val(13);

    // Total number of cores participating in this op
    constexpr uint32_t num_cores = get_compile_time_arg_val(14);

    // Pick the core that will collate the intermediate outputs
    constexpr uint32_t reduce_core_id = (bool)get_compile_time_arg_val(15);

    constexpr uint32_t reduce_core_x = get_compile_time_arg_val(16);
    constexpr uint32_t reduce_core_y = get_compile_time_arg_val(17);

    // start and end coordinates of the cores that will be used to compute the intermediate outputs
    // At maximum, there can be two groups of cores (suffix 0 and 1)
    constexpr uint32_t start_core_x0 = get_compile_time_arg_val(18);
    constexpr uint32_t start_core_y0 = get_compile_time_arg_val(19);
    constexpr uint32_t end_core_x0 = get_compile_time_arg_val(20);
    constexpr uint32_t end_core_y0 = get_compile_time_arg_val(21);

    constexpr uint32_t start_core_x1 = get_compile_time_arg_val(22);
    constexpr uint32_t start_core_y1 = get_compile_time_arg_val(23);
    constexpr uint32_t end_core_x1 = get_compile_time_arg_val(24);
    constexpr uint32_t end_core_y1 = get_compile_time_arg_val(25);

    constexpr uint32_t num_cores0 = get_compile_time_arg_val(26);  // Number of cores in group 0
    constexpr uint32_t num_cores1 = get_compile_time_arg_val(27);  // Number of cores in group 1

    // Semaphore to fire when intermediate outputs can be started to compute
    constexpr uint32_t start_sem_idx = get_compile_time_arg_val(28);

    // Semaphore to fire when intermediate outputs for one page are ready
    constexpr uint32_t done_sem_idx = get_compile_time_arg_val(29);

    //-------------------------------------------------------------------------
    // Flag to identify if this core will collate intermediate outputs
    const bool is_reduce_core = (core_id == reduce_core_id);

    const InterleavedAddrGen<src_is_dram> s_src = {.bank_base_address = src_base_addr, .page_size = src_page_size};
    const InterleavedAddrGen<dst_is_dram> s_dst = {.bank_base_address = dst_base_addr, .page_size = dst_page_size};

    // CB in L1 memory for storing input
    const uint32_t src_cb_addr = get_write_ptr(src_cb_idx);
    volatile tt_l1_ptr uint16_t* in_vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_cb_addr);

    // CB in L1 memory of reducer core for storing output
    const uint32_t dst_cb_addr = get_write_ptr(dst_cb_idx);
    volatile tt_l1_ptr uint32_t* out_idxs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_cb_addr);

    // CB in L1 memory for storing partial idx output
    const uint32_t red_idx_cb_local_base_addr = get_write_ptr(red_idxs_cb_idx);
    const uint32_t red_idx_offset = core_id * red_idx_size_per_core;

    const uint32_t red_idx_cb_local_addr = red_idx_cb_local_base_addr + red_idx_offset;
    volatile tt_l1_ptr uint32_t* red_idxs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_idx_cb_local_addr);

    const uint64_t red_idx_noc_addr = get_noc_addr(reduce_core_x, reduce_core_y, red_idx_cb_local_addr);

    // CB in L1 memory for storing partial val output
    const uint32_t red_val_cb_local_base_addr = get_write_ptr(red_vals_cb_idx);
    const uint32_t red_val_offset = core_id * red_val_size_per_core;

    const uint32_t red_val_cb_local_addr = red_val_cb_local_base_addr + red_val_offset;
    volatile tt_l1_ptr uint16_t* red_vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(red_val_cb_local_addr);

    const uint64_t red_val_noc_addr = get_noc_addr(reduce_core_x, reduce_core_y, red_val_cb_local_addr);

    // Semaphore addresses
    const uint32_t start_sem_local_addr = get_semaphore(start_sem_idx);
    uint64_t start_sem_noc_addr0 =
        get_noc_multicast_addr(start_core_x0, start_core_y0, end_core_x0, end_core_y0, start_sem_local_addr);
    uint64_t start_sem_noc_addr1 =
        get_noc_multicast_addr(start_core_x1, start_core_y1, end_core_x1, end_core_y1, start_sem_local_addr);
    volatile tt_l1_ptr uint32_t* start_sem_local_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_sem_local_addr);

    const uint32_t done_sem_local_addr = get_semaphore(done_sem_idx);
    const uint64_t done_sem_noc_addr = get_noc_addr(reduce_core_x, reduce_core_y, done_sem_local_addr);
    volatile tt_l1_ptr uint32_t* done_sem_local_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem_local_addr);

    uint32_t max_idx = 0;
    uint16_t max_val = NEG_INF_BFLOAT16;
    noc_semaphore_set(done_sem_local_ptr, 0);

    // -------------------------------------------------------------------------
    // Main loop - run by all cores
    for (uint32_t k = 0; k < outer_dim_units; ++k) {
        if (is_reduce_core) {
            noc_semaphore_set(done_sem_local_ptr, 0);
            noc_semaphore_set(start_sem_local_ptr, k + 1);

            if constexpr (num_cores > 1) {
                if (k > 0) {
                    noc_semaphore_set_multicast_loopback_src(start_sem_local_addr, start_sem_noc_addr0, num_cores0);
                }

                if (num_cores1 > 0) {
                    noc_semaphore_set_multicast(start_sem_local_addr, start_sem_noc_addr1, num_cores1);
                }

                noc_async_atomic_barrier();
            }
        }

        // Wait to start
        if (k > 0) {
            noc_semaphore_wait(start_sem_local_ptr, k + 1);
        }

        find_argmax_for_core<src_is_dram, reduce_all>(
            inner_dim_units,
            k,
            src_offset,
            s_src,
            src_cb_addr,
            src_read_size,
            red_dim_offset,
            red_dim_units_this_core,
            in_vals,
            red_dim_units,
            max_idx,
            max_val,
            red_idxs,
            red_vals);

        if constexpr (not reduce_all) {
            // We now write these local values to the equivalent position in the reduction core
            if (core_id != reduce_core_id) {
                noc_async_write(red_idx_cb_local_addr, red_idx_noc_addr, red_idx_size_per_core);
                noc_async_write(red_val_cb_local_addr, red_val_noc_addr, red_val_size_per_core);
                noc_async_write_barrier();
            }

            noc_semaphore_inc(done_sem_noc_addr, 1);
            noc_async_atomic_barrier();

            // If this is the reducer core, wait for the semaphore to be set
            if (is_reduce_core) {
                if constexpr (num_cores > 1) {
                    noc_semaphore_wait(done_sem_local_ptr, num_cores);
                }

                // Find argmax from intermediate outputs and write to output
                for (uint32_t j = 0; j < inner_dim_units; ++j) {
                    out_idxs[j] = find_argmax_from_intermediate_outputs<num_cores>(
                        j,
                        red_val_cb_local_base_addr,
                        red_idx_cb_local_base_addr,
                        red_val_size_per_core,
                        red_idx_size_per_core);
                }

                const uint64_t dst_noc_addr = get_noc_addr(k, s_dst);
                noc_async_write(dst_cb_addr, dst_noc_addr, dst_page_size);
                noc_async_write_barrier();
            }
        }
    }  // for (uint32_t k = 0; k < outer_dim_units; ++k)

    if constexpr (reduce_all) {
        red_idxs[0] = max_idx;
        red_vals[0] = max_val;

        // We now write these local values to the equivalent position in the reduction core
        if (core_id != reduce_core_id) {
            noc_async_write(red_idx_cb_local_addr, red_idx_noc_addr, red_idx_size_per_core);
            noc_async_write(red_val_cb_local_addr, red_val_noc_addr, red_val_size_per_core);
            noc_async_write_barrier();
        }

        noc_semaphore_inc(done_sem_noc_addr, 1);
        noc_async_atomic_barrier();

        // If this is the reducer core, wait for the semaphore to be set
        if (is_reduce_core) {
            if constexpr (num_cores > 1) {
                noc_semaphore_wait(done_sem_local_ptr, num_cores);
            }

            out_idxs[0] = find_argmax_from_intermediate_outputs<num_cores>(
                0,  // For reduce_all, we only have one inner_dim_unit
                red_val_cb_local_base_addr,
                red_idx_cb_local_base_addr,
                red_val_size_per_core,
                red_idx_size_per_core);

            const uint64_t dst_noc_addr = get_noc_addr(0, s_dst);
            noc_async_write(dst_cb_addr, dst_noc_addr, dst_page_size);
            noc_async_write_barrier();
        }
    }  // if constexpr (reduce_all)
}
