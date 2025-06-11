// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cinttypes>
#include <cstdint>

#include "compile_time_args.h"
#include "dataflow_api.h"

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "debug/dprint_pages.h"
#endif

// Fill an L1 buffer with the given val
inline bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
    return true;
}

template <
    uint32_t stick_nbytes,
    uint32_t input_aligned_page_size,
    bool is_block_sharded,
    bool is_width_sharded,
    bool is_col_major,
    bool main_thread>
void copy_sticks_async_to_temp(
    const tt_l1_ptr uint16_t* config_data,
    const uint16_t my_noc_x,
    const uint16_t my_noc_y,
    const uint32_t in_base_l1_addr,
    const uint32_t temp_base_l1_addr,
    const uint32_t out_base_l1_addr) {
    uint32_t i = 0;
    uint32_t length = config_data[2];
    uint32_t remote_entry_count = 0;

    constexpr bool noc_orient_x = ((is_block_sharded && !is_col_major) || is_width_sharded);
    constexpr bool noc_orient_y = ((is_block_sharded && is_col_major) || is_width_sharded);

    const uint64_t base_addr_temp = get_noc_addr(my_noc_x, my_noc_y, temp_base_l1_addr);
    uint64_t dst_addr_temp = base_addr_temp;
    while (length) {
        const uint16_t noc_x = noc_orient_x ? my_noc_x : config_data[i + 0];
        const uint16_t noc_y = noc_orient_y ? my_noc_y : config_data[i + 1];
        const uint64_t base_addr_final = get_noc_addr(noc_x, noc_y, out_base_l1_addr);
        length = config_data[i + 2];
        i += 3;
        for (uint16_t j = 0; j < length; j += 3) {
            const uint16_t nsticks = config_data[i + j + 2];
            const uint32_t size = nsticks * stick_nbytes;
            if (remote_entry_count % 2 != main_thread) {
                const uint16_t src_local_idx = config_data[i + j + 0];
                const uint16_t dst_local_idx = config_data[i + j + 1];
                const uint32_t src_offset = src_local_idx * input_aligned_page_size;
                const uint32_t dst_offset = dst_local_idx * stick_nbytes;
                uint32_t src_addr = in_base_l1_addr + src_offset;
                const uint64_t dst_addr_final = base_addr_final + dst_offset;
                if constexpr (stick_nbytes == input_aligned_page_size) {
                    noc_async_write(src_addr, dst_addr_temp, size);
                    dst_addr_temp +=
                        size;  // remote sticks from each config entry are written contiguously into the temp buffer
                } else {
                    for (uint16_t k = 0; k < nsticks; k++) {
                        noc_async_write(src_addr, dst_addr_temp, stick_nbytes);
                        dst_addr_temp += stick_nbytes;
                        src_addr += input_aligned_page_size;
                    }
                }
            } else {
                dst_addr_temp += size;  // increment space in the temp buffer for the other data movement core
            }

            remote_entry_count++;
        }

        i += length;
    }
}

template <
    uint32_t stick_nbytes,
    uint32_t input_aligned_page_size,
    bool is_block_sharded,
    bool is_width_sharded,
    bool is_col_major,
    bool main_thread,
    bool padding_exists>
void copy_sticks_async_from_temp(
    const tt_l1_ptr uint16_t* config_data,
    const uint16_t my_noc_x,
    const uint16_t my_noc_y,
    const uint32_t temp_base_l1_addr,
    const uint32_t out_base_l1_addr) {
    if constexpr (!main_thread && padding_exists) {
        return;
    }

    uint32_t i = 0;
    uint32_t length = config_data[2];
    uint32_t remote_entry_count = 0;

    uint64_t src_addr = temp_base_l1_addr;

    constexpr bool noc_orient_x = ((is_block_sharded && !is_col_major) || is_width_sharded);
    constexpr bool noc_orient_y = ((is_block_sharded && is_col_major) || is_width_sharded);

    while (length) {
        const uint16_t noc_x = noc_orient_x ? my_noc_x : config_data[i + 0];
        const uint16_t noc_y = noc_orient_y ? my_noc_y : config_data[i + 1];
        const uint64_t base_addr = get_noc_addr(noc_x, noc_y, out_base_l1_addr);
        length = config_data[i + 2];
        i += 3;
        for (uint16_t j = 0; j < length; j += 3) {
            const uint16_t nsticks = config_data[i + j + 2];
            const uint32_t size = nsticks * stick_nbytes;

            if ((remote_entry_count % 2 != main_thread) || (main_thread && padding_exists)) {
                const uint16_t dst_local_idx = config_data[i + j + 1];
                const uint32_t dst_offset = dst_local_idx * stick_nbytes;
                uint64_t dst_addr = base_addr + dst_offset;
                if constexpr (stick_nbytes == input_aligned_page_size) {
                    noc_async_write(src_addr, dst_addr, size);
                    src_addr +=
                        size;  // remote sticks from each config entry are read contiguously from the temp buffer
                } else {
                    for (uint16_t k = 0; k < nsticks; k++) {
                        noc_async_write(src_addr, dst_addr, stick_nbytes);
                        dst_addr += stick_nbytes;
                        src_addr += stick_nbytes;
                    }
                }
            } else {
                src_addr += size;  // increment space in the temp buffer for the other data movement core
            }

            remote_entry_count++;
        }

        i += length;
    }
}

template <uint32_t stick_nbytes, uint32_t input_aligned_page_size, uint32_t sync_cb_id, bool main_thread>
void copy_sticks_async_local(
    const tt_l1_ptr uint16_t* config_data,
    const uint16_t my_noc_x,
    const uint16_t my_noc_y,
    const uint32_t in_base_l1_addr,
    const uint32_t out_base_l1_addr,
    const uint32_t in_out_buffer_start_delta) {
    uint32_t i = 0;
    uint32_t length = config_data[2];

    const uint64_t base_addr = get_noc_addr(my_noc_x, my_noc_y, out_base_l1_addr);
    while (length) {
        length = config_data[i + 2];
        i += 3;
        for (uint16_t j = 0; j < length; j += 3) {
            const uint16_t src_local_idx = config_data[i + j + 0];
            const uint16_t dst_local_idx = config_data[i + j + 1];
            const uint16_t nsticks = config_data[i + j + 2];
            const uint32_t size = nsticks * stick_nbytes;
            const uint32_t dst_offset = dst_local_idx * stick_nbytes;
            const uint32_t src_offset = src_local_idx * input_aligned_page_size;

            const uint64_t dst_addr = base_addr + dst_offset;
            const uint32_t src_addr = in_base_l1_addr + src_offset;

            const uint32_t dst_relative_src = src_local_idx + in_out_buffer_start_delta;
            const bool is_not_overlap_copy =
                dst_local_idx + nsticks < dst_relative_src || dst_relative_src + nsticks < dst_local_idx;

            if (dst_relative_src == dst_local_idx) {
                continue;  // no need to copy, src and dst are the same
            }

            if constexpr (main_thread) {
                cb_reserve_back(sync_cb_id, 1);
                noc_async_write_barrier();  // wait for last local copy to finish to preseve order
                cb_push_back(sync_cb_id, 1);
            } else {
                cb_wait_front(sync_cb_id, 1);  // wait for main thread to be ready to copy
            }

            if (is_not_overlap_copy) {  // dst and src data do not overlap, can copy big chunks
                const uint32_t half_size = size / 2;
                if constexpr (main_thread) {
                    noc_async_write(src_addr, dst_addr, half_size);
                } else {
                    noc_async_write(src_addr + half_size, dst_addr + half_size, half_size);
                }
            } else {  // dst and src data overlaps, stick by stick copy is necessary, note front half and back half of
                      // stick copies are orthogonal so relative order doesn't matter
                const bool is_forward_copy = dst_local_idx > dst_relative_src;
                const uint32_t half_stick = stick_nbytes / 2;
                const uint32_t start_idx = is_forward_copy ? nsticks - 1 : 0;
                const uint32_t end_idx = is_forward_copy ? -1 : nsticks;
                const uint32_t step = is_forward_copy ? -1 : 1;
                for (uint32_t k = start_idx; end_idx != k; k += step) {
                    if constexpr (main_thread) {
                        noc_async_write(src_addr + k * stick_nbytes, dst_addr + k * stick_nbytes, half_stick);
                    } else {
                        noc_async_write(
                            src_addr + k * stick_nbytes + half_stick,
                            dst_addr + k * stick_nbytes + half_stick,
                            half_stick);
                    }
                }
            }

            if constexpr (!main_thread) {
                noc_async_write_barrier();
                cb_pop_front(sync_cb_id, 1);  // signal to main thread that secondary thread is ready to copy
            }
        }

        i += length;
    }
}

/*
In Place Halo Kernel Details:
For the in place version of halo the input and output buffers overlap,
thus it is necessary to order the local, remote and padding data
movement operations such that the local data originating in each shard
is not overwritten before being copied to all it's final destinations,
which may span several output shards. This is done with the following
steps:

1. (NC,BR) copy the remote sticks from "my" shard to temp buffer
2. (NC,BR) copy the local sticks from "my" shard to their destinations
3. (NC,BR) wait on semaphores for all cores to finish 1. and 2.
4. (NC (if padding)) write the padding sticks to their destinations
4. (BR, NC (if no padding)) copy the remote sticks from temp buffer to their destinations

Note that for remote copies, since the temp buffer is used the order of
the remote copies does not matter, thus we can alternate copies between
cores, but for the local copies the order does matter, thus we split each
copy across the two DM cores when possible.
*/

void kernel_main() {
    constexpr uint32_t main_thread = get_compile_time_arg_val(0);
    constexpr uint32_t padding_exists = get_compile_time_arg_val(1);
    constexpr uint32_t padding_config_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t local_config_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t remote_config_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t remote_temp_cb_id = get_compile_time_arg_val(5);  // temp buffer for in place halo
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(6);          // the innput shard buffer
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(7);           // either the input shard or untilize output
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(8);          // output shard with padding and halo goes here
    constexpr uint32_t pad_cb_id = get_compile_time_arg_val(9);          // cb for const pad val buffer
    constexpr uint32_t pad_val_u32 = get_compile_time_arg_val(10);       // pad value to fill pad buffer with
    constexpr uint32_t in_npages = get_compile_time_arg_val(11);         // number of sticks
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(12);      // stick size in bytes (post untilize)
    constexpr uint32_t is_block_sharded = get_compile_time_arg_val(13);
    constexpr bool is_col_major = get_compile_time_arg_val(14) == 1;
    constexpr uint32_t is_width_sharded = get_compile_time_arg_val(15);
    constexpr uint32_t input_aligned_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t remote_read = get_compile_time_arg_val(17);  // Unused parameter
    constexpr uint32_t num_active_cores = get_compile_time_arg_val(18);
    constexpr uint32_t noc_TL_x = get_compile_time_arg_val(19);
    constexpr uint32_t noc_TL_y = get_compile_time_arg_val(20);
    constexpr uint32_t noc_BR_x = get_compile_time_arg_val(21);
    constexpr uint32_t noc_BR_y = get_compile_time_arg_val(22);
    constexpr uint32_t rectangular_x = get_compile_time_arg_val(23);
    constexpr uint32_t rectangular_y = get_compile_time_arg_val(24);
    constexpr uint32_t last_active_x = get_compile_time_arg_val(25);
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(26);
    constexpr uint32_t in_out_buffer_start_delta = get_compile_time_arg_val(27);
    constexpr uint32_t untilize_temp_cb_id =
        get_compile_time_arg_val(28);  // temp buffer for in place untilize with wide tensors
    constexpr uint32_t tile_cols = get_compile_time_arg_val(29);
    constexpr uint32_t tile_rows = get_compile_time_arg_val(30);
    constexpr uint32_t sync_cb_id1 = get_compile_time_arg_val(31);
    constexpr uint32_t sync_cb_id2 = get_compile_time_arg_val(32);

    constexpr uint32_t elem_nbytes = sizeof(uint16_t);
    constexpr uint16_t pad_core_id = 0xFFFF;
    constexpr uint32_t TILE_SIZE_BYTES = get_tile_size(in_cb_id);

    const uint16_t my_noc_x = NOC_X(my_x[noc_index]);
    const uint16_t my_noc_y = NOC_Y(my_y[noc_index]);
    const uint32_t noop_core = get_arg_val<uint32_t>(0);
    const uint32_t cast_core = get_arg_val<uint32_t>(1);
    if (noop_core) {
        return;
    }

    const uint32_t in_base_l1_addr = get_read_ptr(in_cb_id);
    const uint32_t out_base_l1_addr = get_write_ptr(out_cb_id);
    const uint32_t untilize_temp_l1_addr = get_read_ptr(untilize_temp_cb_id);

    if constexpr (main_thread) {
        cb_reserve_back(src_cb_id, in_npages);
        cb_push_back(src_cb_id, in_npages);
    }

    // make sure untilized data is available
    // for wide tensors a temp CB must be used due to implementation of the untilize LLK function vs pack_untilize
    if (untilize_temp_cb_id && main_thread) {
        for (uint32_t i = 0; i < tile_rows; ++i) {
            cb_wait_front(untilize_temp_cb_id, tile_cols);
            cb_reserve_back(in_cb_id, tile_cols);

            const uint32_t in_l1_addr = get_write_ptr(in_cb_id);
            const uint64_t in_l1_noc_addr = get_noc_addr(my_noc_x, my_noc_y, in_l1_addr);
            noc_async_write(untilize_temp_l1_addr, in_l1_noc_addr, TILE_SIZE_BYTES * tile_cols);
            noc_async_write_barrier();

            cb_push_back(in_cb_id, tile_cols);
            cb_pop_front(untilize_temp_cb_id, tile_cols);
        }
    }
    cb_wait_front(in_cb_id, in_npages);

    // copy remote sticks to temp buffer
    const uint32_t temp_base_l1_addr = get_write_ptr(remote_temp_cb_id);
    const uint32_t remote_config_data_l1_addr = get_read_ptr(remote_config_cb_id);
    const tt_l1_ptr uint16_t* remote_config_data =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(remote_config_data_l1_addr);
    copy_sticks_async_to_temp<
        stick_nbytes,
        input_aligned_page_size,
        is_block_sharded,
        is_width_sharded,
        is_col_major,
        main_thread>(remote_config_data, my_noc_x, my_noc_y, in_base_l1_addr, temp_base_l1_addr, out_base_l1_addr);

    noc_async_write_barrier();
    if constexpr (main_thread) {
        cb_reserve_back(sync_cb_id1, 1);
        cb_push_back(sync_cb_id1, 1);
        cb_wait_front(sync_cb_id2, 1);
        cb_pop_front(sync_cb_id2, 1);
    } else {
        cb_reserve_back(sync_cb_id2, 1);
        cb_push_back(sync_cb_id2, 1);
        cb_wait_front(sync_cb_id1, 1);
        cb_pop_front(sync_cb_id1, 1);
    }

    // move local sticks
    const uint32_t local_config_data_l1_addr = get_read_ptr(local_config_cb_id);
    const tt_l1_ptr uint16_t* local_config_data =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(local_config_data_l1_addr);
    copy_sticks_async_local<stick_nbytes, input_aligned_page_size, sync_cb_id1, main_thread>(
        local_config_data, my_noc_x, my_noc_y, in_base_l1_addr, out_base_l1_addr, in_out_buffer_start_delta);

    const uint32_t semaphore_addr = get_semaphore(semaphore_id);
    const uint64_t semaphore_noc_addr = get_noc_addr(noc_TL_x, noc_TL_y, semaphore_addr);
    noc_async_write_barrier();
    if constexpr (main_thread) {
        cb_wait_front(sync_cb_id2, 1);
        cb_pop_front(sync_cb_id2, 1);

        // incremement the semaphore
        noc_semaphore_inc(semaphore_noc_addr, 1);
    } else {
        cb_reserve_back(sync_cb_id2, 1);
        cb_push_back(sync_cb_id2, 1);
    }

    // wait for all cores to finish copying local and remote data
    if (cast_core) {
        volatile tt_l1_ptr uint32_t* semaphore_noc_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_noc_addr);
        noc_semaphore_wait(semaphore_noc_addr_ptr, num_active_cores);

        const uint64_t mcast_noc_addr = get_noc_multicast_addr(noc_TL_x, noc_TL_y, noc_BR_x, noc_BR_y, semaphore_addr);

        noc_semaphore_set_multicast(semaphore_addr, mcast_noc_addr, rectangular_x * rectangular_y - 1);
    }

    // wait for multicast
    volatile tt_l1_ptr uint32_t* semaphore_noc_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_noc_addr);
    noc_semaphore_wait(semaphore_noc_addr_ptr, num_active_cores);

    // insert padding
    if constexpr (!main_thread && padding_exists) {
        // construct the pad stick in its buffer
        cb_reserve_back(pad_cb_id, 1);
        const uint16_t pad_val = pad_val_u32;
        fill_with_val(get_write_ptr(pad_cb_id), stick_nbytes / elem_nbytes, pad_val);
        cb_push_back(pad_cb_id, 1);

        const uint32_t padding_config_l1_addr = get_read_ptr(padding_config_cb_id);
        const volatile tt_l1_ptr uint16_t* config_data =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(padding_config_l1_addr);
        const uint64_t padding_l1_addr = get_noc_addr(my_noc_x, my_noc_y, get_read_ptr(pad_cb_id));
        const uint32_t dst_base_addr = out_base_l1_addr;
        uint16_t nsticks = 1;
        for (uint16_t j = 0; nsticks; j += 2) {
            const uint16_t dst_local_idx = config_data[j + 0];
            nsticks = config_data[j + 1];

            uint64_t dst_addr = dst_base_addr + dst_local_idx * stick_nbytes;
            for (uint16_t k = 0; k < nsticks; ++k) {
                noc_async_read(padding_l1_addr, dst_addr, stick_nbytes);
                dst_addr += stick_nbytes;
            }
        }
    }

    // copy remote sticks from temp buffer to final destinations
    copy_sticks_async_from_temp<
        stick_nbytes,
        input_aligned_page_size,
        is_block_sharded,
        is_width_sharded,
        is_col_major,
        main_thread,
        padding_exists>(remote_config_data, my_noc_x, my_noc_y, temp_base_l1_addr, out_base_l1_addr);

    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
