// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
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

template <uint32_t stick_nbytes, uint32_t input_aligned_page_size>
void copy_sticks_async_to_temp(
    const tt_l1_ptr uint16_t* config_data,
    const uint16_t my_noc_x,
    const uint16_t my_noc_y,
    const uint32_t in_base_l1_addr,
    const uint32_t out_base_l1_addr) {
    int i = 0;
    int length = config_data[i + 2];

    const uint64_t base_addr = get_noc_addr(my_noc_x, my_noc_y, out_base_l1_addr);
    uint64_t dst_addr = base_addr;

    while (length) {
        length = config_data[i + 2];
        i += 3;
        for (uint16_t j = 0; j < length; j += 3) {
            uint16_t src_local_idx = config_data[i + j + 0];
            uint16_t nsticks = config_data[i + j + 2];
            uint32_t size = nsticks * stick_nbytes;
            uint32_t src_offset = src_local_idx * input_aligned_page_size;

            uint32_t src_addr = in_base_l1_addr + src_offset;
            if constexpr (stick_nbytes == input_aligned_page_size) {
                noc_async_write(src_addr, dst_addr, size);
                dst_addr += size;  // remote sticks from each config entry are written contiguously into the temp buffer
            } else {
                for (uint16_t k = 0; k < nsticks; k++) {
                    noc_async_write(src_addr, dst_addr, stick_nbytes);
                    dst_addr += stick_nbytes;
                    src_addr += input_aligned_page_size;
                }
            }
        }

        i += length;
    }
}

template <
    uint32_t stick_nbytes,
    uint32_t input_aligned_page_size,
    bool is_block_sharded,
    bool is_width_sharded,
    bool is_col_major>
void copy_sticks_async_from_temp(
    const tt_l1_ptr uint16_t* config_data,
    const uint16_t my_noc_x,
    const uint16_t my_noc_y,
    const uint32_t in_base_l1_addr,
    const uint32_t out_base_l1_addr) {
    int i = 0;
    int length = config_data[i + 2];

    uint64_t src_addr = in_base_l1_addr;

    while (length) {
        uint16_t noc_x = ((is_block_sharded && !is_col_major) || is_width_sharded) ? my_noc_x : config_data[i + 0];
        uint16_t noc_y = ((is_block_sharded && is_col_major) || is_width_sharded) ? my_noc_y : config_data[i + 1];
        length = config_data[i + 2];
        i += 3;
        const uint64_t base_addr = get_noc_addr(noc_x, noc_y, out_base_l1_addr);
        for (uint16_t j = 0; j < length; j += 3) {
            uint16_t dst_local_idx = config_data[i + j + 1];
            uint16_t nsticks = config_data[i + j + 2];
            uint32_t size = nsticks * stick_nbytes;
            uint32_t dst_offset = dst_local_idx * stick_nbytes;

            uint64_t dst_addr = base_addr + dst_offset;
            if constexpr (stick_nbytes == input_aligned_page_size) {
                noc_async_write(src_addr, dst_addr, size);
                src_addr += size;  // remote sticks from each config entry are read contiguously from the temp buffer
            } else {
                for (uint16_t k = 0; k < nsticks; k++) {
                    noc_async_write(src_addr, dst_addr, stick_nbytes);
                    dst_addr += stick_nbytes;
                    src_addr += stick_nbytes;
                }
            }
        }

        i += length;
    }
}

template <uint32_t stick_nbytes, uint32_t input_aligned_page_size>
void copy_sticks_async(
    const tt_l1_ptr uint16_t* config_data,
    const uint16_t my_noc_x,
    const uint16_t my_noc_y,
    const uint32_t in_base_l1_addr,
    const uint32_t out_base_l1_addr,
    const uint32_t in_out_buffer_start_delta) {
    int i = 0;
    int length = config_data[i + 2];

    while (length) {
        length = config_data[i + 2];
        i += 3;
        const uint64_t base_addr = get_noc_addr(my_noc_x, my_noc_y, out_base_l1_addr);
        const uint64_t base_addr_src = get_noc_addr(my_noc_x, my_noc_y, in_base_l1_addr);
        for (uint16_t j = 0; j < length; j += 3) {
            uint16_t src_local_idx = config_data[i + j + 0];
            uint16_t dst_local_idx = config_data[i + j + 1];
            uint16_t nsticks = config_data[i + j + 2];
            uint32_t size = nsticks * stick_nbytes;
            uint32_t dst_offset = dst_local_idx * stick_nbytes;
            uint32_t src_offset = src_local_idx * input_aligned_page_size;

            uint64_t dst_addr = base_addr + dst_offset;
            uint32_t src_addr = in_base_l1_addr + src_offset;

            bool is_forward_copy = dst_local_idx > src_local_idx + in_out_buffer_start_delta &&
                                   dst_local_idx <= src_local_idx + in_out_buffer_start_delta + nsticks;
            bool is_overlap_copy = (dst_local_idx > src_local_idx + in_out_buffer_start_delta &&
                                    dst_local_idx <= src_local_idx + in_out_buffer_start_delta + nsticks) ||
                                   (dst_local_idx + nsticks >= src_local_idx + in_out_buffer_start_delta &&
                                    dst_local_idx + nsticks < src_local_idx + in_out_buffer_start_delta + nsticks);
            if (is_overlap_copy) {      // dst and src data overlaps, stick by stick copy is necessary
                if (is_forward_copy) {  // dst data is being moved "in front" of the source data, reverse
                                        // ordering of stick by stick copy is necessary
                    for (int16_t k = nsticks - 1; k >= 0; k--) {
                        noc_async_write(src_addr + k * stick_nbytes, dst_addr + k * stick_nbytes, stick_nbytes);
                    }
                } else {
                    for (uint16_t k = 0; k < nsticks; k++) {
                        noc_async_write(src_addr + k * stick_nbytes, dst_addr + k * stick_nbytes, stick_nbytes);
                    }
                }
            } else {
                noc_async_write(src_addr, dst_addr, size);
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

1. (BR) copy the remote sticks from "my" shard to temp buffer
2. (BR) copy the local sticks from "my" shard to their destinations
3. (NC,BR) wait on semaphores for all cores to finish 1. and 2.
4. (NC) write the padding sticks to their destinations
4. (BR) copy the remote sticks from temp buffer to their destinations
*/

void kernel_main() {
    constexpr uint32_t padding_config_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_config_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t remote_config_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t remote_temp_cb_id = get_compile_time_arg_val(3);  // temp buffer for in place halo
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(4);          // the innput shard buffer
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(5);           // either the input shard or untilize output
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(6);      // output shard with padding and halo goes here
    constexpr uint32_t pad_cb_id = get_compile_time_arg_val(7);      // cb for const pad val buffer
    constexpr uint32_t pad_val_u32 = get_compile_time_arg_val(8);    // pad value to fill pad buffer with
    constexpr uint32_t in_npages = get_compile_time_arg_val(9);      // number of sticks
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(10);  // stick size in bytes (post untilize)
    constexpr uint32_t is_block_sharded = get_compile_time_arg_val(11);
    constexpr bool is_col_major = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t is_width_sharded = get_compile_time_arg_val(13);
    constexpr uint32_t input_aligned_page_size = get_compile_time_arg_val(14);
    constexpr uint32_t remote_read = get_compile_time_arg_val(15);  // Unused parameter
    constexpr uint32_t num_active_cores = get_compile_time_arg_val(16);
    constexpr uint32_t noc_TL_x = get_compile_time_arg_val(17);
    constexpr uint32_t noc_TL_y = get_compile_time_arg_val(18);
    constexpr uint32_t noc_BR_x = get_compile_time_arg_val(19);
    constexpr uint32_t noc_BR_y = get_compile_time_arg_val(20);
    constexpr uint32_t rectangular_x = get_compile_time_arg_val(21);
    constexpr uint32_t rectangular_y = get_compile_time_arg_val(22);
    constexpr uint32_t last_active_x = get_compile_time_arg_val(23);
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(24);
    constexpr uint32_t in_out_buffer_start_delta = get_compile_time_arg_val(25);
    constexpr uint32_t untilize_temp_cb_id =
        get_compile_time_arg_val(26);  // temp buffer for in place untilize with wide tensors
    constexpr uint32_t tile_cols = get_compile_time_arg_val(27);
    constexpr uint32_t tile_rows = get_compile_time_arg_val(28);

    constexpr uint32_t elem_nbytes = sizeof(uint16_t);
    constexpr uint16_t pad_core_id = 0xFFFF;
    constexpr uint32_t TILE_SIZE_BYTES = get_tile_size(in_cb_id);

    const uint16_t my_noc_x = NOC_X(my_x[noc_index]);
    const uint16_t my_noc_y = NOC_Y(my_y[noc_index]);
    uint32_t noop_core = get_arg_val<uint32_t>(0);
    uint32_t cast_core = get_arg_val<uint32_t>(1);
    if (noop_core) {
        return;
    }

    const uint32_t in_base_l1_addr = get_read_ptr(in_cb_id);
    const uint32_t out_base_l1_addr = get_write_ptr(out_cb_id);
    const uint32_t untilize_temp_l1_addr = get_read_ptr(untilize_temp_cb_id);

    if constexpr (local_config_cb_id) {
        cb_reserve_back(src_cb_id, in_npages);
        cb_push_back(src_cb_id, in_npages);
    }

    // make sure untilized data is available
    // for wide tensors a temp CB must be used due to implementation of the untilize LLK function vs pack_untilize
    if (untilize_temp_cb_id && local_config_cb_id) {
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
    if constexpr (remote_config_cb_id && remote_temp_cb_id) {
        const uint32_t temp_base_l1_addr = get_write_ptr(remote_temp_cb_id);
        uint32_t config_data_l1_addr = get_read_ptr(remote_config_cb_id);
        const tt_l1_ptr uint16_t* config_data = reinterpret_cast<const tt_l1_ptr uint16_t*>(config_data_l1_addr);
        copy_sticks_async_to_temp<stick_nbytes, input_aligned_page_size>(
            config_data, my_noc_x, my_noc_y, in_base_l1_addr, temp_base_l1_addr);
    }

    noc_async_read_barrier();
    noc_async_write_barrier();

    // move local sticks
    if constexpr (local_config_cb_id) {
        uint32_t config_data_l1_addr = get_read_ptr(local_config_cb_id);
        const tt_l1_ptr uint16_t* config_data = reinterpret_cast<const tt_l1_ptr uint16_t*>(config_data_l1_addr);
        copy_sticks_async<stick_nbytes, input_aligned_page_size>(
            config_data, my_noc_x, my_noc_y, in_base_l1_addr, out_base_l1_addr, in_out_buffer_start_delta);
    }

    noc_async_read_barrier();
    noc_async_write_barrier();

    // incremement the semaphore
    uint32_t semaphore_addr = get_semaphore(semaphore_id);
    const uint64_t semaphore_noc_addr = get_noc_addr(noc_TL_x, noc_TL_y, semaphore_addr);
    noc_semaphore_inc(semaphore_noc_addr, 1);

    // wait for all cores to finish copying local and remote data
    if (cast_core) {
        volatile tt_l1_ptr uint32_t* semaphore_noc_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_noc_addr);
        noc_semaphore_wait(semaphore_noc_addr_ptr, 2 * num_active_cores);

        const uint64_t mcast_noc_addr = get_noc_multicast_addr(noc_TL_x, noc_TL_y, noc_BR_x, noc_BR_y, semaphore_addr);

        noc_semaphore_set_multicast(semaphore_addr, mcast_noc_addr, rectangular_x * rectangular_y - 1);
    }

    // wait for multicast
    volatile tt_l1_ptr uint32_t* semaphore_noc_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_noc_addr);
    noc_semaphore_wait(semaphore_noc_addr_ptr, 2 * num_active_cores);

    // insert padding
    if constexpr (padding_config_cb_id) {
        // construct the pad stick in its buffer
        cb_reserve_back(pad_cb_id, 1);
        const uint16_t pad_val = pad_val_u32;
        fill_with_val(get_write_ptr(pad_cb_id), stick_nbytes / elem_nbytes, pad_val);
        cb_push_back(pad_cb_id, 1);

        uint32_t padding_config_l1_addr = get_read_ptr(padding_config_cb_id);
        volatile tt_l1_ptr uint16_t* config_data =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(padding_config_l1_addr);
        const uint64_t padding_l1_addr = get_noc_addr(my_noc_x, my_noc_y, get_read_ptr(pad_cb_id));
        const uint32_t dst_base_addr = out_base_l1_addr;
        uint16_t nsticks = 1;
        for (uint16_t j = 0; nsticks; j += 2) {
            uint16_t dst_local_idx = config_data[j + 0];
            nsticks = config_data[j + 1];

            uint64_t dst_addr = dst_base_addr + dst_local_idx * stick_nbytes;
            for (uint16_t k = 0; k < nsticks; ++k) {
                noc_async_read(padding_l1_addr, dst_addr, stick_nbytes);
                dst_addr += stick_nbytes;
            }
        }
    }

    // copy remote sticks from temp buffer to final destinations
    if constexpr (remote_config_cb_id && remote_temp_cb_id) {
        const uint32_t temp_base_l1_addr = get_read_ptr(remote_temp_cb_id);
        uint32_t config_data_l1_addr = get_read_ptr(remote_config_cb_id);
        const tt_l1_ptr uint16_t* config_data = reinterpret_cast<const tt_l1_ptr uint16_t*>(config_data_l1_addr);
        copy_sticks_async_from_temp<
            stick_nbytes,
            input_aligned_page_size,
            is_block_sharded,
            is_width_sharded,
            is_col_major>(config_data, my_noc_x, my_noc_y, temp_base_l1_addr, out_base_l1_addr);
    }

    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
