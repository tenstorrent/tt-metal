// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "dataflow_api.h"

#define ENABLE_DEBUG 1

#if ENABLE_DEBUG
#include "debug/dprint.h"
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

struct GatherConfigHeader {
    uint16_t noc_x;
    uint16_t noc_y;
    uint16_t length;
};

constexpr uint32_t GATHER_CONFIG_HEADER_NUM_ELEMENTS = 3;

struct GatherStep {
    uint16_t src_local_idx;
    uint16_t dst_local_idx;
    uint16_t nsticks;
};

FORCE_INLINE static void decode_gather_config_header(
    const uint16_t* const config_data, int offset, GatherConfigHeader& header) {
    // [ noc_x, noc_y, length ]
    header.noc_x = config_data[offset + 0];
    header.noc_y = config_data[offset + 1];
    header.length = config_data[offset + 2];
}

FORCE_INLINE static void decode_gather_config_step(const uint16_t* const config_data, int offset, GatherStep& step) {
    // [ src_local_idx, dst_local_idx, nsticks ]
    step.src_local_idx = config_data[offset + 0];
    step.dst_local_idx = config_data[offset + 1];
    step.nsticks = config_data[offset + 2];
}

template <
    uint32_t stick_nbytes,
    uint32_t input_aligned_page_size,
    bool is_block_sharded,
    bool is_width_sharded,
    bool is_read,
    bool is_col_major,
    bool blocking = false>
FORCE_INLINE static void copy_stick(
    const GatherStep& step, uint64_t base_addr, uint32_t in_base_l1_addr, uint32_t out_base_l1_addr) {
    uint16_t nsticks = step.nsticks;
    uint16_t src_local_idx = step.src_local_idx;
    uint16_t dst_local_idx = step.dst_local_idx;

    uint32_t size = nsticks * stick_nbytes;
    uint32_t dst_offset = dst_local_idx * stick_nbytes;
    uint32_t src_offset = src_local_idx * input_aligned_page_size;

    if constexpr (is_read) {
        uint32_t dst_addr = out_base_l1_addr + dst_offset;
        uint64_t src_addr = base_addr + src_offset;
        if constexpr (stick_nbytes == input_aligned_page_size) {
            // Single large read
            noc_async_read(src_addr, dst_addr, size);
        } else {
            // Multiple smaller reads
            for (uint16_t k = 0; k < nsticks; k++) {
                noc_async_read(src_addr, dst_addr, stick_nbytes);
                dst_addr += stick_nbytes;
                src_addr += input_aligned_page_size;
            }
        }
    } else {
        uint64_t dst_addr = base_addr + dst_offset;
        uint32_t src_addr = in_base_l1_addr + src_offset;
        if constexpr (stick_nbytes == input_aligned_page_size) {
            // Single large write
            noc_async_write(src_addr, dst_addr, size);
        } else {
            // Multiple smaller writes
            for (uint16_t k = 0; k < nsticks; k++) {
                noc_async_write(src_addr, dst_addr, stick_nbytes);
                dst_addr += stick_nbytes;
                src_addr += input_aligned_page_size;
            }
        }
    }
}

template <
    uint32_t stick_nbytes,
    uint32_t input_aligned_page_size,
    bool is_block_sharded,
    bool is_width_sharded,
    bool is_read,
    bool is_col_major,
    bool blocking = false>
FORCE_INLINE static void copy_sticks_async(
    const tt_l1_ptr uint16_t* config_data,
    const tt_l1_ptr uint16_t* blocking_config_data,
    uint16_t my_noc_x,
    uint16_t my_noc_y,
    uint32_t in_base_l1_addr,
    uint32_t out_base_l1_addr,
    uint16_t num_blocks) {
    uint32_t config_data_offset = 0;
    uint16_t length = config_data[config_data_offset + 2];
    while (length) {
        GatherConfigHeader header;
        decode_gather_config_header(config_data, config_data_offset, header);
        config_data_offset += GATHER_CONFIG_HEADER_NUM_ELEMENTS;

        const uint16_t real_noc_x = ((is_block_sharded && !is_col_major) || is_width_sharded) ? my_noc_x : header.noc_x;
        const uint16_t real_noc_y = ((is_block_sharded && is_col_major) || is_width_sharded) ? my_noc_y : header.noc_y;
        const uint64_t base_addr = get_noc_addr(real_noc_x, real_noc_y, is_read ? in_base_l1_addr : out_base_l1_addr);

        length = header.length;

        // TODO: Remove branch here by always providing a blocking config - even if there's only one block
        if constexpr (blocking) {
            GatherStep step;
            uint16_t block_offset = 0;

            for (uint16_t block_id = 0; block_id < num_blocks; block_id++) {
                const uint16_t num_steps_in_block = blocking_config_data[block_id];
                const uint16_t block_stride = num_steps_in_block * GATHER_CONFIG_HEADER_NUM_ELEMENTS;
                const uint16_t block_end_offset = block_offset + block_stride;

                if (block_offset >= length) {
                    break;  // Avoid executing beyond valid steps
                }

                for (uint16_t step_offset = block_offset; step_offset < block_end_offset && step_offset < length;
                     step_offset += GATHER_CONFIG_HEADER_NUM_ELEMENTS) {
                    decode_gather_config_step(config_data, config_data_offset + step_offset, step);
                    copy_stick<
                        stick_nbytes,
                        input_aligned_page_size,
                        is_block_sharded,
                        is_width_sharded,
                        is_read,
                        is_col_major,
                        blocking>(step, base_addr, in_base_l1_addr, out_base_l1_addr);
                }
                block_offset += block_stride;  // Increment AFTER executing block
            }

            config_data_offset += length;
        } else {
            GatherStep step;
            for (uint16_t j = 0; j < length; j += GATHER_CONFIG_HEADER_NUM_ELEMENTS) {
                decode_gather_config_step(config_data, config_data_offset + j, step);
                copy_stick<
                    stick_nbytes,
                    input_aligned_page_size,
                    is_block_sharded,
                    is_width_sharded,
                    is_read,
                    is_col_major,
                    blocking>(step, base_addr, in_base_l1_addr, out_base_l1_addr);
            }
            config_data_offset += length;
        }

        // Get next command length; if 0 we are done
        length = config_data[config_data_offset + 2];
    }
}

void kernel_main() {
    constexpr uint32_t padding_config_cb_id = get_compile_time_arg_val(0);          // has untilized input shard
    constexpr uint32_t local_config_cb_id = get_compile_time_arg_val(1);            // has untilized input shard
    constexpr uint32_t remote_config_cb_id = get_compile_time_arg_val(2);           // has untilized input shard
    constexpr uint32_t blocking_local_config_cb_id = get_compile_time_arg_val(3);   // has untilized input shard
    constexpr uint32_t blocking_remote_config_cb_id = get_compile_time_arg_val(4);  // has untilized input shard
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(5);                     // has untilized input shard
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(6);                      // has untilized input shard
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(7);      // output shard with padding and halo goes here
    constexpr uint32_t pad_cb_id = get_compile_time_arg_val(8);      // cb for const pad val buffer
    constexpr uint32_t pad_val_u32 = get_compile_time_arg_val(9);    // pad value to fill pad buffer with
    constexpr uint32_t in_nsticks = get_compile_time_arg_val(10);    // number of sticks
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(11);  // stick size in bytes (post untilize)
    constexpr uint32_t is_block_sharded = get_compile_time_arg_val(12);
    constexpr uint32_t remote_read = get_compile_time_arg_val(13);
    constexpr bool is_col_major = get_compile_time_arg_val(14) == 1;
    constexpr uint32_t is_width_sharded = get_compile_time_arg_val(15);
    constexpr uint32_t input_aligned_page_size = get_compile_time_arg_val(16);

    constexpr uint32_t elem_nbytes = sizeof(uint16_t);

    const uint16_t my_noc_x = NOC_X(my_x[noc_index]);
    const uint16_t my_noc_y = NOC_Y(my_y[noc_index]);
    const uint32_t in_base_l1_addr = get_read_ptr(in_cb_id);
    const uint32_t out_base_l1_addr = get_write_ptr(out_cb_id);

    uint32_t blocking_local_config_data_l1_addr = get_read_ptr(blocking_local_config_cb_id);
    const tt_l1_ptr uint16_t* blocking_local_config_data =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(blocking_local_config_data_l1_addr);
    const uint16_t num_blocks = blocking_local_config_data[0];

    uint32_t blocking_remote_config_data_l1_addr = get_read_ptr(blocking_remote_config_cb_id);
    const tt_l1_ptr uint16_t* blocking_remote_config_data =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(blocking_remote_config_data_l1_addr);

    if constexpr (local_config_cb_id) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            DPRINT << i << "  " << (int)blocking_local_config_data[i + 1] << ENDL();
        }
    }

    if constexpr (padding_config_cb_id) {
        // construct the pad stick in its buffer
        cb_reserve_back(pad_cb_id, 1);
        const uint16_t pad_val = pad_val_u32;
        fill_with_val(get_write_ptr(pad_cb_id), stick_nbytes / elem_nbytes, pad_val);
        cb_push_back(pad_cb_id, 1);

        uint32_t padding_config_l1_addr = get_read_ptr(padding_config_cb_id);
        volatile tt_l1_ptr uint16_t* config_data =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(padding_config_l1_addr);
        // print_data_u16(padding_config_l1_addr, 1, 16);
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

    // input shards
    if constexpr (local_config_cb_id) {
        cb_reserve_back(src_cb_id, in_nsticks);
        cb_push_back(src_cb_id, in_nsticks);
    }

    cb_wait_front(in_cb_id, in_nsticks);  // make sure untilized data is available
    if constexpr (remote_config_cb_id) {
        uint32_t config_data_l1_addr = get_read_ptr(remote_config_cb_id);
        const tt_l1_ptr uint16_t* config_data = reinterpret_cast<const tt_l1_ptr uint16_t*>(config_data_l1_addr);
        copy_sticks_async<
            stick_nbytes,
            input_aligned_page_size,
            is_block_sharded,
            is_width_sharded,
            remote_read,
            is_col_major,
            true>(
            config_data,
            blocking_remote_config_data + 1,
            my_noc_x,
            my_noc_y,
            in_base_l1_addr,
            out_base_l1_addr,
            num_blocks);
    }

    if constexpr (local_config_cb_id) {
        uint32_t config_data_l1_addr = get_read_ptr(local_config_cb_id);
        const tt_l1_ptr uint16_t* config_data = reinterpret_cast<const tt_l1_ptr uint16_t*>(config_data_l1_addr);
        copy_sticks_async<
            stick_nbytes,
            input_aligned_page_size,
            is_block_sharded,
            is_width_sharded,
            false,
            is_col_major,
            true>(
            config_data,
            blocking_local_config_data + 1,
            my_noc_x,
            my_noc_y,
            in_base_l1_addr,
            out_base_l1_addr,
            num_blocks);
    }

    noc_async_read_barrier();
    noc_async_write_barrier();
}
