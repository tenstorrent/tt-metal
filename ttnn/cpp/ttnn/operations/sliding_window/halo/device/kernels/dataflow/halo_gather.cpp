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

inline bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
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

static inline void decode_gather_config_header(const uint16_t* config_data, int offset, GatherConfigHeader& header) {
    header.noc_x = config_data[offset + 0];
    header.noc_y = config_data[offset + 1];
    header.length = config_data[offset + 2];
}

static inline void decode_gather_config_step(const uint16_t* config_data, int offset, GatherStep& step) {
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
    bool is_col_major>
static inline void copy_stick(
    const GatherStep& step, uint64_t base_addr, uint32_t in_base_l1_addr, uint32_t out_base_l1_addr) {
    const uint16_t nsticks = step.nsticks;
    const uint16_t src_local_idx = step.src_local_idx;
    const uint16_t dst_local_idx = step.dst_local_idx;

    const uint32_t size = nsticks * stick_nbytes;
    const uint32_t dst_offset = dst_local_idx * stick_nbytes;
    const uint32_t src_offset = src_local_idx * input_aligned_page_size;

    if constexpr (is_read) {
        // Reading from remote -> writing into local L1
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
        // Writing from local L1 -> remote
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
    bool is_col_major>
static inline uint16_t copy_block_sticks_async(
    const tt_l1_ptr uint16_t* config_data,      // entire config
    uint32_t config_data_offset,                // offset into config_data
    uint16_t block_id,                          // which block we are processing
    uint16_t block_offset,                      // current offset in instructions
    uint16_t length,                            // total instructions (j-units)
    const tt_l1_ptr uint16_t* blocking_config,  // pointer to the blocking config array
    uint64_t base_addr,
    uint32_t in_base_l1_addr,
    uint32_t out_base_l1_addr) {
    const uint16_t num_steps_in_block = blocking_config[block_id];

    // If there's nothing to do, skip it
    if (num_steps_in_block == 0) {
        return block_offset;
    }

    const uint16_t block_stride = num_steps_in_block * GATHER_CONFIG_HEADER_NUM_ELEMENTS;
    const uint16_t block_end_offset = block_offset + block_stride;

    // Already processed all instructions
    if (block_offset >= length) {
        return block_offset;
    }

    GatherStep step;
    for (uint16_t step_offset = block_offset; step_offset < block_end_offset && step_offset < length;
         step_offset += GATHER_CONFIG_HEADER_NUM_ELEMENTS) {
        decode_gather_config_step(config_data, config_data_offset + step_offset, step);
        copy_stick<stick_nbytes, input_aligned_page_size, is_block_sharded, is_width_sharded, is_read, is_col_major>(
            step, base_addr, in_base_l1_addr, out_base_l1_addr);
    }

    // Return the new offset for the next block
    return block_offset + block_stride;
}

template <
    uint32_t stick_nbytes,
    uint32_t input_aligned_page_size,
    bool is_block_sharded,
    bool is_width_sharded,
    bool is_read,
    bool is_col_major,
    bool blocking = false>
static inline void copy_sticks_async(
    const tt_l1_ptr uint16_t* config_data,
    const tt_l1_ptr uint16_t* blocking_config_data,
    uint16_t my_noc_x,
    uint16_t my_noc_y,
    uint32_t in_base_l1_addr,
    uint32_t out_base_l1_addr,
    uint16_t num_blocks) {
    uint32_t config_data_offset = 0;
    uint16_t length = config_data[config_data_offset + 2];  // # instructions in the first command

    while (length) {
        GatherConfigHeader header;
        decode_gather_config_header(config_data, config_data_offset, header);
        config_data_offset += GATHER_CONFIG_HEADER_NUM_ELEMENTS;

        const uint16_t real_noc_x = ((is_block_sharded && !is_col_major) || is_width_sharded) ? my_noc_x : header.noc_x;
        const uint16_t real_noc_y = ((is_block_sharded && is_col_major) || is_width_sharded) ? my_noc_y : header.noc_y;
        const uint64_t base_addr = get_noc_addr(real_noc_x, real_noc_y, is_read ? in_base_l1_addr : out_base_l1_addr);

        length = header.length;

        if constexpr (blocking) {
            uint16_t block_offset = 0;
            for (uint16_t block_id = 0; block_id < num_blocks; block_id++) {
                uint16_t old_offset = block_offset;
                block_offset = copy_block_sticks_async<
                    stick_nbytes,
                    input_aligned_page_size,
                    is_block_sharded,
                    is_width_sharded,
                    is_read,
                    is_col_major>(
                    config_data,
                    config_data_offset,
                    block_id,
                    block_offset,
                    length,
                    blocking_config_data,
                    base_addr,
                    in_base_l1_addr,
                    out_base_l1_addr);
                if (block_offset > old_offset) {
                    noc_async_read_barrier();
                    noc_async_write_barrier();
                }
                if (block_offset >= length) {
                    break;
                }
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
                    is_col_major>(step, base_addr, in_base_l1_addr, out_base_l1_addr);
            }
            config_data_offset += length;
        }

        // Move to the next command (if any)
        length = config_data[config_data_offset + 2];
    }
}

void kernel_main() {
    constexpr uint32_t padding_config_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_config_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t remote_config_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t blocking_local_config_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t blocking_remote_config_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t pad_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t pad_val_u32 = get_compile_time_arg_val(9);
    constexpr uint32_t in_nsticks = get_compile_time_arg_val(10);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(11);
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

    const uint32_t blocking_local_config_data_l1_addr = get_read_ptr(blocking_local_config_cb_id);
    const tt_l1_ptr uint16_t* blocking_local_config_data =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(blocking_local_config_data_l1_addr);
    const uint16_t num_blocks = blocking_local_config_data[0];

    const uint32_t blocking_remote_config_data_l1_addr = get_read_ptr(blocking_remote_config_cb_id);
    const tt_l1_ptr uint16_t* blocking_remote_config_data =
        reinterpret_cast<const tt_l1_ptr uint16_t*>(blocking_remote_config_data_l1_addr);

    if constexpr (padding_config_cb_id) {
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
            uint64_t dst_addr = dst_base_addr + (dst_local_idx * stick_nbytes);

            for (uint16_t k = 0; k < nsticks; ++k) {
                noc_async_read(padding_l1_addr, dst_addr, stick_nbytes);
                dst_addr += stick_nbytes;
            }
        }
    }

    if constexpr (local_config_cb_id) {
        cb_reserve_back(src_cb_id, in_nsticks);
        cb_push_back(src_cb_id, in_nsticks);
    }

    cb_wait_front(in_cb_id, in_nsticks);

    if constexpr (remote_config_cb_id) {
        const uint32_t config_data_l1_addr = get_read_ptr(remote_config_cb_id);
        const tt_l1_ptr uint16_t* config_data = reinterpret_cast<const tt_l1_ptr uint16_t*>(config_data_l1_addr);
        copy_sticks_async<
            stick_nbytes,
            input_aligned_page_size,
            is_block_sharded,
            is_width_sharded,
            remote_read,  // is_read
            is_col_major,
            true  // blocking
            >(
            config_data,
            blocking_remote_config_data + 1,  // skip the first element (num_blocks)
            my_noc_x,
            my_noc_y,
            in_base_l1_addr,
            out_base_l1_addr,
            num_blocks);
    }

    // Local config
    if constexpr (local_config_cb_id) {
        const uint32_t config_data_l1_addr = get_read_ptr(local_config_cb_id);
        const tt_l1_ptr uint16_t* config_data = reinterpret_cast<const tt_l1_ptr uint16_t*>(config_data_l1_addr);
        copy_sticks_async<
            stick_nbytes,
            input_aligned_page_size,
            is_block_sharded,
            is_width_sharded,
            false,  // is_read
            is_col_major,
            true  // blocking
            >(
            config_data,
            blocking_local_config_data + 1,  // skip first element (num_blocks)
            my_noc_x,
            my_noc_y,
            in_base_l1_addr,
            out_base_l1_addr,
            num_blocks);
    }

    noc_async_read_barrier();
    noc_async_write_barrier();
}
