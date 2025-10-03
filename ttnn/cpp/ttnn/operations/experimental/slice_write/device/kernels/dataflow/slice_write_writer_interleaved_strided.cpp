// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_stick_size = get_arg_val<uint32_t>(1);
    const uint32_t input_stick_size = get_arg_val<uint32_t>(2);
    const uint32_t stick_size_offset = get_arg_val<uint32_t>(3);
    const uint32_t num_dims = get_arg_val<uint32_t>(4);
    const uint32_t start_id = get_arg_val<uint32_t>(5);
    const uint32_t num_sticks_per_core = get_arg_val<uint32_t>(6);
    const uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(7);
    const uint32_t num_read_per_barrier = get_arg_val<uint32_t>(8);

#ifdef DEBUG
    DPRINT << "dst_addr: " << dst_addr << ENDL();
    DPRINT << "output_stick_size: " << output_stick_size << ENDL();
    DPRINT << "input_stick_size: " << input_stick_size << ENDL();
    DPRINT << "stick_size_offset: " << stick_size_offset << ENDL();
    DPRINT << "num_dims: " << num_dims << ENDL();
    DPRINT << "start_id: " << start_id << ENDL();
    DPRINT << "num_sticks_per_core: " << num_sticks_per_core << ENDL();
    DPRINT << "num_sticks_per_core_read: " << num_sticks_per_core_read << ENDL();
    DPRINT << "num_read_per_barrier: " << num_read_per_barrier << ENDL();

#endif
    tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(get_arg_addr(9));
    volatile tt_l1_ptr uint32_t* num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* id_per_dim = num_padded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* rev_stride = id_per_dim + num_dims;

#ifdef DEBUG
    DPRINT << "num_input_sticks_per_dim: ";
    for (uint32_t i = 0; i < num_dims; i++) {
        DPRINT << num_unpadded_sticks[i] << ", ";
    }
    DPRINT << ENDL();
    DPRINT << "num_output_sticks_per_dim: ";
    for (uint32_t i = 0; i < num_dims; i++) {
        DPRINT << num_padded_sticks[i] << ", ";
    }
    DPRINT << ENDL();
    DPRINT << "rev_stride: ";
    for (uint32_t i = 0; i < num_dims; i++) {
        DPRINT << rev_stride[i] << ", ";
    }
    DPRINT << ENDL();
    DPRINT << "id_per_dim: ";
    for (uint32_t i = 0; i < num_dims; i++) {
        DPRINT << id_per_dim[i] << ", ";
    }
    DPRINT << ENDL();
#endif
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t alignment_offset = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr uint32_t page_begins_offset = get_compile_time_arg_val(3);
    constexpr uint32_t element_size = get_compile_time_arg_val(4);
    constexpr auto dst_args = TensorAccessorArgs<5>();

    const auto s0 = TensorAccessor(dst_args, dst_addr, output_stick_size);
    const uint32_t noc_write_size = std::min(output_stick_size, input_stick_size);
    uint32_t dst_stick_id = start_id;
    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    uint32_t sticks_written = 0;
#ifdef DEBUG
    uint32_t base_src_l1_addr = get_read_ptr(cb_id_in);
#endif
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_written < num_sticks_per_core; ++iter) {
#ifdef LAST_DIM_STRIDED
        // read output rows in batches if striding on last dim
        cb_reserve_back(cb_id_out, num_read_per_barrier);
        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            uint32_t l1_write_addr = get_write_ptr(cb_id_out);
            uint64_t src_noc_addr = get_noc_addr(src_stick_id, s0);
            noc_async_read(src_noc_addr, l1_write_addr, output_stick_size);
            l1_write_addr += output_stick_size;
            src_stick_id += rev_stride[1];
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks[j];
                } else {
                    break;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_out, num_read_per_barrier);
#endif

        // begin writing
        cb_wait_front(cb_id_in, num_read_per_barrier);
#ifdef LAST_DIM_STRIDED
        cb_wait_front(cb_id_out, num_read_per_barrier);
        uint32_t output_l1_addr = get_read_ptr(cb_id_out);
#endif
        uint32_t src_buffer_l1_addr = get_read_ptr(cb_id_in);
        for (uint32_t i = 0; i < num_read_per_barrier and sticks_written < num_sticks_per_core; ++i) {
            sticks_written++;
            uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, s0);
#ifdef LAST_DIM_STRIDED
            // fill the read output row with the input with stride
            volatile tt_l1_ptr uint8_t* out_stick =
                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(output_l1_addr + page_begins_offset);
            volatile tt_l1_ptr uint8_t* in_stick =
                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src_buffer_l1_addr + alignment_offset);
            uint32_t out_index = 0;
            for (uint32_t j = 0; j < input_stick_size / element_size; j++) {
                // general writing for all data types byte by byte
                for (uint32_t byte = 0; byte < element_size; byte++) {
                    out_stick[out_index * element_size + byte] = in_stick[j * element_size + byte];
                }
                out_index += rev_stride[0];
            }
            noc_async_write(output_l1_addr, dst_noc_addr, output_stick_size);
            output_l1_addr += output_stick_size;
#else
            noc_async_write(src_buffer_l1_addr + alignment_offset, dst_noc_addr + page_begins_offset, noc_write_size);
#endif
#ifdef DEBUG
            DPRINT << "SRC L1 : " << src_buffer_l1_addr - base_src_l1_addr << " Dst Stick ID " << dst_stick_id
                   << " sticks_written: " << sticks_written << " Coord ";
            for (uint32_t j = 0; j < num_dims; j++) {
                DPRINT << ", " << id_per_dim[j];
            }
            DPRINT << ENDL();
#endif
            src_buffer_l1_addr += stick_size_offset;
            dst_stick_id += rev_stride[1];
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    dst_stick_id += num_padded_sticks[j];
                } else {
                    break;
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_in, num_read_per_barrier);
    }
}
