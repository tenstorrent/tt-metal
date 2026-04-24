// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

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
    DEVICE_PRINT("dst_addr: {}\n", dst_addr);
    DPRINT << "output_stick_size: " << output_stick_size << ENDL();
    DEVICE_PRINT("output_stick_size: {}\n", output_stick_size);
    DPRINT << "input_stick_size: " << input_stick_size << ENDL();
    DEVICE_PRINT("input_stick_size: {}\n", input_stick_size);
    DPRINT << "stick_size_offset: " << stick_size_offset << ENDL();
    DEVICE_PRINT("stick_size_offset: {}\n", stick_size_offset);
    DPRINT << "num_dims: " << num_dims << ENDL();
    DEVICE_PRINT("num_dims: {}\n", num_dims);
    DPRINT << "start_id: " << start_id << ENDL();
    DEVICE_PRINT("start_id: {}\n", start_id);
    DPRINT << "num_sticks_per_core: " << num_sticks_per_core << ENDL();
    DEVICE_PRINT("num_sticks_per_core: {}\n", num_sticks_per_core);
    DPRINT << "num_sticks_per_core_read: " << num_sticks_per_core_read << ENDL();
    DEVICE_PRINT("num_sticks_per_core_read: {}\n", num_sticks_per_core_read);
    DPRINT << "num_read_per_barrier: " << num_read_per_barrier << ENDL();
    DEVICE_PRINT("num_read_per_barrier: {}\n", num_read_per_barrier);

#endif
    tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(get_arg_addr(9));
    volatile tt_l1_ptr uint32_t* num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* id_per_dim = num_padded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* rev_stride = id_per_dim + num_dims;

#ifdef DEBUG
    DPRINT << "num_input_sticks_per_dim: ";
    DEVICE_PRINT("num_input_sticks_per_dim: ");
    for (uint32_t i = 0; i < num_dims; i++) {
        DPRINT << num_unpadded_sticks[i] << ", ";
        DEVICE_PRINT("{} ", num_unpadded_sticks[i]);
    }
    DPRINT << ENDL();
    DPRINT << "num_output_sticks_per_dim: ";
    DEVICE_PRINT("\nnum_output_sticks_per_dim: ");
    for (uint32_t i = 0; i < num_dims; i++) {
        DPRINT << num_padded_sticks[i] << ", ";
        DEVICE_PRINT("{} ", num_padded_sticks[i]);
    }
    DPRINT << ENDL();
    DPRINT << "rev_stride: ";
    DEVICE_PRINT("\nrev_stride: ");
    for (uint32_t i = 0; i < num_dims; i++) {
        DPRINT << rev_stride[i] << ", ";
        DEVICE_PRINT("{} ", rev_stride[i]);
    }
    DPRINT << ENDL();
    DPRINT << "id_per_dim: ";
    DEVICE_PRINT("\nid_per_dim: ");
    for (uint32_t i = 0; i < num_dims; i++) {
        DPRINT << id_per_dim[i] << ", ";
        DEVICE_PRINT("{} ", id_per_dim[i]);
    }
    DPRINT << ENDL();
    DEVICE_PRINT("\n");
#endif
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t alignment_offset = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr uint32_t page_begins_offset = get_compile_time_arg_val(3);
    constexpr uint32_t element_size = get_compile_time_arg_val(4);
    constexpr auto dst_args = TensorAccessorArgs<5>();

    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto s0 = TensorAccessor(dst_args, dst_addr, output_stick_size);
    const uint32_t noc_write_size = std::min(output_stick_size, input_stick_size);

    experimental::Noc noc;
    experimental::CB cb_in(cb_id_in);
    experimental::CB cb_out(cb_id_out);

    uint32_t dst_stick_id = start_id;
    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    uint32_t sticks_written = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_written < num_sticks_per_core; ++iter) {
#ifdef LAST_DIM_STRIDED
        // read output rows in batches if striding on last dim
        cb_out.reserve_back(num_read_per_barrier);
        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            noc.async_read(s0, cb_out, output_stick_size, {.page_id = src_stick_id}, {});
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
        noc.async_read_barrier();
        cb_out.push_back(num_read_per_barrier);
#endif

        // begin writing
        cb_in.wait_front(num_read_per_barrier);
#ifdef LAST_DIM_STRIDED
        cb_out.wait_front(num_read_per_barrier);
        uint32_t output_offset = 0;
#endif
        uint32_t src_offset = 0;
        for (uint32_t i = 0; i < num_read_per_barrier and sticks_written < num_sticks_per_core; ++i) {
            sticks_written++;
#ifdef LAST_DIM_STRIDED
            // fill the read output row with the input with stride
            volatile tt_l1_ptr uint8_t* out_stick = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(
                cb_out.get_read_ptr() + output_offset + page_begins_offset);
            volatile tt_l1_ptr uint8_t* in_stick =
                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(cb_in.get_read_ptr() + src_offset + alignment_offset);
            uint32_t out_index = 0;
            for (uint32_t j = 0; j < input_stick_size / element_size; j++) {
                // general writing for all data types byte by byte
                for (uint32_t byte = 0; byte < element_size; byte++) {
                    out_stick[out_index * element_size + byte] = in_stick[j * element_size + byte];
                }
                out_index += rev_stride[0];
            }
            noc.async_write(cb_out, s0, output_stick_size, {.offset_bytes = output_offset}, {.page_id = dst_stick_id});
            output_offset += output_stick_size;
#else
            noc.async_write(
                cb_in,
                s0,
                noc_write_size,
                {.offset_bytes = src_offset + alignment_offset},
                {.page_id = dst_stick_id, .offset_bytes = page_begins_offset});
#endif
#ifdef DEBUG
            DPRINT << "SRC L1 : " << src_offset << " Dst Stick ID " << dst_stick_id
                   << " sticks_written: " << sticks_written << " Coord ";
            DEVICE_PRINT(
                "SRC L1 : {} Dst Stick ID {} sticks_written: {} Coord ", src_offset, dst_stick_id, sticks_written);
            for (uint32_t j = 0; j < num_dims; j++) {
                DPRINT << ", " << id_per_dim[j];
                DEVICE_PRINT(", {} ", id_per_dim[j]);
            }
            DPRINT << ENDL();
            DEVICE_PRINT("\n");
#endif
            src_offset += stick_size_offset;
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
        noc.async_write_barrier();
        cb_in.pop_front(num_read_per_barrier);
    }
}
