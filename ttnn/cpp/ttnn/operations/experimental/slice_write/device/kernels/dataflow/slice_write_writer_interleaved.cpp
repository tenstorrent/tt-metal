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

#ifdef UNPAD_INPUT_WIDTH
    const uint32_t padding_width_ntiles = get_arg_val<uint32_t>(21);
#endif

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
#ifdef UNPAD_INPUT_WIDTH
    DPRINT << "padding_width_ntiles: " << padding_width_ntiles << ENDL();
    DEVICE_PRINT("padding_width_ntiles: {}\n", padding_width_ntiles);
#endif

#endif
    tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(get_arg_addr(9));
    volatile tt_l1_ptr uint32_t* num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* id_per_dim = num_padded_sticks + num_dims;
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t page_offset = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto s0 = TensorAccessor(dst_args, dst_addr, output_stick_size);
    const uint32_t noc_write_size = std::min(output_stick_size, input_stick_size);

    experimental::Noc noc;
    experimental::CB cb_out0(cb_id_out0);

    uint32_t dst_stick_id = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_out0.wait_front(num_read_per_barrier);
        uint32_t src_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
#ifdef UNPAD_INPUT_WIDTH
            if ((id_per_dim[0] + padding_width_ntiles + 1) <= num_unpadded_sticks[0]) {
                noc.async_write(cb_out0, s0, noc_write_size, {.offset_bytes = src_offset}, {.page_id = dst_stick_id});
            }
#else
            noc.async_write(
                cb_out0, s0, noc_write_size, {.offset_bytes = src_offset + page_offset}, {.page_id = dst_stick_id});
#endif
#ifdef DEBUG
            DPRINT << "SRC L1 : " << src_offset << " Dst Stick ID " << dst_stick_id << " sticks_read: " << sticks_read
                   << " Coord " << id_per_dim[0] << ", " << id_per_dim[1] << ", " << id_per_dim[2] << ", "
                   << id_per_dim[3] << ENDL();
            DEVICE_PRINT(
                "SRC L1 : {} Dst Stick ID {} sticks_read: {} Coord {}, {}, {}, {}\n",
                src_offset,
                dst_stick_id,
                sticks_read,
                id_per_dim[0],
                id_per_dim[1],
                id_per_dim[2],
                id_per_dim[3]);
#endif
            src_offset += stick_size_offset;
            dst_stick_id++;
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
        cb_out0.pop_front(num_read_per_barrier);
    }
}
