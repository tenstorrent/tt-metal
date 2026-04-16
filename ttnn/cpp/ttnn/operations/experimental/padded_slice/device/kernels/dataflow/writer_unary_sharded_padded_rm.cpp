// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t num_elements_per_row = get_arg_val<uint32_t>(1);
    const uint32_t unpadded_row_size_bytes = get_arg_val<uint32_t>(2);
    const uint32_t padded_row_size_bytes = get_arg_val<uint32_t>(3);
    const uint32_t pad_size_bytes = padded_row_size_bytes - unpadded_row_size_bytes;
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_temp_pad = get_compile_time_arg_val(1);
    constexpr uint32_t output_elem_size = get_compile_time_arg_val(2);

    experimental::Noc noc;
    experimental::CB cb_out_obj(cb_id_out);
    experimental::CB cb_temp_pad_obj(cb_temp_pad);

    const uint32_t pad_addr = cb_temp_pad_obj.get_read_ptr();
    const uint32_t out_addr = cb_out_obj.get_read_ptr();
    if (pad_size_bytes == 0) {
        return;  // No padding needed, exit early
    }
#ifdef DEBUG
    DPRINT << "num_units: " << num_units << ", num_elements_per_row: " << num_elements_per_row
           << ", unpadded_row_size_bytes: " << unpadded_row_size_bytes
           << ", padded_row_size_bytes: " << padded_row_size_bytes << ", pad_size_bytes: " << pad_size_bytes << ENDL();
    DEVICE_PRINT(
        "num_units: {}, num_elements_per_row: {}, unpadded_row_size_bytes: {}, padded_row_size_bytes: {}, "
        "pad_size_bytes: {}\n",
        num_units,
        num_elements_per_row,
        unpadded_row_size_bytes,
        padded_row_size_bytes,
        pad_size_bytes);
    DPRINT << "CB Temp Pad " << cb_temp_pad << "pad_addr: " << pad_addr << ", out_addr: " << out_addr << ENDL();
    DEVICE_PRINT("CB Temp Pad {}, pad_addr: {}, out_addr: {}\n", cb_temp_pad, pad_addr, out_addr);
    DPRINT << "Output Elem Size " << output_elem_size << ENDL();
    DEVICE_PRINT("Output Elem Size {}\n", output_elem_size);
#endif

    if constexpr (output_elem_size == 2) {
        volatile tt_l1_ptr uint16_t* pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pad_addr);
        for (uint32_t i = 0; i < num_elements_per_row; ++i) {
            pad_ptr[i] = 0;
        }
    } else if constexpr (output_elem_size == 4) {
        volatile tt_l1_ptr uint32_t* pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_addr);
        for (uint32_t i = 0; i < num_elements_per_row; ++i) {
            pad_ptr[i] = 0;
        }
    }

    uint32_t write_addr = out_addr + unpadded_row_size_bytes;
    // pad_size_bytes is runtime; issue each read as a single-packet UnicastEndpoint NOC transfer
    experimental::UnicastEndpoint self_ep;
    const auto pad_src = experimental::local_addr(pad_addr + unpadded_row_size_bytes, noc.get_noc_id());

    for (uint32_t i = 0; i < num_units; ++i) {
        noc.async_read(self_ep, experimental::CoreLocalMem<uint32_t>(write_addr), pad_size_bytes, pad_src, {});
        write_addr += padded_row_size_bytes;
    }
    noc.async_read_barrier();
}
