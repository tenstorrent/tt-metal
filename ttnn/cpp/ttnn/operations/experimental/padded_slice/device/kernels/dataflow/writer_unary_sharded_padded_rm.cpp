// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);
    const uint32_t num_elements_per_row = get_arg(args::num_elements_per_row);
    const uint32_t unpadded_row_size_bytes = get_arg(args::unpadded_row_size_bytes);
    const uint32_t padded_row_size_bytes = get_arg(args::padded_row_size_bytes);
    const uint32_t pad_size_bytes = padded_row_size_bytes - unpadded_row_size_bytes;
    constexpr uint32_t output_elem_size = get_arg(args::output_elem_size);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_temp_pad(dfb::temp_pad);

    const uint32_t pad_addr = dfb_temp_pad.get_read_ptr();
    const uint32_t out_addr = dfb_out.get_read_ptr();
    if (pad_size_bytes == 0) {
        return;  // No padding needed, exit early
    }
#ifdef DEBUG
    DPRINT(
        "num_units: {}, num_elements_per_row: {}, unpadded_row_size_bytes: {}, padded_row_size_bytes: {}, "
        "pad_size_bytes: {}\n",
        num_units,
        num_elements_per_row,
        unpadded_row_size_bytes,
        padded_row_size_bytes,
        pad_size_bytes);
    DPRINT("Temp Pad pad_addr: {}, out_addr: {}\n", pad_addr, out_addr);
    DPRINT("Output Elem Size {}\n", output_elem_size);
#endif

    static_assert(
        output_elem_size == 1 || output_elem_size == 2 || output_elem_size == 4, "Unsupported output_elem_size");
    if constexpr (output_elem_size == 1) {
        volatile tt_l1_ptr uint8_t* pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(pad_addr);
        for (uint32_t i = 0; i < num_elements_per_row; ++i) {
            pad_ptr[i] = 0;
        }
    } else if constexpr (output_elem_size == 2) {
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

    // pad_size_bytes is runtime; issue each read as a single-packet UnicastEndpoint NOC transfer
    UnicastEndpoint self_ep;
    const auto pad_src = experimental::local_addr(pad_addr + unpadded_row_size_bytes, noc.get_noc_id());

    uint32_t write_offset = unpadded_row_size_bytes;
    for (uint32_t i = 0; i < num_units; ++i) {
        noc.async_read(self_ep, dfb_out, pad_size_bytes, pad_src, {.offset_bytes = write_offset});
        write_offset += padded_row_size_bytes;
    }
    noc.async_read_barrier();
}
