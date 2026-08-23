// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/pool/device/kernels/experimental_device_api.hpp"

template <uint32_t output_elem_size>
TT_KERNEL void writer(
    uint32_t num_units,
    uint32_t num_elements_per_row,
    uint32_t unpadded_row_size_bytes,
    uint32_t padded_row_size_bytes) {
    const uint32_t pad_size_bytes = padded_row_size_bytes - unpadded_row_size_bytes;
    DataflowBuffer output(dfb::output);
    Scratchpad<uint32_t> padding(scratch::padding);

    const uint32_t pad_addr = padding.get_base_address();
    if (pad_size_bytes == 0) {
        // No padding needed, exit early.
        return;
    }

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

    // pad_size_bytes is runtime; issue each read as a single-packet UnicastEndpoint NOC transfer.
    Noc noc;
    UnicastEndpoint self_ep;
    const auto pad_src = experimental::local_addr(pad_addr + unpadded_row_size_bytes, noc.get_noc_id());

    uint32_t write_offset = unpadded_row_size_bytes;
    for (uint32_t i = 0; i < num_units; ++i) {
        noc.async_read(self_ep, output, pad_size_bytes, pad_src, {.offset_bytes = write_offset});
        write_offset += padded_row_size_bytes;
    }
    noc.async_read_barrier();
}
