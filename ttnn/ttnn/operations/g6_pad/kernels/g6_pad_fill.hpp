// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace g6_pad {

// Fills bytes [offset, offset + n_bytes) of cb_id's write slot with the element pattern c.
// c = 0 (every production extension call) goes through the NoC zero path, a loopback read of the
// firmware zero region, because fill_l1_range's RISC-V stores cost ~3 cycles per word and sit on
// the critical path of every core. offset must be 16-B aligned (NoC L1 read alignment).
template <uint32_t elem_bytes>
FORCE_INLINE void fill_constant(uint32_t cb_id, uint32_t offset, uint32_t n_bytes, uint32_t fill_bits) {
    if (fill_bits == 0) {
        Noc noc;
        CircularBuffer cb(cb_id);
        noc.async_write_zeros(cb, n_bytes, {.offset_bytes = offset});
        noc.write_zeros_l1_barrier();
    } else {
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(get_write_ptr(cb_id) + offset, n_bytes, fill_bits);
    }
}

}  // namespace g6_pad
