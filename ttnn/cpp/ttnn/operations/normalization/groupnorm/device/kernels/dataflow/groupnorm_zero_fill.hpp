// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

// Zero-fills the entire L1 region backing a local circular buffer by issuing
// a sequence of NOC reads from MEM_ZEROS into successive chunks of the CB.
//
// Used by e.g. reader_mcast_sender_unary_gn.cpp (the non-welford mcast
// reader) for cb_ex_external.
//
// Preconditions:
//   * Must be called before the first cb_reserve_back / cb_push_back on this
//     CB. At that point no consumer has read from it yet, so writing zeros
//     into the full L1 region is race-free.
//   * cb_id refers to a local CB on this core.
//
// Sizing:
//   The function reads the CB's full byte size from the runtime CB interface
//   (fifo_size, in bytes on dataflow cores), so it does NOT depend on any
//   CB sizing parameters from the caller.
//
// Args:
//   cb_id:
//     ID of a local circular buffer on this core. Its full L1 backing region
//     (fifo_size bytes) will be overwritten with zeros.
//   noc:
//     NOC handle to use for the async reads and the trailing read barrier.
//   local_noc_x / local_noc_y:
//     NOC coordinates of this core (the kernel invoking the helper). Used as
//     the source of the MEM_ZEROS read; this core's MEM_ZEROS_BASE is the
//     canonical zero source on Tensix.
inline void zero_whole_cb(uint32_t cb_id, const experimental::Noc& noc, uint32_t local_noc_x, uint32_t local_noc_y) {
    auto& iface = get_local_cb_interface(cb_id);
    // start writing at base CB address, which is the limit - size.
    uint32_t sram_write_addr = iface.fifo_limit - iface.fifo_size;
    uint32_t bytes_remaining = iface.fifo_size;

    experimental::UnicastEndpoint zeros_ep;
    while (bytes_remaining > 0) {
        const uint32_t chunk = bytes_remaining > MEM_ZEROS_SIZE ? MEM_ZEROS_SIZE : bytes_remaining;
        noc.async_read(
            zeros_ep,
            experimental::CoreLocalMem<uint32_t>(sram_write_addr),
            chunk,
            {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = MEM_ZEROS_BASE},
            {});
        sram_write_addr += chunk;
        bytes_remaining -= chunk;
    }
    noc.async_read_barrier();
}
