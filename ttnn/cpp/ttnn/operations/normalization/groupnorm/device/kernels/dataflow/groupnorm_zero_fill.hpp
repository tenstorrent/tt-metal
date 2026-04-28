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
// Intended for cb_ex_external in the groupnorm reader kernels. Those readers
// only ever write to a fixed subset of byte positions per iteration (the
// per-core slot data positions); all other bytes -- the "intra-slot gap" of
// each per-core slot when datum_size_bytes < 16, and any "trailing tile gap"
// past the last used slot -- are read by the downstream reduce_tile sum but
// never touched by the producer. Pre-zeroing the whole CB once at kernel
// startup guarantees those gap bytes stay zero across every iteration: the
// consumer is read-only on the CB, and the producer's per-iter writes never
// land on a gap byte.
//
// Preconditions:
//   * Must be called before the first cb_reserve_back / cb_push_back on this
//     CB. At that point no consumer has read from it yet, so writing zeros
//     into the full L1 region is race-free.
//   * cb_id refers to a local CB on this core.
//
// Sizing:
//   The function reads the CB's full byte size from the runtime CB interface
//   (fifo_size, in bytes on dataflow cores), so it does NOT depend on the
//   "page count == per-iteration reservation" invariant of the calling
//   reader. If the CB is ever resized (e.g. double-buffered) the helper
//   still zeros every L1 byte the producer can later write into.
//
// local_noc_x / local_noc_y:
//   NOC coordinates of THIS core (the sender invoking the helper). Used as
//   the source of the MEM_ZEROS read; this core's MEM_ZEROS_BASE is the
//   canonical zero source on Tensix.
inline void zero_whole_cb(uint32_t cb_id, experimental::Noc& noc, uint32_t local_noc_x, uint32_t local_noc_y) {
    auto& iface = get_local_cb_interface(cb_id);
    uint32_t l1_write_addr = iface.fifo_limit - iface.fifo_size;
    uint32_t bytes_remaining = iface.fifo_size;

    experimental::UnicastEndpoint zeros_ep;
    while (bytes_remaining > 0) {
        const uint32_t chunk = bytes_remaining > MEM_ZEROS_SIZE ? MEM_ZEROS_SIZE : bytes_remaining;
        noc.async_read(
            zeros_ep,
            experimental::CoreLocalMem<uint32_t>(l1_write_addr),
            chunk,
            {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = MEM_ZEROS_BASE},
            {});
        l1_write_addr += chunk;
        bytes_remaining -= chunk;
    }
    noc.async_read_barrier();
}
