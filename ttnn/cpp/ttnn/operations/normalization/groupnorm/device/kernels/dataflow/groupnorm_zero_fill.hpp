// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

// Zero-fills the entire L1 region backing a local circular buffer via
// Noc::async_write_zeros.
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
inline void zero_whole_cb(uint32_t cb_id, const Noc& noc) {
    auto& iface = get_local_cb_interface(cb_id);
    CircularBuffer cb(cb_id);
    noc.async_write_zeros(cb, iface.fifo_size);
    noc.write_zeros_l1_barrier();
}
