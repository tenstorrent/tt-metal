// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_protocol.hpp"
#include "api/dataflow/noc.h"

namespace dataflow_kernel_lib {

// Guard protects source L1 before returning. CallerManaged lets the caller provide that
// protection; modes may still need completion barriers for data-before-ready ordering.
enum class SourceL1Guard { Guard, CallerManaged };

namespace detail {

// Returns once the NoC has read the L1 source of every untagged (trid 0) write this core issued,
// so the source can be reused. On Wormhole the NIU has been seen reading a source after
// NIU_MST_WRITE_REQS_OUTGOING_ID drops (see #63951), so wait for write acks there.
template <typename NocT>
FORCE_INLINE void wait_write_sources_read(NocT& noc) {
#ifdef ARCH_WORMHOLE
    noc.async_write_barrier();
#else
    noc.template async_writes_flushed<NocOptions::TXN_ID>({.trid = 0});
#endif
}

}  // namespace detail

}  // namespace dataflow_kernel_lib
