// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device-only: one run as one NoC command, shared by both all_gather algorithms.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/page.h"
#include "api/core_local_mem.h"

#include <cstdint>

// A run of `chunks` chunks occupies `chunks` CB slots of `slot_bytes` (the chunk, or a padded output page).
// When a slot fits one NoC command (slot_bytes <= NOC_MAX_BURST_SIZE) the whole run is one command: the host caps
// a column at one burst (check_walk_ct). A larger slot gets xfer 1 from the host, so the walk only ever emits runs
// of one chunk there, and the NoC API splits that one chunk.

// Reads one run into L1: (chunks - 1) * slot_bytes + last_bytes contiguous bytes. `last_bytes` is the
// chunk for an input read (a multi-chunk input run exists only where the input steps by the slot, and the
// last chunk's pad, if any, is not read), and the slot for a relay read of our own output.
// `opts`/`vals` carry the transaction id for the trid-pipelined reader.
template <uint32_t slot_bytes, uint32_t last_bytes = slot_bytes, NocOptions opts = NocOptions::DEFAULT>
FORCE_INLINE void read_run(
    const Noc& noc, uint64_t src_noc_addr, uint32_t l1_addr, uint32_t chunks, const NocOptVals& vals = {}) {
    static_assert(last_bytes <= slot_bytes, "a chunk fits its slot");
    if constexpr (slot_bytes <= NOC_MAX_BURST_SIZE) {
        noc.async_read<opts, NOC_MAX_BURST_SIZE>(
            tensor_accessor::Page(src_noc_addr, 0),
            CoreLocalMem<uint32_t>(l1_addr),
            (chunks - 1) * slot_bytes + last_bytes,
            {},
            {},
            vals);
    } else {
        ASSERT(chunks == 1);
        noc.async_read<opts>(
            tensor_accessor::Page(src_noc_addr, 0), CoreLocalMem<uint32_t>(l1_addr), last_bytes, {}, {}, vals);
    }
}

// Same, tagged with transaction id `trid` (multicast reader's in-flight CB pipeline).
template <uint32_t slot_bytes, uint32_t last_bytes = slot_bytes>
FORCE_INLINE void read_run(const Noc& noc, uint64_t src_noc_addr, uint32_t l1_addr, uint32_t chunks, uint32_t trid) {
    read_run<slot_bytes, last_bytes, NocOptions::TXN_ID>(noc, src_noc_addr, l1_addr, chunks, {.trid = trid});
}

// Writes one run from L1 into this device's output: chunks * slot_bytes bytes (a padded slot's pad lands in
// the output page's own pad). Posted, on a separate VC so it doesn't contend with the fabric writes on the
// same NOC; callers flush with async_writes_flushed<NocOptions::POSTED>().
template <uint32_t slot_bytes>
FORCE_INLINE void local_write(const Noc& noc, uint32_t l1_addr, uint64_t dst_noc_addr, uint32_t chunks) {
    if constexpr (slot_bytes <= NOC_MAX_BURST_SIZE) {
        noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
            CoreLocalMem<uint32_t>(l1_addr),
            tensor_accessor::Page(dst_noc_addr, 0),
            chunks * slot_bytes,
            {},
            {},
            {.vc = NOC_UNICAST_WRITE_VC + 1});
    } else {
        ASSERT(chunks == 1);
        noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC>(
            CoreLocalMem<uint32_t>(l1_addr),
            tensor_accessor::Page(dst_noc_addr, 0),
            slot_bytes,
            {},
            {},
            {.vc = NOC_UNICAST_WRITE_VC + 1});
    }
}
