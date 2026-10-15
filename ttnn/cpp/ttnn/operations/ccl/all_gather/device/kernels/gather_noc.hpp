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

// `slot_bytes` is the chunk size: a run of `chunks` chunks is chunks * slot_bytes contiguous bytes on
// both sides. `one_command`: the whole run fits one NoC command. When a chunk exceeds a burst the walk
// only ever emits runs of one chunk (chunks_per_transfer() is 1), so the generic path moves exactly one
// chunk and lets the NoC API split it.

// Reads one run into L1. `opts`/`vals` carry the transaction id for the trid-pipelined reader.
template <
    uint32_t slot_bytes,
    bool one_command = (slot_bytes <= NOC_MAX_BURST_SIZE),
    NocOptions opts = NocOptions::DEFAULT>
FORCE_INLINE void read_run(
    const Noc& noc, uint64_t src_noc_addr, uint32_t l1_addr, uint32_t chunks, const NocOptVals& vals = {}) {
    if constexpr (one_command) {
        noc.async_read<opts, NOC_MAX_BURST_SIZE>(
            tensor_accessor::Page(src_noc_addr, 0), CoreLocalMem<uint32_t>(l1_addr), chunks * slot_bytes, {}, {}, vals);
    } else {
        noc.async_read<opts>(
            tensor_accessor::Page(src_noc_addr, 0), CoreLocalMem<uint32_t>(l1_addr), slot_bytes, {}, {}, vals);
    }
}

// Same, tagged with transaction id `trid` (multicast reader's in-flight CB pipeline).
template <uint32_t slot_bytes, bool one_command = (slot_bytes <= NOC_MAX_BURST_SIZE)>
FORCE_INLINE void read_run(const Noc& noc, uint64_t src_noc_addr, uint32_t l1_addr, uint32_t chunks, uint32_t trid) {
    read_run<slot_bytes, one_command, NocOptions::TXN_ID>(noc, src_noc_addr, l1_addr, chunks, {.trid = trid});
}

// Writes one run from L1 into this device's output. Posted, on a separate VC so it doesn't contend with
// the fabric writes on the same NOC; callers flush with async_writes_flushed<NocOptions::POSTED>().
template <uint32_t slot_bytes, bool one_command = (slot_bytes <= NOC_MAX_BURST_SIZE)>
FORCE_INLINE void local_write(const Noc& noc, uint32_t l1_addr, uint64_t dst_noc_addr, uint32_t chunks) {
    if constexpr (one_command) {
        noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
            CoreLocalMem<uint32_t>(l1_addr),
            tensor_accessor::Page(dst_noc_addr, 0),
            chunks * slot_bytes,
            {},
            {},
            {.vc = NOC_UNICAST_WRITE_VC + 1});
    } else {
        noc.async_write<NocOptions::POSTED | NocOptions::CUSTOM_VC>(
            CoreLocalMem<uint32_t>(l1_addr),
            tensor_accessor::Page(dst_noc_addr, 0),
            slot_bytes,
            {},
            {},
            {.vc = NOC_UNICAST_WRITE_VC + 1});
    }
}
