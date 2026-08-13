// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

// Merges consecutive transfers into the largest possible ones.
//
// Contiguity is *measured*, never predicted: two transfers merge only when both ends continue
// exactly where the previous pair finished. That single test is the whole trick -- a bank
// round-robin, a shard edge, a stripe edge and inter-page alignment padding each break the run on
// their own, so interleaved / sharded / ND / DRAM / L1 / concat / split need no special case here,
// in the caller, or on the host. There is nothing to get wrong per layout because nothing is
// assumed about any layout.
//
// `max_bytes` bounds one emitted transfer: a fabric packet payload, or NOC_MAX_BURST_SIZE for a
// local read/write. A single chunk larger than that is emitted in `max_bytes` pieces, which is why
// callers never need their own "page bigger than a packet" path.
//
// This sits in the innermost loop of both kernels and runs once per chunk, so it is force-inlined
// on purpose: letting `bytes` stop being a compile-time constant here cost 15% on a small-page
// shape. Measure before relaxing that.
// `merge` off makes add() a pass-through, for the configurations where neighbouring chunks can never
// be adjacent anyway (interleaved output walked in page order). Under-merging is always correct, so
// this is a pure performance hint.
template <uint32_t max_bytes, bool merge, typename Emit>
class RunCoalescer {
public:
    explicit RunCoalescer(Emit emit) : emit_{emit} {}

    ~RunCoalescer() { ASSERT(bytes_ == 0); }  // outstanding run! flush() not called

    FORCE_INLINE void add(uint32_t local_addr, uint64_t remote_addr, uint32_t bytes) {
        while (bytes > max_bytes) {
            flush();
            emit_(local_addr, remote_addr, max_bytes);
            local_addr += max_bytes;
            remote_addr += max_bytes;
            bytes -= max_bytes;
        }
        if constexpr (merge) {
            if (bytes_ != 0 && local_addr == local_ + bytes_ && remote_addr == remote_ + bytes_ &&
                bytes_ + bytes <= max_bytes) {
                bytes_ += bytes;
                return;
            }
            flush();
            local_ = local_addr;
            remote_ = remote_addr;
            bytes_ = bytes;
        } else {
            emit_(local_addr, remote_addr, bytes);
        }
    }

    FORCE_INLINE void flush() {
        if (bytes_ != 0) {
            emit_(local_, remote_, bytes_);
            bytes_ = 0;
        }
    }

private:
    Emit emit_;
    uint32_t local_ = 0;
    uint64_t remote_ = 0;
    uint32_t bytes_ = 0;
};
