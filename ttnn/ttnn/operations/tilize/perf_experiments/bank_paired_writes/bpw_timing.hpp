// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_paired_writes STEP 1 — TIMING-ONLY writer issue pattern (output data is WRONG on purpose).
//
// Output TILE page p of a DRAM TensorMemoryLayout::INTERLEAVED buffer lives in bank p % NB at
// offset (p / NB) * page, so pages p, p + NB, p + 2 NB ... are contiguous in ONE bank and can go
// out as one NoC write of up to k pages (k * 2 KiB <= NOC_MAX_BURST_SIZE = 8 KiB on WH).
//
// The writer keeps the op's CB handshake exactly (wait quantum -> issue -> flush -> pop), but per
// quantum it issues the next groups of a precomputed group list until the tiles issued catch up
// with the tiles popped. The L1 source is the CB front (garbage). Every output page is written
// exactly once, so the DRAM traffic volume is the baseline's.
//
// BPW_MODE 0 ("split"): the op's own work split (core owns tiles [t0, t0 + n)); a group is up to
//   k same-bank pages of that range (local l, l + NB, ...). [1,1,16384,64]: 16 tiles/core ->
//   4 pairs + 8 singles at k >= 2.
// BPW_MODE 1 ("ideal"): a bank-aligned re-split: virtual tile v = S*NB*k + i*k + j maps to
//   physical page S*NB*k + j*NB + i, so every core's k-aligned virtual chunk is one bank-contiguous
//   group of k pages (a partial last superblock stays identity, single pages).
// Group issue order: by bank, starting at bank (rot % NB), rot = core index -> concurrent requests
// of the 64 cores spread over the banks. BPW_K == 1 is the control (same order, single pages).

#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace bpw {

constexpr uint32_t NB = 12;  // WH DRAM banks
constexpr uint32_t MAX_GROUPS = 128;

template <
    uint32_t cb,
    uint32_t quantum_pages,
    uint32_t out_tile_bytes,
    uint32_t K,
    uint32_t MODE,
    uint32_t ORDER,
    uint32_t C,
    typename Accessor>
FORCE_INLINE void store_all(
    const Accessor& accessor,
    uint32_t t0,
    uint32_t n,
    uint32_t total_tiles,
    uint32_t rot,
    uint32_t row_rot,
    uint32_t rows) {
    uint16_t g_first[MAX_GROUPS];
    uint8_t g_cnt[MAX_GROUPS];
    uint32_t ng = 0;
    const uint32_t super = NB * K;
    const uint32_t full_end = (total_tiles / super) * super;
    // group of local tile l: (leader local index, physical first page, page count)
    auto group_of = [&](uint32_t l, uint32_t& key, uint32_t& first, uint32_t& cnt) {
        if constexpr (MODE == 0) {
            const uint32_t lead = l - ((l / NB) % K) * NB;
            uint32_t c = 0;
            while (c < K && lead + c * NB < n) {
                ++c;
            }
            key = lead;
            first = t0 + lead;
            cnt = c;
        } else {
            const uint32_t v = t0 + l;
            const uint32_t vs = v - (v % K);
            if (vs + K <= full_end && vs >= t0 && vs + K <= t0 + n) {
                key = vs - t0;
                first = (vs / super) * super + (vs % super) / K;
                cnt = K;
            } else {
                key = l;
                first = v;
                cnt = 1;
            }
        }
    };
    if constexpr (ORDER == 0) {
        // bank order starting at bank rot % NB
        for (uint32_t bb = 0; bb < NB; ++bb) {
            const uint32_t bank = (bb + rot) % NB;
            for (uint32_t l = 0; l < n; ++l) {
                uint32_t key, first, cnt;
                group_of(l, key, first, cnt);
                if (key == l && first % NB == bank) {
                    g_first[ng] = first;
                    g_cnt[ng] = cnt;
                    ++ng;
                }
            }
        }
    } else {
        // the op's store walk order: tile-rows from row_rot, columns from rot % C; a group goes
        // out when its first member comes up
        uint64_t emitted = 0;
        for (uint32_t i = 0; i < rows; ++i) {
            const uint32_t r = (row_rot + i) % rows;
            for (uint32_t j = 0; j < C; ++j) {
                const uint32_t l = r * C + (rot + j) % C;
                uint32_t key, first, cnt;
                group_of(l, key, first, cnt);
                if (!(emitted & (uint64_t(1) << key))) {
                    emitted |= uint64_t(1) << key;
                    g_first[ng] = first;
                    g_cnt[ng] = cnt;
                    ++ng;
                }
            }
        }
    }

    uint32_t gi = 0, issued = 0, popped = 0;
    uint32_t remaining = n;
    while (remaining > 0) {
        const uint32_t pages = remaining < quantum_pages ? remaining : quantum_pages;
        {
            MaybeDeviceZoneScope("bpw_wait");
            cb_wait_front(cb, pages);
        }
        popped += pages;
        {
            MaybeDeviceZoneScope("bpw_issue");
            const uint32_t l1 = get_read_ptr(cb);
            while (gi < ng && issued < popped) {
#if defined(BPW_SPLIT) && BPW_SPLIT
                // control: the SAME address sequence as the group, as g_cnt separate one-page writes
                for (uint32_t c = 0; c < g_cnt[gi]; ++c) {
                    noc_async_write(
                        l1 + c * out_tile_bytes, accessor.get_noc_addr(g_first[gi] + c * NB), out_tile_bytes);
                }
#else
                noc_async_write(l1, accessor.get_noc_addr(g_first[gi]), g_cnt[gi] * out_tile_bytes);
#endif
                issued += g_cnt[gi];
                ++gi;
            }
        }
        {
            MaybeDeviceZoneScope("bpw_flush");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb, pages);
        remaining -= pages;
    }
}

}  // namespace bpw
