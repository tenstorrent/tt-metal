// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// tt/owned_gather.py's _COMMON, verbatim (NC.7).
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
using namespace tt::tt_fabric;

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a record row (a page of both buffers)
constexpr uint32_t K = get_compile_time_arg_val(1);     // rows a packet at most
constexpr uint32_t N = get_compile_time_arg_val(2);     // chips
constexpr uint32_t C = get_compile_time_arg_val(3);     // mesh columns
constexpr uint32_t AXIS = get_compile_time_arg_val(4);  // 0: the whole mesh is one group, 1: a column
constexpr uint32_t L = get_compile_time_arg_val(5);     // links (port cores) a direction
constexpr uint32_t NSEG = get_compile_time_arg_val(6);  // segment slots
constexpr uint32_t NW = get_compile_time_arg_val(7);    // destination-map words (8 chips a word)
constexpr uint32_t R = get_compile_time_arg_val(8);     // rows an owner slot of the history buffer
constexpr uint32_t B = get_compile_time_arg_val(9);     // chunks a batch
constexpr uint32_t SPIN = get_compile_time_arg_val(10);
constexpr auto o_args = TensorAccessorArgs<11>();
constexpr auto l_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();

// common args (per-run values): [0, NSEG) record buffers, [NSEG, 2 NSEG) lo, [2 NSEG, 3 NSEG) hi, then NW map words
FORCE_INLINE uint32_t seg_of(uint32_t d) {
    return (get_common_arg_val<uint32_t>(3 * NSEG + d / 8) >> (4 * (d % 8))) & 0xFu;
}
FORCE_INLINE bool in_group(uint32_t d, uint32_t me) { return AXIS == 0 || (d % C) == (me % C); }
FORCE_INLINE uint32_t owner(uint32_t d) { return AXIS == 0 ? d : d / C; }
constexpr uint32_t GROUP = AXIS == 0 ? N : N / C;

FORCE_INLINE void wait_min(uint32_t addr, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < SPIN && *p < v; ++i) {
        invalidate_l1_cache();
    }
}

FORCE_INLINE void take_back(uint32_t addr, uint32_t v) {
    noc_semaphore_inc(get_noc_addr(addr), 0u - v);
    noc_async_atomic_barrier();
}

// DRAM banks of an interleaved buffer: page j is in the bank of page 0 again at j = banks (probe, <= 64)
template <typename A>
FORCE_INLINE uint32_t banks_of(const A& acc) {
    const uint64_t a0 = acc.get_noc_addr(0);
    for (uint32_t j = 1; j <= 64; ++j) {
        if (acc.get_noc_addr(j) == a0 + PAGE) {
            return j;
        }
    }
    return 1;
}

// runs of rows [lo, hi): blocks of nb K rows from lo; in a block, bank class b's rows first, first + nb, ... (one
// DRAM bank, contiguous in both buffers). Chunk c = block c / nb, class c % nb.
FORCE_INLINE uint32_t n_chunks(uint32_t lo, uint32_t hi, uint32_t nb) {
    if (hi <= lo) {
        return 0;
    }
    const uint32_t full = (hi - lo) / (nb * K), rem = (hi - lo) - full * nb * K;
    return full * nb + (rem < nb ? rem : nb);
}
FORCE_INLINE void chunk_at(uint32_t c, uint32_t lo, uint32_t hi, uint32_t nb, uint32_t& first, uint32_t& cnt) {
    const uint32_t i0 = lo + (c / nb) * nb * K;
    const uint32_t end = i0 + nb * K < hi ? i0 + nb * K : hi;
    first = i0 + c % nb;
    cnt = (end - first + nb - 1) / nb;
}
