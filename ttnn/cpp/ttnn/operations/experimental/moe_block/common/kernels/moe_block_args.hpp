// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A kernel's per-core runtime values, read from a DRAM table instead of runtime args (common/args_table.hpp).
//
// Why: runtime args are copied into every chip's dispatch queue on every launch; values that only depend on the plan
// are written once to a table in each chip's DRAM instead, and the kernel reads its own core's block when it starts.
//
// Use: call moe_block_args::load() first thing, then moe_block_args::arg(i) wherever the kernel used
// get_arg_val<uint32_t>(i) (same indices: the block holds the core's args as the host built them). With the named
// compile-time arg moe_block_args_enabled 0 (the kernel's args left as runtime args) both fall back to the runtime
// args, so a kernel builds either way.
//
// The named compile-time args (kernels/args_table_names.hpp): the host gives every kernel that includes this file all
// of them (fill_args_table_named_args), the value 0 for what the kernel does not use.
//
// The block: the core's args as the host built them, with every address arg holding only its offset, then at word
// moe_block_args_fixups_at the fixups: n, then n entries (arg index | source << 16).
//
// The common args at moe_block_args_common_index: [0] the table's address, [1] a block's bytes, [2] the DRAM banks,
// [3] the grid width, [4] the L1 address the block lands at (unused with moe_block_args_block_in_cb), [5] the number
// of sources S, then the S source addresses (all rewritten by the host on a program-cache hit).
//
// What load() does: reads the block of this core (page y * grid width + x of the table, the pages interleaved over the
// DRAM banks) into L1 - CB moe_block_args_block_cb at byte moe_block_args_block_cb_offset when
// moe_block_args_block_in_cb is 1, else the common arg's address - then adds each fixup's source address to its arg.
// Public dataflow APIs only.
//
// arg_pointer(i) / common_pointer(i): where arg i sits in L1 - a kernel reads its fixed args once into a named struct
// and hands helpers a pointer to a variable-length table of args (read in place, never an index into the arg list).
// A kernel whose args hold several blocks (the combine's port-or-accumulator cores) picks its own with set_base().
//
// Also (the MoE head, whose chips differ in common arg values too): moe_block_args::common(i) for
// get_common_arg_val(i) - with moe_block_args_common_in_block 1 the kernel's common args sit in its block from word
// moe_block_args_common_at on. A compute kernel cannot read DRAM: the data-movement kernel on its core
// (moe_block_args_peer_page_sender 1) hands it its args as one page of CB moe_block_args_peer_cb after its own load(),
// and the compute kernel (moe_block_args_compute_reads_page 1) reads them from that page.
#pragma once

#include <cstdint>

#include "args_table_names.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compile_time_args.h"

namespace moe_block_args {

inline constexpr bool kComputeReadsPage = get_named_compile_time_arg_val(moe_block_args_names::kComputeReadsPage) != 0;
inline constexpr uint32_t kPeerCb = get_named_compile_time_arg_val(moe_block_args_names::kPeerCb);
inline constexpr uint32_t kCommonAt = get_named_compile_time_arg_val(moe_block_args_names::kCommonAt);

// (compute, with kComputeReadsPage) the page the data-movement kernel pushed: this core's runtime args, its common
// args from kCommonAt on
inline volatile tt_l1_ptr uint32_t* g_block = nullptr;
inline void load() {
    if constexpr (kComputeReadsPage) {
        cb_wait_front(kPeerCb, 1);  // (never popped: the page holds the args for the whole launch)
        g_block = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(kPeerCb, 0));
    }
}
inline __attribute__((always_inline)) uint32_t arg(uint32_t i) {
    if constexpr (kComputeReadsPage) {
        return g_block[i];
    } else {
        return get_arg_val<uint32_t>(i);
    }
}
inline __attribute__((always_inline)) uint32_t common(uint32_t i) {
    if constexpr (kComputeReadsPage) {
        return g_block[kCommonAt + i];
    } else {
        return get_common_arg_val<uint32_t>(i);
    }
}
inline __attribute__((always_inline)) volatile tt_l1_ptr uint32_t* arg_pointer(uint32_t i) {
    if constexpr (kComputeReadsPage) {
        return g_block + i;
    } else {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(i));
    }
}
inline __attribute__((always_inline)) volatile tt_l1_ptr uint32_t* common_pointer(uint32_t i) {
    if constexpr (kComputeReadsPage) {
        return g_block + kCommonAt + i;
    } else {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_addr(i));
    }
}

}  // namespace moe_block_args

#else
#include "api/dataflow/dataflow_api.h"

namespace moe_block_args {

// (see the top of this file)
inline constexpr bool kEnabled = get_named_compile_time_arg_val(moe_block_args_names::kEnabled) != 0;
inline constexpr uint32_t kCommonIndex = get_named_compile_time_arg_val(moe_block_args_names::kCommonIndex);
inline constexpr uint32_t kFixupsAt = get_named_compile_time_arg_val(moe_block_args_names::kFixupsAt);
inline constexpr bool kBlockInCb = get_named_compile_time_arg_val(moe_block_args_names::kBlockInCb) != 0;
inline constexpr uint32_t kBlockCb = get_named_compile_time_arg_val(moe_block_args_names::kBlockCb);
inline constexpr uint32_t kBlockCbOffset = get_named_compile_time_arg_val(moe_block_args_names::kBlockCbOffset);
inline constexpr bool kCommonInBlock = get_named_compile_time_arg_val(moe_block_args_names::kCommonInBlock) != 0;
inline constexpr uint32_t kCommonAt = get_named_compile_time_arg_val(moe_block_args_names::kCommonAt);
inline constexpr bool kPeerPageSender = get_named_compile_time_arg_val(moe_block_args_names::kPeerPageSender) != 0;
inline constexpr uint32_t kPeerAt = get_named_compile_time_arg_val(moe_block_args_names::kPeerAt);
inline constexpr uint32_t kPeerWords = get_named_compile_time_arg_val(moe_block_args_names::kPeerWords);
inline constexpr uint32_t kPeerCb = get_named_compile_time_arg_val(moe_block_args_names::kPeerCb);

inline volatile tt_l1_ptr uint32_t* g_block = nullptr;
inline uint32_t g_base = 0;

// (kEnabled) where the block lands in L1
FORCE_INLINE uint32_t block_l1_address() {
    if constexpr (kBlockInCb) {
        return get_write_ptr(kBlockCb) + kBlockCbOffset;
    } else {
        return get_common_arg_val<uint32_t>(kCommonIndex + 4);
    }
}

inline void load() {
    if constexpr (kEnabled) {
        constexpr uint32_t a = kCommonIndex;
        const uint32_t table = get_common_arg_val<uint32_t>(a), block = get_common_arg_val<uint32_t>(a + 1);
        const uint32_t banks = get_common_arg_val<uint32_t>(a + 2), gx = get_common_arg_val<uint32_t>(a + 3);
        const uint32_t l1 = block_l1_address();
        const uint32_t page = get_absolute_logical_y() * gx + get_absolute_logical_x();
        noc_async_read(get_noc_addr_from_bank_id<true>(page % banks, table + (page / banks) * block), l1, block);
        noc_async_read_barrier();
        g_block = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
        g_base = 0;
        const uint32_t n = g_block[kFixupsAt];
        for (uint32_t k = 0; k < n; ++k) {  // (this launch's buffer addresses onto their args' offsets)
            const uint32_t e = g_block[kFixupsAt + 1 + k];
            g_block[e & 0xFFFF] += get_common_arg_val<uint32_t>(a + 6 + (e >> 16));
        }
        if constexpr (kPeerPageSender) {
            // (the compute kernel's args, from word kPeerAt of this block: one page of its CB, fixups applied)
            cb_reserve_back(kPeerCb, 1);
            volatile tt_l1_ptr uint32_t* peer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kPeerCb));
            for (uint32_t k = 0; k < kPeerWords; ++k) {
                peer[k] = g_block[kPeerAt + k];
            }
            cb_push_back(kPeerCb, 1);
        }
    }
}

// (set_base offsets the args of a kernel with several blocks)
FORCE_INLINE void set_base(uint32_t off) { g_base = off; }
FORCE_INLINE uint32_t arg(uint32_t i) {
    if constexpr (kEnabled) {
        return g_block[g_base + i];
    } else {
        return get_arg_val<uint32_t>(g_base + i);
    }
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* arg_pointer(uint32_t i) {
    if constexpr (kEnabled) {
        return g_block + g_base + i;
    } else {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(g_base + i));
    }
}
FORCE_INLINE uint32_t common(uint32_t i) {
    if constexpr (kEnabled && kCommonInBlock) {
        return g_block[kCommonAt + i];
    } else {
        return get_common_arg_val<uint32_t>(i);
    }
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* common_pointer(uint32_t i) {
    if constexpr (kEnabled && kCommonInBlock) {
        return g_block + kCommonAt + i;
    } else {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_addr(i));
    }
}

}  // namespace moe_block_args
#endif
