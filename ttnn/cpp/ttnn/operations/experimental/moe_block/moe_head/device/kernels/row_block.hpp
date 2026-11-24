// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row blocks of the MoE head: the kernels' loop over a call's rows, a block of rows at a time.
//
// Why: a call whose rows do not fit the head's L1 at once runs as k blocks of rows, one after another, in one launch
// (the head's planner picks k from its L1 fit - plan_moe_head, HeadFit.row_blocks; 1 wherever the rows fit). k is the
// named compile-time arg "row_blocks" of every kernel. The owner-order outputs keep the block layout - block b's owned
// tokens after block b - 1's - which the tail takes as its own (MoEHeadOutputs.row_blocks).
//
// Every MoE head kernel's kernel_main has this shape (the helpers below):
//
//     moe_block_args::load();                    // this core's args from the DRAM table
//     for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
//         row_block::begin(block);              // the block index the shared headers read (row_block::index())
//         ...this kernel's work for one row block...
//         row_block::end(block);                // the chip barrier before the next block; after the last, the reset
//     }
//
// What changes from one block to the next - each kernel computes it from the block index:
//   * x and the shared expert's output: block b's rows are row_block::kXStrideBytes / row_block::kYStrideBytes bytes
//   further in every bank;
//   * the dispatch buffer's rows of this chip's tokens: row_block::kDispatchRowStride rows further (the prep units);
//   * the call's first row in the SP row (a padded forward's span): row_block::kPaddedRowStride rows further;
//   * the router's owner page and route words: row_block::kRouterPageStride pages / row_block::kRouteWordsStride
//   bytes further;
//   * the routing planner plans in the last block only (every block's tokens); earlier blocks run it empty.
// Everything else (CB layouts, slot rings, counters) carries on from block to block as from one row chunk to the next:
// the kernels apply their chunk-boundary rules at a block boundary too, and count ring slots across the blocks.
//
// The barrier between two blocks (BLOCKS > 1, row_block::end). Every RISC of every core finishes block b: its NoC
// traffic flushed (data movement) or its Tensix pipeline idle (compute), then it marks its done word. Each core has one
// lead (its NCRISC, else its BRISC): the lead waits for its core's RISCs, counts in on the coordinator core and waits
// for the go (phase A: no core still runs block b, nothing of block b is in flight). The lead then resets its core's
// program semaphores to 0 and counts in again (phase B: every core reset); after the second go every RISC runs block
// b + 1. That is plain synchronization with public APIs: semaphores and NoC atomics, nothing of the launch state is
// touched.
//
// The words: persistent, zero, the same L1 address on every core (the host's "blocks" words, 16 B apart): RISC r's
// done word at 16 r (BRISC 0, NCRISC 1, TRISC 2 .. 4), the go word at 80, the coordinator's count at 96 (one
// definition with the host's: head_shared_words.hpp). They are back at 0 when the kernels end, for the next launch.
//
// The barrier's table: common runtime args from the named arg row_block_barrier_arg (real common args, after the DRAM
// table's - the kernel's own are in its block), from the host (head_row_blocks.cpp), read once at the first begin()
// (BarrierArgs):
//   [0] the words' address
//   [1] the coordinator: logical x | y << 8 | NoC x << 16 | NoC y << 24
//   [2] the program's cores (the barrier's count)
//   then (a lead kernel, row_block_barrier_lead) the cores it leads: either 1 << 31 and one role for every core it runs
//   on, or n and n x (logical x | y << 8, role); a role = (the core's RISCs: bit r for RISC r, the core's program
//   semaphores: bit s) then (the coordinator kernel, row_block_barrier_coordinator) the cores it releases: n, n x (NoC
//   x | y << 16)
#pragma once

#include <cstdint>
// (every kernel's args: moe_block_args::arg / moe_block_args::common, read from this core's block of the DRAM tables -
// head_workload.cpp head_args_to_dram; kernel_main loads them first; it also brings the compute / dataflow APIs the
// barrier below uses)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
// (the block words' offsets and the table's marker bits, shared with the host - head_row_blocks.cpp, head_program.hpp)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/head_shared_words.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/head_named_args.hpp"

namespace row_block {
constexpr uint32_t BLOCKS = get_named_compile_time_arg_val(moe_head_named_args::kRowBlocks);
static_assert(BLOCKS >= 1, "row blocks: at least one");
// (the host's row block values - head_workload.cpp; 0 with one block)
namespace names = moe_head_named_args;
constexpr uint32_t kBarrierArg = get_named_compile_time_arg_val(names::kRowBlockBarrierArg);
constexpr uint32_t kXStrideBytes = get_named_compile_time_arg_val(names::kRowBlockXStrideBytes);
constexpr uint32_t kYStrideBytes = get_named_compile_time_arg_val(names::kRowBlockYStrideBytes);
constexpr uint32_t kDispatchRowStride = get_named_compile_time_arg_val(names::kRowBlockDispatchRowStride);
constexpr uint32_t kPaddedRowStride = get_named_compile_time_arg_val(names::kRowBlockPaddedRowStride);
constexpr uint32_t kRouterPageStride = get_named_compile_time_arg_val(names::kRowBlockRouterPageStride);
constexpr uint32_t kRouteWordsStride = get_named_compile_time_arg_val(names::kRowBlockRouteWordsStride);
// (this kernel leads its cores' barrier / is the coordinator: data movement kernels only - the host's compute kernels
// have both 0)
constexpr bool kBarrierLead = get_named_compile_time_arg_val(names::kRowBlockBarrierLead) != 0;
constexpr bool kBarrierCoordinator = get_named_compile_time_arg_val(names::kRowBlockBarrierCoordinator) != 0;
inline uint32_t g_current_block = 0;  // (set by begin(): the shared headers read it through index() / is_last())
inline __attribute__((always_inline)) uint32_t index() { return g_current_block; }
inline __attribute__((always_inline)) bool is_last() { return g_current_block + 1 == BLOCKS; }

// (this RISC's done word: STRIDE x its index)
#if defined(COMPILE_FOR_BRISC)
constexpr uint32_t MY_RISC = 0;
#elif defined(COMPILE_FOR_NCRISC)
constexpr uint32_t MY_RISC = 1;
#elif defined(UCK_CHLKC_UNPACK)
constexpr uint32_t MY_RISC = 2;
#elif defined(UCK_CHLKC_MATH)
constexpr uint32_t MY_RISC = 3;
#else
constexpr uint32_t MY_RISC = 4;
#endif
constexpr uint32_t RISCS = moe_head_row_block_words::kRiscs, WORDS = RISCS + 1;  // (the RISCs' done words, the go word)
constexpr uint32_t STRIDE = moe_head_row_block_words::kStride;                   // (bytes between the words)
constexpr uint32_t GO_WORD = moe_head_row_block_words::kGoWord, COUNT_WORD = moe_head_row_block_words::kCountWord;
static_assert(WORDS == 6, "the done words of the 5 RISCs, then the go word");

// The barrier's table (BLOCKS > 1): the common args from kBarrierArg, read once - at the first block's
// begin() - into g_barrier_args; the roles / release lists past [2] are read in place (a view of the table).
struct BarrierArgs {
    uint32_t words_address = 0;  // [0] the block words' address
    uint32_t coordinator = 0;    // [1] the coordinator: logical x | y << 8 | NoC x << 16 | NoC y << 24
    uint32_t program_cores = 0;  // [2] the program's cores (the barrier's count)
    const volatile tt_l1_ptr uint32_t* table = nullptr;  // the whole table, [0] on
    // (lead) the roles from [3]: kEveryCoreAlike (1 << 31) and one role for every core alike, or n and n x (core, role)
    bool every_core_alike() const { return table[3] >> moe_head_row_block_words::kEveryCoreAlikeShift; }
    uint32_t role_cores() const { return table[3]; }
    uint32_t role_core(uint32_t core) const { return table[4 + 3 * core]; }  // (logical x | y << 8)
    uint32_t role_riscs(uint32_t core) const { return table[5 + 3 * core]; }
    uint32_t role_semaphores(uint32_t core) const { return table[6 + 3 * core]; }
    // (coordinator) the release list, past the roles: n, then n x (NoC x | y << 16)
    uint32_t releases_at() const { return every_core_alike() ? 6 : 4 + 3 * role_cores(); }
    uint32_t released_cores() const { return table[releases_at()]; }
    uint32_t released_core_noc_xy(uint32_t core) const { return table[releases_at() + 1 + core]; }
};
inline BarrierArgs g_barrier_args;

inline __attribute__((always_inline)) volatile tt_l1_ptr uint32_t* word(uint32_t offset) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_barrier_args.words_address + offset);
}
inline __attribute__((always_inline)) void wait_at_least(uint32_t offset, uint32_t value) {
    while (*word(offset) < value) {
#ifndef COMPILE_FOR_TRISC
        invalidate_l1_cache();  // (the data movement RISCs' L1 data cache; the compute RISCs have none)
#endif
    }
}

// (every RISC) my block is over here: my NoC traffic out of L1 and acked, or my Tensix pipeline idle
__attribute__((noinline)) inline void finish() {
#ifndef COMPILE_FOR_TRISC
    noc_async_full_barrier();
#else
    ckernel::tensix_sync();
#endif
}

#if !defined(COMPILE_FOR_TRISC)
struct Role {
    uint32_t riscs = 0, semaphores = 0;
};
inline Role g_role;  // (my core's role if this kernel leads it - riscs 0: not the lead; set with g_barrier_args)
__attribute__((noinline)) inline Role my_role() {
    if (g_barrier_args.every_core_alike()) {
        return Role{g_barrier_args.table[4], g_barrier_args.table[5]};
    }
    const uint32_t me = moe_head_row_block_words::kLogicalX.pack(get_absolute_logical_x()) |
                        moe_head_row_block_words::kLogicalY.pack(uint32_t(get_absolute_logical_y()));
    for (uint32_t core = 0, cores = g_barrier_args.role_cores(); core < cores; ++core) {
        if (g_barrier_args.role_core(core) == me) {
            return Role{g_barrier_args.role_riscs(core), g_barrier_args.role_semaphores(core)};
        }
    }
    return Role{};
}
// (lead) one phase of the barrier: count in on the coordinator, which releases every core with go = value
__attribute__((noinline)) inline void chip_phase(uint32_t value) {
    namespace words = moe_head_row_block_words;  // (the coordinator / release words' fields, shared with the host)
    const uint32_t coordinator = g_barrier_args.coordinator;
    noc_semaphore_inc(
        get_noc_addr(
            words::kCoordinatorNocX.get(coordinator),
            words::kCoordinatorNocY.get(coordinator),
            g_barrier_args.words_address + COUNT_WORD),
        1);
    noc_async_atomic_barrier();
    if (kBarrierCoordinator && words::kLogicalX.get(coordinator) == get_absolute_logical_x() &&
        words::kLogicalY.get(coordinator) == get_absolute_logical_y()) {
        const uint32_t cores = g_barrier_args.program_cores;
        wait_at_least(COUNT_WORD, cores);
        noc_semaphore_inc(
            get_noc_addr(g_barrier_args.words_address + COUNT_WORD), 0u - cores);  // (re-armed before any release)
        noc_async_atomic_barrier();
        for (uint32_t core = 0, released = g_barrier_args.released_cores(); core < released; ++core) {
            const uint32_t noc_xy = g_barrier_args.released_core_noc_xy(core);
            noc_inline_dw_write(
                get_noc_addr(
                    words::kReleaseNocX.get(noc_xy),
                    words::kReleaseNocY.get(noc_xy),
                    g_barrier_args.words_address + GO_WORD),
                value);
        }
        noc_async_write_barrier();
    }
    wait_at_least(GO_WORD, value);
}
#endif

// (BLOCKS > 1, the first block's begin) the barrier's table, once
__attribute__((noinline)) inline void read_barrier_args() {
    const volatile tt_l1_ptr uint32_t* table =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(get_common_arg_addr(kBarrierArg));
    g_barrier_args.table = table;
    g_barrier_args.words_address = table[0];
    g_barrier_args.coordinator = table[1];
    g_barrier_args.program_cores = table[2];
#if !defined(COMPILE_FOR_TRISC)
    if constexpr (kBarrierLead) {
        g_role = my_role();
    }
#endif
}

// between block b and b + 1
__attribute__((noinline)) inline void between(uint32_t block) {
    finish();
    const uint32_t phase_a = 2 * block + 1, phase_b = 2 * block + 2;
    *word(STRIDE * MY_RISC) = phase_a;
#if !defined(COMPILE_FOR_TRISC)
    if constexpr (kBarrierLead) {
        if (const Role role = g_role; role.riscs) {  // the lead: my core's RISCs, the chip; my semaphores; the chip
            for (uint32_t risc = 0; risc < RISCS; ++risc) {
                if (role.riscs >> risc & 1) {
                    wait_at_least(STRIDE * risc, phase_a);
                }
            }
            chip_phase(phase_a);
            for (uint32_t semaphore = 0; semaphore < 32; ++semaphore) {
                if (role.semaphores >> semaphore & 1) {
                    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore)), 0);
                }
            }
            chip_phase(phase_b);
            return;
        }
    }
#endif
    wait_at_least(GO_WORD, phase_b);
}

// after the last block: the words back to 0 for the next launch (the lead, once its core's RISCs are done)
__attribute__((noinline)) inline void after_last() {
    finish();
    const uint32_t last_phase = 2 * BLOCKS - 1;
    *word(STRIDE * MY_RISC) = last_phase;
#if !defined(COMPILE_FOR_TRISC)
    if constexpr (kBarrierLead) {
        if (const Role role = g_role; role.riscs) {
            for (uint32_t risc = 0; risc < RISCS; ++risc) {
                if (role.riscs >> risc & 1) {
                    wait_at_least(STRIDE * risc, last_phase);
                }
            }
            for (uint32_t word_index = 0; word_index < WORDS; ++word_index) {
                *word(STRIDE * word_index) = 0;
            }
        }
    }
#endif
}

// At the top of each block: the block the shared headers see (and, at the first of several, the barrier's table).
inline __attribute__((always_inline)) void begin(uint32_t block) {
    g_current_block = block;
    if constexpr (BLOCKS > 1) {
        if (block == 0) {
            read_barrier_args();
        }
    }
}

// At the end of each block: the chip barrier before the next block, or after the last block the barrier words' reset
// (nothing with one block).
inline __attribute__((always_inline)) void end([[maybe_unused]] uint32_t block) {
    if constexpr (BLOCKS > 1) {
        if (block + 1 < BLOCKS) {
            between(block);
        } else {
            after_last();
        }
    }
}
}  // namespace row_block
