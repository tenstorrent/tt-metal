// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar permute, the one kernel: runs on every DM thread of the cluster. Thread t moves blocks
// first_block + t, first_block + t + T, ... of the cluster's range, each through its own pair of L1 windows.
// The block walk itself is permute_block_engine.hpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include "ttnn/operations/experimental/quasar/permute/device/kernels/permute_block_engine.hpp"

namespace {

// NoC + L1-window side of the block walk for one thread.
template <typename Src, typename Dst>
struct ThreadIO {
    Noc noc;
    const Src& src;
    const Dst& dst;
    uint32_t in_addr;   // L1 address of the read window
    uint32_t out_addr;  // L1 address of the write window

    void read(uint32_t page, uint32_t offset_bytes, uint32_t len_bytes, uint32_t window_offset) {
        noc.async_read(
            src,
            CoreLocalMem<uint32_t>(in_addr + window_offset),
            len_bytes,
            {.page_id = page, .offset_bytes = offset_bytes},
            {});
    }
    void read_barrier() { noc.async_read_barrier(); }

    void write(
        uint32_t page, uint32_t offset_bytes, uint32_t len_bytes, bool from_read_window, uint32_t window_offset) {
        noc.async_write(
            CoreLocalMem<uint32_t>((from_read_window ? in_addr : out_addr) + window_offset),
            dst,
            len_bytes,
            {},
            {.page_id = page, .offset_bytes = offset_bytes});
    }
    void write_barrier() { noc.async_write_barrier(); }

    // The element reorder goes through the uncached L1 alias: a NoC read lands in L1 behind the hart's data
    // cache, and a NoC write reads L1 behind it too. The fence orders the reorder's stores ahead of the writes.
    void publish() { __asm__ __volatile__("fence" ::: "memory"); }
    uint8_t* in_window() { return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(in_addr) + MEM_L1_UNCACHED_BASE); }
    uint8_t* out_window() {
        return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(out_addr) + MEM_L1_UNCACHED_BASE);
    }
};

}  // namespace

void kernel_main() {
    namespace plan_words = ttnn::prim::qsr::permute_plan;

    constexpr uint32_t elem_size = get_arg(args::elem_size);
    const uint32_t first_block = get_arg(args::first_block);
    const uint32_t num_blocks = get_arg(args::num_blocks);
    const auto plan = get_arg(args::plan);

    const uint32_t thread = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();

    const Scratchpad<uint8_t> windows(scratch::windows);
    uintptr_t windows_addr = local_mem::detail::AddressAccess::address(windows.local_mem());
    if (windows_addr >= MEM_L1_UNCACHED_BASE) {
        windows_addr -= MEM_L1_UNCACHED_BASE;
    }
    const uint32_t in_bytes = plan[plan_words::kInBufBytes];
    const uint32_t out_bytes = plan[plan_words::kOutBufBytes];
    const uint32_t in_addr = static_cast<uint32_t>(windows_addr) + thread * (in_bytes + out_bytes);

    const auto src = TensorAccessor(tensor::input);
    const auto dst = TensorAccessor(tensor::output);
    ThreadIO<decltype(src), decltype(dst)> io{Noc{}, src, dst, in_addr, in_addr + in_bytes};

    ttnn::prim::qsr::permute_engine::run_blocks<elem_size>(
        plan, io, first_block + thread, first_block + num_blocks, num_threads);
}
