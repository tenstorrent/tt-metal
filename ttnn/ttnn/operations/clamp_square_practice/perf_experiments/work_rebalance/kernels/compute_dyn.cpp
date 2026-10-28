// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// work_rebalance experiment — DYNAMIC compute. The per-core block count is unknown at launch, so every
// TRISC (unpack, math, pack — all three must run the same number of chain iterations) polls two local
// L1 words written by the reader: SEM_PUBLISHED (blocks claimed so far) and SEM_DONE. Read order is
// done-then-published; the reader writes published before done, so done==1 implies published is final.
// TRISCs have no get_semaphore(); the address is rebuilt from the launch message exactly like firmware
// does (kernel_config_base + sem_offset). Semaphores are reset by dispatch before every launch.
// One eltwise_chain call per block (re-emits inits per block — a cost of the scheme, measured).
// CT: [cb_in, cb_out, block_tiles]   RT: [min_bits, max_bits]

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdev/dev_msgs.h"
#include "internal/tt-1xx/cache.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

constexpr uint32_t SEM_PUBLISHED = 1;
constexpr uint32_t SEM_DONE = 2;

inline uint32_t local_sem_addr(uint32_t id) {
    tt_l1_ptr mailboxes_t* const mb = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
    const launch_msg_t* lm = &(mb->launch[mb->launch_msg_rd_ptr]);
    return lm->kernel_config.kernel_config_base[0] + lm->kernel_config.sem_offset[0] + id * L1_ALIGNMENT;
}

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    const uint32_t min_bits = get_arg_val<uint32_t>(0);
    const uint32_t max_bits = get_arg_val<uint32_t>(1);

    volatile tt_l1_ptr uint32_t* published = (volatile tt_l1_ptr uint32_t*)local_sem_addr(SEM_PUBLISHED);
    volatile tt_l1_ptr uint32_t* done = (volatile tt_l1_ptr uint32_t*)local_sem_addr(SEM_DONE);

    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer output_tiles(cb_output_tiles);
    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);

    using namespace compute_kernel_lib;
    MaybeDeviceZoneScope("compute_chain");
    uint32_t processed = 0;
    for (;;) {
        uint32_t d, n;
        do {
            invalidate_l1_cache();
            d = *done;
            n = *published;
        } while (n == processed && d == 0);
        if (n == processed) {
            break;
        }
        for (; processed < n; ++processed) {
            eltwise_chain(
                IterationShape::tiles(block_tiles).block_size(block_tiles, BlockTailSync::FullBlock),
                CopyTile<
                    input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                    Dst::D0>{input_tiles},
                Clamp<Dst::D0>{min_bits, max_bits},
                Square<Dst::D0>{},
                PackTile<output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{output_tiles});
        }
    }
}
