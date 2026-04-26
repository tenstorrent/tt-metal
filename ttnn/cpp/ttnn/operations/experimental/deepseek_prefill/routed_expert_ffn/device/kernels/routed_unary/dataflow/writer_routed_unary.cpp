// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Forked from ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/dataflow/writer_unary_ng.cpp
// Only change: NCRISC-side guard block at the top of kernel_main. See ../../guard.h.
//
// Sharded-only path: output CB (c_2) is backed by the output tensor's L1 shard; the
// writer just waits for the compute kernel to populate it — no NoC writes needed.

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"

#include "../../guard.h"

void kernel_main() {
    // NCRISC (RISCV_1): read the guard tables directly from DRAM (no mailbox
    // writes — BRISC handles the BRISC->TRISC handoff). Early-return on skip so
    // we don't block waiting for a CB push that never happens.
    if (guard_check_brisc()) {
        return;
    }

    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    constexpr auto cb_id_dst = tt::CBIndex::c_2;
    experimental::CircularBuffer cb_dst(cb_id_dst);

    cb_dst.wait_front(num_pages);
}
