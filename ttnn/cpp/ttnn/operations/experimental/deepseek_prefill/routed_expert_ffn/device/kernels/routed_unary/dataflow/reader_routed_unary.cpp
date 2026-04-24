// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Forked from ttnn/cpp/ttnn/operations/eltwise/unary_ng/device/kernels/dataflow/reader_unary_ng.cpp
// Only change: BRISC-side guard block at the top of kernel_main. See ../../guard.h.
//
// Sharded-only path: input CB (c_0) is backed by the input tensor's L1 shard; the
// reader just reserves/pushes the CB to signal the compute kernel that the shard
// contents are ready (which they always are — the tensor is already on device in L1).

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"

#include "../../guard.h"

void kernel_main() {
    // BRISC (RISCV_0): publish the skip decision to the three TRISC mailboxes, and
    // early-return on skip so we don't stall the compute kernel with a CB push it
    // no longer needs.
    if (guard_check_wait()) {
        return;
    }

    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    constexpr auto cb_id_src = tt::CBIndex::c_0;
    experimental::CircularBuffer cb_src(cb_id_src);

    cb_src.reserve_back(num_pages);
    cb_src.push_back(num_pages);
}
