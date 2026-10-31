// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The dispatch's port sender (RISCV_0, NoC1): one fabric connection (its link), opened around the send phase.
// Compile-time arg 0: the manifest CB (see the reader); the phase's compile-time args from 1. Runtime args: the
// phase's, then the fabric connection's.

#include "sparse_dispatch_2d_sender.hpp"

void kernel_main() {
    constexpr uint32_t cb_man = get_compile_time_arg_val(0);
    size_t a = 0;
    const sa::SenderArgs args = sa::sender_args(a, get_write_ptr(cb_man));
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    conn.open();
    sa::send_phase<1>(args, conn);
    conn.close();
}
