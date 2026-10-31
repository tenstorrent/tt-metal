// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The dispatch's port reader (RISCV_1, NoC0). Compile-time arg 0: the manifest CB (this port core's input rings, at
// the same address on every port core of every chip: the program's first CB); the phase's compile-time args from 1.

#include "sparse_dispatch_2d_reader.hpp"

void kernel_main() {
    constexpr uint32_t cb_man = get_compile_time_arg_val(0);
    size_t a = 0;
    const sa::ReaderArgs args = sa::reader_args(a, get_write_ptr(cb_man));
    (void)sa::read_phase<1>(args);
}
