// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "experimental/circular_buffer.h"
#include "internal/mod_div_lib.h"
#include "api/debug/dprint.h"

#include "sanitizer/types.h"
#include "sanitizer/impl.h"

TT_ALWAYS_INLINE void test(
    ct_string message,
    const llk::san::UnwindContext update,
    const llk::san::UnwindContext current)
{

    // const ct_string kernel         = CTSTR(KERNEL_NAME);
    // const ct_string update_header  = CTSTR("Last relevant operand state update");
    // const ct_string current_header = CTSTR("Current operand state");
    // const ct_string unknown        = CTSTR("<unknown>");
    // const ct_string file           = CTSTR("<file>");
    // const ct_string line           = CTSTR("<line>");

    // const bool update_known  = update.pc != UINTPTR_MAX;
    // const bool current_known = current.pc != UINTPTR_MAX;

    DEVICE_PRINT("DJUBRE!!!\n");

    // DEVICE_PRINT(
    //     "┌─[ llk::san ]─[ error ]──────\n"
    //     "│  {}\n"
    //     "│\n"
    //     "│\n"
    //     "│  ┌[ Current Kernel ]─\n"
    //     "│  └── {}\n"
    //     "│\n"
    //     "│  ┌[ {} ]─\n"
    //     "│  ├── Compute API ─┬ {:#x}\n"
    //     "│  │                └ {}:{}\n"
    //     "│  └── Callsite ────┬ {:#x}\n"
    //     "│                   └ {}:{}\n"
    //     "│\n"
    //     "│  ┌[ {} ]─\n"
    //     "│  ├── Compute API ─┬ {:#x}\n"
    //     "│  │                └ {}:{}\n"
    //     "│  └── Callsite ────┬ {:#x}\n"
    //     "│                   └ {}:{}\n"
    //     "└─────────────────────────────",
    //     message,
    //     kernel,
    //     update_header,
    //     update.pc,
    //     update_known ? file : unknown,
    //     update_known ? line : unknown,
    //     update.ra,
    //     update_known ? file : unknown,
    //     update_known ? line : unknown,
    //     current_header,
    //     current.pc,
    //     current_known ? file : unknown,
    //     current_known ? line : unknown,
    //     current.ra,
    //     current_known ? file : unknown,
    //     current_known ? line : unknown);

}


void kernel_main() {
    ckernel::fence_compiler();
    DEVICE_PRINT("1\n");


    test(
        CTSTR("crkni"),
        llk::san::sanitizer->context.pack.configure_pack,
        llk::san::sanitizer->context.pack.current);


    DEVICE_PRINT("2\n");
    ckernel::fence_compiler();
}
