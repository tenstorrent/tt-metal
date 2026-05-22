// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DIAGNOSTIC: minimal writer that drains 1 tile from cb_out and exits.
// Used to bisect the unified-op hang.
// Note: factory passes cb_activated at CT arg 0 and cb_out at CT arg 1
// (matches the real writer_unified_re.cpp's layout).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    cb_wait_front(cb_out, 1);
    cb_pop_front(cb_out, 1);
}
