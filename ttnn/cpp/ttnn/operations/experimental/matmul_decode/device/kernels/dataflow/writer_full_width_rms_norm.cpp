// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_width_rms_norm_transport.hpp"

void kernel_main() {
    // Runtime arg 0 stores the grouped-table offset; the currently active topology begins at 1.
    run_full_width_rms_norm_transport(1);
}
