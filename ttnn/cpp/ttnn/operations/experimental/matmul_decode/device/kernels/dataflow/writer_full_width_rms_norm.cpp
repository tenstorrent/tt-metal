// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_width_rms_norm_transport.hpp"

void kernel_main() {
    const uint32_t grouped_metadata_arg_base = get_arg_val<uint32_t>(0);
    run_full_width_rms_norm_transport(grouped_metadata_arg_base);
}
