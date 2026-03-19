// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    DPRINT_MATH(DPRINT << "Hanging the device, use this only for testing!!!" << ENDL());
    while (true);
}
