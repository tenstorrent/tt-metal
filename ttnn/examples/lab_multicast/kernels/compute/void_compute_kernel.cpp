// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "compute_kernel_api.h"

// Void compute kernel - no computation is performed.
// This kernel exists as a placeholder for the compute pipeline stage.
// The user is encouraged to extend this kernel to perform operations on the received tile
// (e.g., element-wise math, column-wise or row-wise transformations, etc.) as an exercise.
void kernel_main() {
    // Nothing to compute. Print response message.
    DPRINT_MATH(DPRINT << "Void compute kernel is running." << ENDL() << ENDL());
}
