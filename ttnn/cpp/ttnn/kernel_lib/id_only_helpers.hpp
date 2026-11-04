// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Marks the id-only forms of the kernel_lib helpers (reduce, tilize, untilize, prepare_reduce_scaler,
// calculate_and_prepare_reduce_scaler, generate_index_tile, prepare_zero_tile, Accumulate::at). They build
// their own DataflowBuffers from the ids and call the form that takes the kernel's DataflowBuffers.
//
// On Quasar a DataflowBuffer drains when it goes out of scope, so a helper-owned one blocks at the end of
// the call until the other side catches up, and can deadlock (#60848). Kernels that run on Quasar pass
// their own DataflowBuffers; the id-only forms stay for the kernels that have not been migrated yet.
#ifdef ARCH_QUASAR
#define KERNEL_LIB_ID_ONLY_HELPER                                                                  \
    [[deprecated(                                                                                  \
        "On Quasar a DataflowBuffer drains when it goes out of scope (#60848): pass the kernel's " \
        "DataflowBuffer instead of a DFB id")]]
#else
#define KERNEL_LIB_ID_ONLY_HELPER
#endif
