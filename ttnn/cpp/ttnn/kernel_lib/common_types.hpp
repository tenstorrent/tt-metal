// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/** Types shared by the kernel library helpers. */

namespace compute_kernel_lib {

/** Selects no accumulation; the accumulation code compiles out. */
struct NoAccumulation {};

/** Default post-op that does nothing. */
struct NoOp {
    ALWI void operator()(uint32_t = 0) const {}
};

}  // namespace compute_kernel_lib
