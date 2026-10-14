// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Constants shared by the generic reduce compute kernels. These are op-level policy — which
// scalars the kernel skips — rather than reduce primitives, which live in ttnn/cpp/ttnn/kernel_lib.

/**
 * @brief Bit pattern of 1.0f, the reduction scalar that is a no-op.
 *
 * A post-multiply by this value must be skipped.
 */
inline constexpr uint32_t k_identity_scaler_bits = 0x3F800000u;
