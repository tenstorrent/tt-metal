// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/**
 * @file compute_kernel_lib_common.hpp
 * @brief Common types and constants shared by tilize/untilize helpers
 *
 * This header provides shared definitions to avoid redefinition errors
 * when both tilize_helpers.hpp and untilize_helpers.hpp are included.
 */

namespace compute_kernel_lib {

// =============================================================================
// Constants
// =============================================================================

/// Invalid CB sentinel value (matches NUM_CIRCULAR_BUFFERS)
/// Used to indicate no DT reconfiguration when passed as reconfig_from_cb
constexpr uint32_t INVALID_CB = 32;

// =============================================================================
// Enums
// =============================================================================

/**
 * @brief Controls init/uninit behavior at function boundaries
 *
 * InitAndUninit: Default - standalone operation, calls both init and uninit
 * InitOnly: First in a sequence of operations, calls only init
 * UninitOnly: Last in a sequence, calls only uninit
 * Neither: Middle of a sequence, skips both init and uninit
 */
enum class InitUninitMode : uint8_t { InitAndUninit, InitOnly, UninitOnly, Neither };

/**
 * @brief Controls whether and when the function waits for input data
 *
 * Wait: Default - calls cb_wait_front for tiles per iteration
 * WaitUpfront: Wait for all tiles before processing starts (used by untilize)
 * NoWait: No waiting - caller manages synchronization
 */
enum class WaitMode : uint8_t { Wait, WaitUpfront, NoWait };

}  // namespace compute_kernel_lib
