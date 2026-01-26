// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

/**
 * @file reduce_helper_policies.hpp
 * @brief Policy structs for controlling reduce operation behavior
 *
 * This file contains all policy structs used by the reduce helper library:
 * - Input policies: control how input tiles are synchronized and consumed
 * - Reconfig policies: control data format reconfiguration
 */

namespace compute_kernel_lib::reduce_policies {

// =============================================================================
// Input Policy Structs - control how input tiles are synchronized and consumed
// =============================================================================

/**
 * @brief When to synchronize on input tiles
 */
enum class WaitMode {
    PER_TILE,   // wait/process/pop one tile at a time
    PER_BATCH,  // wait for batch, process all, pop batch
    UPFRONT,    // wait for everything upfront
    NONE        // caller manages synchronization
};

/**
 * @brief Streaming policy - processes tiles one at a time
 *
 * Wait: per-tile, Pop: yes
 * Safe for numerical precision, compatible with any CB size.
 * Use when tiles arrive one at a time from dataflow.
 */
struct StreamingPolicy {
    static constexpr WaitMode wait = WaitMode::PER_TILE;
    static constexpr bool pop = true;
};

/**
 * @brief Streaming batched policy - processes tiles in batches
 *
 * Wait: per-batch, Pop: yes
 * Optimal performance when tiles are pre-loaded in CB per batch/row.
 */
struct StreamingBatchedPolicy {
    static constexpr WaitMode wait = WaitMode::PER_BATCH;
    static constexpr bool pop = true;
};

/**
 * @brief Preloaded policy - caller manages synchronization
 *
 * Wait: none (caller already waited), Pop: no (caller manages)
 * Use when tiles are already in CB and caller handles wait/pop externally.
 */
struct PreloadedPolicy {
    static constexpr WaitMode wait = WaitMode::NONE;
    static constexpr bool pop = false;
};

/**
 * @brief Persistent policy - tiles remain for reuse
 *
 * Wait: upfront (all tiles), Pop: no (tiles persist)
 * Ideal for softmax pattern where tiles are reused in subsequent operations.
 */
struct PersistentPolicy {
    static constexpr WaitMode wait = WaitMode::UPFRONT;
    static constexpr bool pop = false;
};

// =============================================================================
// Reconfig Policies - control data format reconfiguration before reduce
// =============================================================================

/**
 * @brief No reconfig policy - skip all data format reconfiguration
 *
 * Use when reduce is first operation or formats already match.
 */
struct ReconfigNonePolicy {
    static constexpr bool reconfig_input = false;
    static constexpr bool reconfig_output = false;
};

/**
 * @brief Reconfig input policy - reconfigure unpacker only
 *
 * Calls reconfig_data_format(icb, icb_scaler) before reduce.
 * Use when input format changed but output format is still correct.
 */
struct ReconfigInputPolicy {
    static constexpr bool reconfig_input = true;
    static constexpr bool reconfig_output = false;
};

/**
 * @brief Reconfig output policy - reconfigure packer only
 *
 * Calls pack_reconfig_data_format(ocb) before reduce.
 * Use when output format changed but input format is still correct.
 */
struct ReconfigOutputPolicy {
    static constexpr bool reconfig_input = false;
    static constexpr bool reconfig_output = true;
};

/**
 * @brief Reconfig both policy - reconfigure both unpacker and packer (DEFAULT)
 *
 * Calls both reconfig_data_format(icb, icb_scaler) and pack_reconfig_data_format(ocb).
 * Use when both input and output formats may have changed from previous operation.
 */
struct ReconfigBothPolicy {
    static constexpr bool reconfig_input = true;
    static constexpr bool reconfig_output = true;
};

// =============================================================================
// Type Traits for Policy Validation
// =============================================================================

/**
 * @brief Type trait to detect valid input policies
 */
template <typename T>
struct is_input_policy : std::false_type {};

template <>
struct is_input_policy<StreamingPolicy> : std::true_type {};

template <>
struct is_input_policy<StreamingBatchedPolicy> : std::true_type {};

template <>
struct is_input_policy<PreloadedPolicy> : std::true_type {};

template <>
struct is_input_policy<PersistentPolicy> : std::true_type {};

template <typename T>
inline constexpr bool is_input_policy_v = is_input_policy<T>::value;

/**
 * @brief Type trait to detect valid reconfig policies
 */
template <typename T>
struct is_reconfig_policy : std::false_type {};

template <>
struct is_reconfig_policy<ReconfigNonePolicy> : std::true_type {};

template <>
struct is_reconfig_policy<ReconfigInputPolicy> : std::true_type {};

template <>
struct is_reconfig_policy<ReconfigOutputPolicy> : std::true_type {};

template <>
struct is_reconfig_policy<ReconfigBothPolicy> : std::true_type {};

template <typename T>
inline constexpr bool is_reconfig_policy_v = is_reconfig_policy<T>::value;

}  // namespace compute_kernel_lib::reduce_policies
