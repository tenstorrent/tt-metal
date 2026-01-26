// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

/**
 * @file cb_policies.hpp
 * @brief Policy types for circular buffer management in compute helpers
 *
 * This header provides type-based policies that control HOW and WHEN circular buffer
 * operations (wait/pop/reserve/push) are performed in compute kernel helpers.
 *
 * Key design principle: Separation of concerns
 * - Policy = timing/pattern (when to wait/pop: per-tile, per-chunk, upfront, etc.)
 * - Tile count = derived internally from broadcast dimension
 *
 * Example: Streaming policy with BroadcastDim::SCALAR will:
 * - Wait for 1 B tile (derived from SCALAR broadcast)
 * - Pop 1 B tile at the end (not per-iteration)
 * The policy controls WHEN, the broadcast dim controls HOW MANY.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/cb_policies.hpp"
 *
 *   using namespace cb_policies;
 *
 *   // Using predefined combinations (equivalent to old BinaryInputMode)
 *   binary_op<BinaryOpType::ADD, BroadcastDim::NONE, Streaming, Streaming>(...)
 *
 *   // Custom combinations for advanced use cases
 *   using MyPolicy = InputPolicy<WaitUpfront, PopAtEnd>;
 *   binary_op<BinaryOpType::MUL, BroadcastDim::ROW, Streaming, MyPolicy>(...)
 */

namespace cb_policies {

// =============================================================================
// Wait Policies - WHEN to call cb_wait_front
// =============================================================================

/// Wait for 1 tile at a time, inside the processing loop
struct WaitPerTile {
    static constexpr bool per_tile = true;
    static constexpr bool per_chunk = false;
    static constexpr bool upfront = false;
    static constexpr bool caller_managed = false;
};

/// Wait for chunk of tiles at a time (DEST_LIMIT tiles)
struct WaitPerChunk {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = true;
    static constexpr bool upfront = false;
    static constexpr bool caller_managed = false;
};

/// Wait for all tiles once at start of operation
struct WaitUpfront {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool upfront = true;
    static constexpr bool caller_managed = false;
};

/// Caller is responsible for cb_wait_front before calling
struct WaitCallerManaged {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool upfront = false;
    static constexpr bool caller_managed = true;
};

// =============================================================================
// Pop Policies - WHEN to call cb_pop_front
// =============================================================================

/// Pop 1 tile immediately after processing
struct PopPerTile {
    static constexpr bool per_tile = true;
    static constexpr bool per_chunk = false;
    static constexpr bool at_end = false;
    static constexpr bool never = false;
    static constexpr bool caller_managed = false;
};

/// Pop chunk of tiles after processing chunk
struct PopPerChunk {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = true;
    static constexpr bool at_end = false;
    static constexpr bool never = false;
    static constexpr bool caller_managed = false;
};

/// Pop all tiles at end of operation
struct PopAtEnd {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool at_end = true;
    static constexpr bool never = false;
    static constexpr bool caller_managed = false;
};

/// Never pop - tiles persist for subsequent operations
struct PopNever {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool at_end = false;
    static constexpr bool never = true;
    static constexpr bool caller_managed = false;
};

/// Caller is responsible for cb_pop_front after operation
struct PopCallerManaged {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool at_end = false;
    static constexpr bool never = false;
    static constexpr bool caller_managed = true;
};

// =============================================================================
// Output Policies - WHEN to call cb_reserve_back/cb_push_back
// =============================================================================

/// Reserve/push 1 tile at a time (streaming)
struct OutputPerTile {
    static constexpr bool per_tile = true;
    static constexpr bool per_chunk = false;
    static constexpr bool bulk = false;
};

/// Reserve/push chunk of tiles at a time (DEST_LIMIT tiles)
struct OutputPerChunk {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = true;
    static constexpr bool bulk = false;
};

/// Reserve all upfront, push all at end
struct OutputBulk {
    static constexpr bool per_tile = false;
    static constexpr bool per_chunk = false;
    static constexpr bool bulk = true;
};

// =============================================================================
// Combined Input Policy
// =============================================================================

/// Combined input policy: how to handle an input CB
template <typename WaitPolicy, typename PopPolicy>
struct InputPolicy {
    using wait = WaitPolicy;
    using pop = PopPolicy;

    // Convenience accessors
    static constexpr bool waits_per_tile = WaitPolicy::per_tile;
    static constexpr bool waits_per_chunk = WaitPolicy::per_chunk;
    static constexpr bool waits_upfront = WaitPolicy::upfront;
    static constexpr bool waits_caller_managed = WaitPolicy::caller_managed;

    static constexpr bool pops_per_tile = PopPolicy::per_tile;
    static constexpr bool pops_per_chunk = PopPolicy::per_chunk;
    static constexpr bool pops_at_end = PopPolicy::at_end;
    static constexpr bool pops_never = PopPolicy::never;
    static constexpr bool pops_caller_managed = PopPolicy::caller_managed;
};

// =============================================================================
// Predefined Combinations (map to old BinaryInputMode equivalents)
// =============================================================================

/// STREAMING equivalent: Wait/pop 1 tile at a time (DEFAULT - simplest, most broadly applicable)
using Streaming = InputPolicy<WaitPerTile, PopPerTile>;

/// STREAMING_BATCHED equivalent: Wait/pop chunks of DEST_LIMIT tiles
using StreamingBatched = InputPolicy<WaitPerChunk, PopPerChunk>;

/// PRELOADED equivalent: Caller manages all wait/pop
using Preloaded = InputPolicy<WaitCallerManaged, PopCallerManaged>;

/// PERSISTENT equivalent: Wait all upfront, never pop
using Persistent = InputPolicy<WaitUpfront, PopNever>;

// =============================================================================
// Type Traits for Policy Detection
// =============================================================================

/// Check if a policy is the Streaming policy
template <typename Policy>
struct is_streaming_policy : std::false_type {};

template <>
struct is_streaming_policy<Streaming> : std::true_type {};

template <typename Policy>
inline constexpr bool is_streaming_policy_v = is_streaming_policy<Policy>::value;

/// Check if a policy is the StreamingBatched policy
template <typename Policy>
struct is_streaming_batched_policy : std::false_type {};

template <>
struct is_streaming_batched_policy<StreamingBatched> : std::true_type {};

template <typename Policy>
inline constexpr bool is_streaming_batched_policy_v = is_streaming_batched_policy<Policy>::value;

/// Check if a policy is the Preloaded policy
template <typename Policy>
struct is_preloaded_policy : std::false_type {};

template <>
struct is_preloaded_policy<Preloaded> : std::true_type {};

template <typename Policy>
inline constexpr bool is_preloaded_policy_v = is_preloaded_policy<Policy>::value;

/// Check if a policy is the Persistent policy
template <typename Policy>
struct is_persistent_policy : std::false_type {};

template <>
struct is_persistent_policy<Persistent> : std::true_type {};

template <typename Policy>
inline constexpr bool is_persistent_policy_v = is_persistent_policy<Policy>::value;

// =============================================================================
// Input Policy Type Traits
// =============================================================================

/// Check if a type is a valid InputPolicy
template <typename T>
struct is_input_policy : std::false_type {};

template <typename WaitPolicy, typename PopPolicy>
struct is_input_policy<InputPolicy<WaitPolicy, PopPolicy>> : std::true_type {};

template <typename T>
inline constexpr bool is_input_policy_v = is_input_policy<T>::value;

/// Check if a type is a valid OutputPolicy
template <typename T>
struct is_output_policy : std::false_type {};

template <>
struct is_output_policy<OutputPerTile> : std::true_type {};

template <>
struct is_output_policy<OutputPerChunk> : std::true_type {};

template <>
struct is_output_policy<OutputBulk> : std::true_type {};

template <typename T>
inline constexpr bool is_output_policy_v = is_output_policy<T>::value;

}  // namespace cb_policies
