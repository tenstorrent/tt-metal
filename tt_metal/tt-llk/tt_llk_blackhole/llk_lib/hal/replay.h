// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel.h"

namespace hal::replay
{

/**
 * @brief Describe a contiguous sequence in the current thread's circular replay buffer.
 *
 * The 32-entry buffer wraps at its end, so a range may cross slot 31. A count of 64
 * traverses the buffer twice and is encoded as zero in the REPLAY instruction.
 */
struct BufferRange
{
    std::uint32_t start; // ISA name: Index
    std::uint32_t count; // ISA name: Count
};

/**
 * @brief Select whether recorded instructions also continue through the Tensix pipeline.
 */
enum class RecordBehavior : bool
{
    RecordOnly,
    RecordAndExecute
};

namespace detail
{
static constexpr std::uint32_t BUFFER_SIZE = 32;
static constexpr std::uint32_t MAX_COUNT   = 64;
static constexpr std::uint32_t COUNT_MASK  = MAX_COUNT - 1;

constexpr bool is_valid(const BufferRange range)
{
    return range.start < BUFFER_SIZE && range.count >= 1 && range.count <= MAX_COUNT;
}

constexpr std::uint32_t encoded_count(const std::uint32_t count)
{
    return count & COUNT_MASK;
}

constexpr std::uint32_t get_operation(const BufferRange range)
{
    return TT_OP_REPLAY(range.start, encoded_count(range.count), false, false);
}

template <BufferRange Range, RecordBehavior Behavior>
inline __attribute__((always_inline)) void begin_recording()
{
    __builtin_rvtt_ttreplay(Range.start, encoded_count(Range.count), Behavior == RecordBehavior::RecordAndExecute, true);
}

template <RecordBehavior Behavior>
inline __attribute__((always_inline)) void begin_recording(const BufferRange range)
{
    __builtin_rvtt_ttreplay(range.start, encoded_count(range.count), Behavior == RecordBehavior::RecordAndExecute, true);
}

template <typename BeginRecording, typename Callable, typename... Args>
inline __attribute__((always_inline, flatten)) void record(BeginRecording &&begin_recording, Callable &&callable, Args &&...args)
{
    // Gathering is controlled by the JIT build and is disabled by default due to tt-metal#16439.
#if defined(ENABLE_GATHERING)
    ckernel::disable_gathering();
#endif

    std::forward<BeginRecording>(begin_recording)();
    std::forward<Callable>(callable)(std::forward<Args>(args)...);

#if defined(ENABLE_GATHERING)
    ckernel::enable_gathering();
#endif
}
} // namespace detail

/**
 * @brief Record a compile-time-sized instruction sequence into this thread's replay buffer.
 *
 * @tparam Range: Circular buffer range receiving the sequence.
 * @tparam Behavior: Whether each instruction is only recorded or also executed.
 * @tparam Callable: Callable that emits exactly Range.count instructions after MOP expansion.
 * @tparam Args: Arguments forwarded to callable.
 * @param callable: Instruction-emitting callable invoked immediately after recording begins.
 * @param args: Arguments forwarded to callable.
 * @note Ensure callable produces exactly Range.count instructions after MOP expansion. Recording has no
 *       explicit terminator; a mismatch records instructions before or after the intended sequence.
 */
template <BufferRange Range, RecordBehavior Behavior = RecordBehavior::RecordOnly, typename Callable, typename... Args>
inline __attribute__((always_inline, flatten)) void record(Callable &&callable, Args &&...args)
{
    static_assert(detail::is_valid(Range), "Replay range requires start < 32 and count in [1, 64]");

    detail::record(
        [] __attribute__((always_inline)) { detail::begin_recording<Range, Behavior>(); }, std::forward<Callable>(callable), std::forward<Args>(args)...);
}

/**
 * @brief Record a runtime-positioned instruction sequence into this thread's replay buffer.
 *
 * @tparam Behavior: Whether each instruction is only recorded or also executed.
 * @tparam Callable: Callable that emits exactly range.count instructions after MOP expansion.
 * @tparam Args: Arguments forwarded to callable.
 * @param range: Circular buffer range receiving the sequence.
 * @param callable: Instruction-emitting callable invoked immediately after recording begins.
 * @param args: Arguments forwarded to callable.
 * @note Ensure callable produces exactly range.count instructions after MOP expansion. Recording has no
 *       explicit terminator; a mismatch records instructions before or after the intended sequence.
 */
template <RecordBehavior Behavior = RecordBehavior::RecordOnly, typename Callable, typename... Args>
inline __attribute__((always_inline, flatten)) void record(const BufferRange range, Callable &&callable, Args &&...args)
{
    LLK_ASSERT(detail::is_valid(range), "Replay range requires start < 32 and count in [1, 64]");

    detail::record(
        [range] __attribute__((always_inline)) { detail::begin_recording<Behavior>(range); }, std::forward<Callable>(callable), std::forward<Args>(args)...);
}

/**
 * @brief Replay a compile-time-selected buffer range on the current thread.
 *
 * @tparam Range: Circular buffer range to expand into the instruction stream.
 * @note Call @ref record for Range before this function unless the range was populated earlier.
 */
template <BufferRange Range>
inline __attribute__((always_inline)) void run()
{
    static_assert(detail::is_valid(Range), "Replay range requires start < 32 and count in [1, 64]");

    __builtin_rvtt_ttreplay(Range.start, detail::encoded_count(Range.count), false, false);
}

/**
 * @brief Replay a runtime-selected buffer range on the current thread.
 *
 * @param range: Circular buffer range to expand into the instruction stream.
 * @note Call @ref record for range before this function unless the range was populated earlier.
 */
inline __attribute__((always_inline)) void run(const BufferRange range)
{
    LLK_ASSERT(detail::is_valid(range), "Replay range requires start < 32 and count in [1, 64]");

    __builtin_rvtt_ttreplay(range.start, detail::encoded_count(range.count), false, false);
}

/**
 * @brief Encode a compile-time-selected replay operation without issuing it.
 *
 * @tparam Range: Circular buffer range the encoded operation will replay.
 * @note Use the result where another expander accepts an encoded operation, such as a MOP field.
 */
template <BufferRange Range>
constexpr std::uint32_t get_operation()
{
    static_assert(detail::is_valid(Range), "Replay range requires start < 32 and count in [1, 64]");

    return detail::get_operation(Range);
}

/**
 * @brief Encode a runtime-selected replay operation without issuing it.
 *
 * @param range: Circular buffer range the encoded operation will replay.
 * @note Use the result where another expander accepts an encoded operation, such as a MOP field.
 */
inline __attribute__((always_inline)) std::uint32_t get_operation(const BufferRange range)
{
    LLK_ASSERT(detail::is_valid(range), "Replay range requires start < 32 and count in [1, 64]");

    return detail::get_operation(range);
}

} // namespace hal::replay
