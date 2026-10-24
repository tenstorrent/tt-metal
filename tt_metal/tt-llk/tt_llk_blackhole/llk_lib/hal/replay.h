// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel.h"
#include "llk_assert.h"
#include "lltt.h"

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
inline constexpr std::uint32_t BUFFER_SIZE = 32;
inline constexpr std::uint32_t MAX_COUNT   = 64;
inline constexpr std::uint32_t COUNT_MASK  = MAX_COUNT - 1;

constexpr bool is_valid(const BufferRange range)
{
    return range.start < BUFFER_SIZE && range.count >= 1 && range.count <= MAX_COUNT;
}

constexpr std::uint32_t encoded_count(const std::uint32_t count)
{
    return count & COUNT_MASK;
}

constexpr lltt::ExecBool exec_while_recording(const RecordBehavior behavior)
{
    return behavior == RecordBehavior::RecordAndExecute ? lltt::Exec : lltt::NoExec;
}

constexpr std::uint32_t get_operation(const BufferRange range)
{
    return TT_OP_REPLAY(range.start, encoded_count(range.count), false, false);
}

template <BufferRange Range, RecordBehavior Behavior>
inline __attribute__((always_inline)) void begin_recording()
{
    lltt::record<exec_while_recording(Behavior)>(Range.start, encoded_count(Range.count));
}

template <RecordBehavior Behavior>
inline __attribute__((always_inline)) void begin_recording(BufferRange range)
{
#ifdef ENABLE_LLK_ASSERT
    // The diagnostic range check ahead of this call leaves a specialized code path on which the
    // count is a provably invalid constant, which the recording intrinsic rejects at compile
    // time. Keeping a runtime value opaque removes that path; the register constraint adds no
    // code. A constant count skips it so the REPLAY stays an immediate instruction.
    if (!__builtin_constant_p(range.count))
    {
        asm("" : "+r"(range.count));
    }
#endif
    lltt::record<exec_while_recording(Behavior)>(range.start, encoded_count(range.count));
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
 * @note Keep Range.count at or below 32; the SFPI REPLAY builtin rejects larger immediate counts.
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
 * @note Ensure range.start folds to a compile-time constant; the SFPI REPLAY builtin rejects a
 *       runtime start index. Only range.count may be dynamic.
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
 * @note Keep Range.count at or below 32; the SFPI REPLAY builtin rejects larger immediate counts.
 */
template <BufferRange Range>
inline __attribute__((always_inline)) void run()
{
    static_assert(detail::is_valid(Range), "Replay range requires start < 32 and count in [1, 64]");

    lltt::replay(Range.start, detail::encoded_count(Range.count));
}

/**
 * @brief Replay a runtime-selected buffer range on the current thread.
 *
 * @param range: Circular buffer range to expand into the instruction stream.
 * @note Call @ref record for range before this function unless the range was populated earlier.
 * @note Ensure range.start folds to a compile-time constant; the SFPI REPLAY builtin rejects a
 *       runtime start index. Only range.count may be dynamic.
 */
inline __attribute__((always_inline)) void run(const BufferRange range)
{
    LLK_ASSERT(detail::is_valid(range), "Replay range requires start < 32 and count in [1, 64]");

    lltt::replay(range.start, detail::encoded_count(range.count));
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
