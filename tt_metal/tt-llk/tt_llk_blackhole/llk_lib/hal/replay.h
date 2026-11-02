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
 * @brief Describe a contiguous sequence in the current thread's replay buffer.
 *
 * A range must not cross slot 31; the ISA leaves wrapping around the 32-entry buffer and a
 * zero-encoded count of 64 non-contractual.
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

constexpr bool is_valid(const BufferRange range)
{
    return range.start < BUFFER_SIZE && range.count >= 1 && range.count <= BUFFER_SIZE - range.start;
}

constexpr lltt::ExecBool exec_while_recording(const RecordBehavior behavior)
{
    return behavior == RecordBehavior::RecordAndExecute ? lltt::Exec : lltt::NoExec;
}

constexpr std::uint32_t get_operation(const BufferRange range)
{
    return TT_OP_REPLAY(range.start, range.count, false, false);
}

template <BufferRange Range, RecordBehavior Behavior>
inline __attribute__((always_inline)) void begin_recording()
{
    lltt::record<exec_while_recording(Behavior)>(Range.start, Range.count);
}

template <RecordBehavior Behavior>
inline __attribute__((always_inline)) void begin_recording(BufferRange range)
{
    // With LLK_ASSERT enabled, jump threading can specialize the range check's failure path for
    // count == 0, and the recording intrinsic rejects that constant at compile time. Hiding a
    // runtime count from the optimizer prevents it; a constant count keeps the immediate REPLAY.
    if (!__builtin_constant_p(range.count))
    {
        asm("" : "+r"(range.count));
    }
    lltt::record<exec_while_recording(Behavior)>(range.start, range.count);
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
 * @tparam Range: Buffer range receiving the sequence.
 * @tparam Behavior: Whether each instruction is only recorded or also executed.
 * @tparam Callable: Callable that emits exactly Range.count instructions after MOP expansion.
 * @tparam Args: Arguments forwarded to callable.
 * @param callable: Instruction-emitting callable invoked immediately after recording begins.
 * @param args: Arguments forwarded to callable.
 * @note Ensure callable produces exactly Range.count instructions after MOP expansion. Recording has no
 *       explicit terminator; a mismatch records instructions before or after the intended sequence.
 * @note Ensure the recorded body never contains a REPLAY; a REPLAY cannot be one of the
 *       instructions it loads. In RecordOnly also ensure it contains no MOP or MOP_CFG;
 *       a load-only REPLAY may record only individual single-store pushes.
 */
template <BufferRange Range, RecordBehavior Behavior = RecordBehavior::RecordOnly, typename Callable, typename... Args>
inline __attribute__((always_inline, flatten)) void record(Callable &&callable, Args &&...args)
{
    static_assert(detail::is_valid(Range), "Replay range requires count >= 1 and start + count <= 32");

    detail::record(
        [] __attribute__((always_inline)) { detail::begin_recording<Range, Behavior>(); }, std::forward<Callable>(callable), std::forward<Args>(args)...);
}

/**
 * @brief Record a runtime-positioned instruction sequence into this thread's replay buffer.
 *
 * @tparam Behavior: Whether each instruction is only recorded or also executed.
 * @tparam Callable: Callable that emits exactly range.count instructions after MOP expansion.
 * @tparam Args: Arguments forwarded to callable.
 * @param range: Buffer range receiving the sequence.
 * @param callable: Instruction-emitting callable invoked immediately after recording begins.
 * @param args: Arguments forwarded to callable.
 * @note Ensure callable produces exactly range.count instructions after MOP expansion. Recording has no
 *       explicit terminator; a mismatch records instructions before or after the intended sequence.
 * @note Ensure the recorded body never contains a REPLAY; a REPLAY cannot be one of the
 *       instructions it loads. In RecordOnly also ensure it contains no MOP or MOP_CFG;
 *       a load-only REPLAY may record only individual single-store pushes.
 * @note Ensure range.start folds to a compile-time constant; the SFPI REPLAY builtin rejects a
 *       runtime start index. Only range.count may be dynamic.
 */
template <RecordBehavior Behavior = RecordBehavior::RecordOnly, typename Callable, typename... Args>
inline __attribute__((always_inline, flatten)) void record(const BufferRange range, Callable &&callable, Args &&...args)
{
    LLK_ASSERT(detail::is_valid(range), "Replay range requires count >= 1 and start + count <= 32");

    detail::record(
        [range] __attribute__((always_inline)) { detail::begin_recording<Behavior>(range); }, std::forward<Callable>(callable), std::forward<Args>(args)...);
}

/**
 * @brief Replay a compile-time-selected buffer range on the current thread.
 *
 * @tparam Range: Buffer range to expand into the instruction stream.
 * @note Call @ref record for Range before this function unless the range was populated earlier.
 */
template <BufferRange Range>
inline __attribute__((always_inline)) void run()
{
    static_assert(detail::is_valid(Range), "Replay range requires count >= 1 and start + count <= 32");

    lltt::replay(Range.start, Range.count);
}

/**
 * @brief Replay a runtime-selected buffer range on the current thread.
 *
 * @param range: Buffer range to expand into the instruction stream.
 * @note Call @ref record for range before this function unless the range was populated earlier.
 * @note Ensure range.start folds to a compile-time constant; the SFPI REPLAY builtin rejects a
 *       runtime start index. Only range.count may be dynamic.
 */
inline __attribute__((always_inline)) void run(const BufferRange range)
{
    LLK_ASSERT(detail::is_valid(range), "Replay range requires count >= 1 and start + count <= 32");

    lltt::replay(range.start, range.count);
}

/**
 * @brief Encode a compile-time-selected replay operation without issuing it.
 *
 * @tparam Range: Buffer range the encoded operation will replay.
 * @note Use the result where another expander accepts an encoded operation, such as a MOP field.
 */
template <BufferRange Range>
constexpr std::uint32_t get_operation()
{
    static_assert(detail::is_valid(Range), "Replay range requires count >= 1 and start + count <= 32");

    return detail::get_operation(Range);
}

/**
 * @brief Encode a runtime-selected replay operation without issuing it.
 *
 * @param range: Buffer range the encoded operation will replay.
 * @note Use the result where another expander accepts an encoded operation, such as a MOP field.
 */
inline __attribute__((always_inline)) std::uint32_t get_operation(const BufferRange range)
{
    LLK_ASSERT(detail::is_valid(range), "Replay range requires count >= 1 and start + count <= 32");

    return detail::get_operation(range);
}

} // namespace hal::replay
