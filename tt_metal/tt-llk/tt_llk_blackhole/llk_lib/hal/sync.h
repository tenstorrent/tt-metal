// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"
#include "utils/helpers.h"

namespace hal::sync
{

/** @brief Select the RISC MMIO or Tensix instruction path for a semaphore operation. */
enum class Access : std::uint8_t
{
    MMIO,
    Tensix
};

/**
 * @brief Select one of the four Blackhole Tensix mutexes.
 *
 * The M0/M2/M3/M4 names are physical indices.
 */
enum class Mutex : std::uint8_t
{
    M0 = 0,
    M2 = 2,
    M3 = 3,
    M4 = 4
};

/** @brief Select one of the eight physical Tensix semaphores by index. */
enum class Semaphore : std::uint8_t
{
    S0 = 0,
    S1 = 1,
    S2 = 2,
    S3 = 3,
    S4 = 4,
    S5 = 5,
    S6 = 6,
    S7 = 7
};

/** @brief Select one or more Tensix semaphores atomically. */
enum class SemaphoreMask : std::uint8_t
{
    None = 0,
    S0   = 1u << 0,
    S1   = 1u << 1,
    S2   = 1u << 2,
    S3   = 1u << 3,
    S4   = 1u << 4,
    S5   = 1u << 5,
    S6   = 1u << 6,
    S7   = 1u << 7,
    All  = 0xffu
};

/** @brief Select instruction classes blocked by a wait gate. */
enum class StallTarget : std::uint16_t
{
    HardwareDefault = 0,
    Compute         = 1u << 0,
    Tdma            = Compute,
    Sync            = 1u << 1,
    Pack            = 1u << 2,
    Unpack          = 1u << 3,
    Mover           = 1u << 4,
    Xmov            = Mover,
    Scalar          = 1u << 5,
    Thcon           = Scalar,
    Math            = 1u << 6,
    Config          = 1u << 7,
    Cfg             = Config,
    Sfpu            = 1u << 8,
    All             = 0x1ffu
};

/** @brief Select Blackhole STALLWAIT completion conditions. */
enum class StallCondition : std::uint16_t
{
    HardwareDefault      = 0,
    ScalarIdle           = 1u << 0,
    Unpacker0Idle        = 1u << 1,
    Unpacker1Idle        = 1u << 2,
    PackerIdle           = 1u << 3,
    MathIdle             = 1u << 4,
    SrcACleared          = 1u << 5,
    SrcBCleared          = 1u << 6,
    SrcAValid            = 1u << 7,
    SrcBValid            = 1u << 8,
    MoverIdle            = 1u << 9,
    RiscvAccessProcessed = 1u << 10,
    SfpuIdle             = 1u << 11,
    ConfigUnitIdle       = 1u << 12,
    All                  = 0x1fffu
};

/** @brief Select the SEMWAIT predicates that keep the wait active. */
enum class SemaphoreCondition : std::uint8_t
{
    WhileZero    = 1u << 0,
    WhileMaximum = 1u << 1
};

inline constexpr SemaphoreMask operator|(const SemaphoreMask lhs, const SemaphoreMask rhs)
{
    return static_cast<SemaphoreMask>(static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs));
}

inline constexpr SemaphoreMask operator&(const SemaphoreMask lhs, const SemaphoreMask rhs)
{
    return static_cast<SemaphoreMask>(static_cast<std::uint8_t>(lhs) & static_cast<std::uint8_t>(rhs));
}

inline constexpr StallTarget operator|(const StallTarget lhs, const StallTarget rhs)
{
    return static_cast<StallTarget>(static_cast<std::uint16_t>(lhs) | static_cast<std::uint16_t>(rhs));
}

inline constexpr StallTarget operator&(const StallTarget lhs, const StallTarget rhs)
{
    return static_cast<StallTarget>(static_cast<std::uint16_t>(lhs) & static_cast<std::uint16_t>(rhs));
}

inline constexpr StallCondition operator|(const StallCondition lhs, const StallCondition rhs)
{
    return static_cast<StallCondition>(static_cast<std::uint16_t>(lhs) | static_cast<std::uint16_t>(rhs));
}

inline constexpr StallCondition operator&(const StallCondition lhs, const StallCondition rhs)
{
    return static_cast<StallCondition>(static_cast<std::uint16_t>(lhs) & static_cast<std::uint16_t>(rhs));
}

inline constexpr SemaphoreCondition operator|(const SemaphoreCondition lhs, const SemaphoreCondition rhs)
{
    return static_cast<SemaphoreCondition>(static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs));
}

inline constexpr SemaphoreCondition operator&(const SemaphoreCondition lhs, const SemaphoreCondition rhs)
{
    return static_cast<SemaphoreCondition>(static_cast<std::uint8_t>(lhs) & static_cast<std::uint8_t>(rhs));
}

namespace detail
{
inline __attribute__((always_inline)) void assert_operand(const bool valid, [[maybe_unused]] const char* message)
{
    LLK_ASSERT(valid, message);
}

constexpr void require_valid_operand(const bool valid, const char* message)
{
    if (__builtin_is_constant_evaluated())
    {
        if (!valid)
        {
            __builtin_trap();
        }
    }
    else
    {
        assert_operand(valid, message);
    }
}

constexpr bool is_valid(const Access access)
{
    return access == Access::MMIO || access == Access::Tensix;
}

constexpr bool is_valid(const Mutex mutex)
{
    return mutex == Mutex::M0 || mutex == Mutex::M2 || mutex == Mutex::M3 || mutex == Mutex::M4;
}

constexpr bool is_valid(const Semaphore semaphore)
{
    return hal::to_underlying(semaphore) < 8u;
}

constexpr bool is_valid(const SemaphoreMask mask)
{
    return hal::to_underlying(mask) != 0u;
}

constexpr bool is_valid(const StallTarget targets)
{
    return hal::to_underlying(targets) <= hal::to_underlying(StallTarget::All);
}

constexpr bool is_valid(const StallCondition conditions)
{
    return hal::to_underlying(conditions) <= hal::to_underlying(StallCondition::All);
}

constexpr bool is_valid(const SemaphoreCondition conditions)
{
    const std::uint32_t encoded = hal::to_underlying(conditions);
    return encoded != 0u && encoded <= 0x3u;
}

constexpr std::uint32_t semaphore_bit(const Semaphore semaphore)
{
    return 1u << hal::to_underlying(semaphore);
}

} // namespace detail

namespace mutex
{

/** @brief Encode ATGETM without issuing it. */
template <Mutex M>
inline constexpr std::uint32_t acquire_operation()
{
    static_assert(detail::is_valid(M), "Blackhole mutex index must be 0 or in [2, 4]");
    return TT_OP_ATGETM(hal::to_underlying(M));
}

/** @brief Encode a runtime-selected ATGETM without issuing it. */
inline constexpr __attribute__((always_inline)) std::uint32_t acquire_operation(const Mutex mutex)
{
    detail::require_valid_operand(detail::is_valid(mutex), "Blackhole mutex index must be 0 or in [2, 4]");
    return TT_OP_ATGETM(hal::to_underlying(mutex));
}

/** @brief Acquire a compile-time-selected mutex with one immediate ATGETM. */
template <Mutex M>
inline __attribute__((always_inline)) void acquire()
{
    (void)acquire_operation<M>();
    TTI_ATGETM(hal::to_underlying(M));
}

/** @brief Acquire a runtime-selected mutex through the Tensix instruction buffer. */
inline __attribute__((always_inline)) void acquire(const Mutex mutex)
{
    LLK_ASSERT(detail::is_valid(mutex), "Blackhole mutex index must be 0 or in [2, 4]");
    TT_ATGETM(hal::to_underlying(mutex));
}

/** @brief Encode ATRELM without issuing it. */
template <Mutex M>
inline constexpr std::uint32_t release_operation()
{
    static_assert(detail::is_valid(M), "Blackhole mutex index must be 0 or in [2, 4]");
    return TT_OP_ATRELM(hal::to_underlying(M));
}

/** @brief Encode a runtime-selected ATRELM without issuing it. */
inline constexpr __attribute__((always_inline)) std::uint32_t release_operation(const Mutex mutex)
{
    detail::require_valid_operand(detail::is_valid(mutex), "Blackhole mutex index must be 0 or in [2, 4]");
    return TT_OP_ATRELM(hal::to_underlying(mutex));
}

/** @brief Release a compile-time-selected mutex with one immediate ATRELM. */
template <Mutex M>
inline __attribute__((always_inline)) void release()
{
    (void)release_operation<M>();
    TTI_ATRELM(hal::to_underlying(M));
}

/** @brief Release a runtime-selected mutex through the Tensix instruction buffer. */
inline __attribute__((always_inline)) void release(const Mutex mutex)
{
    LLK_ASSERT(detail::is_valid(mutex), "Blackhole mutex index must be 0 or in [2, 4]");
    TT_ATRELM(hal::to_underlying(mutex));
}

} // namespace mutex

namespace semaphore
{

/** @brief Encode SEMINIT without issuing it. */
template <SemaphoreMask Mask, std::uint32_t Initial, std::uint32_t Maximum>
inline constexpr std::uint32_t init_operation()
{
    static_assert(detail::is_valid(Mask), "SEMINIT requires at least one semaphore");
    static_assert(Initial < 16u, "SEMINIT initial value must fit in four bits");
    static_assert(Maximum < 16u, "SEMINIT maximum value must fit in four bits");
    return TT_OP_SEMINIT(Maximum, Initial, hal::to_underlying(Mask));
}

/** @brief Encode a runtime-selected SEMINIT without issuing it. */
inline constexpr __attribute__((always_inline)) std::uint32_t init_operation(const SemaphoreMask mask, const std::uint32_t initial, const std::uint32_t maximum)
{
    detail::require_valid_operand(detail::is_valid(mask), "SEMINIT requires at least one semaphore");
    detail::require_valid_operand(initial < 16u, "SEMINIT initial value must fit in four bits");
    detail::require_valid_operand(maximum < 16u, "SEMINIT maximum value must fit in four bits");
    return TT_OP_SEMINIT(maximum, initial, hal::to_underlying(mask));
}

/**
 * @brief Initialize compile-time-selected semaphores through Tensix.
 *
 * MMIO has no equivalent: only SEMINIT can assign both Value and Max.
 */
template <SemaphoreMask Mask, std::uint32_t Initial, std::uint32_t Maximum>
inline __attribute__((always_inline)) void init()
{
    (void)init_operation<Mask, Initial, Maximum>();
    TTI_SEMINIT(Maximum, Initial, hal::to_underlying(Mask));
}

/** @brief Initialize runtime-selected semaphores through the Tensix instruction buffer. */
inline __attribute__((always_inline)) void init(const SemaphoreMask mask, const std::uint32_t initial, const std::uint32_t maximum)
{
    LLK_ASSERT(detail::is_valid(mask), "SEMINIT requires at least one semaphore");
    LLK_ASSERT(initial < 16u, "SEMINIT initial value must fit in four bits");
    LLK_ASSERT(maximum < 16u, "SEMINIT maximum value must fit in four bits");
    TT_SEMINIT(maximum, initial, hal::to_underlying(mask));
}

/** @brief Encode SEMPOST without issuing it. */
template <SemaphoreMask Mask>
inline constexpr std::uint32_t post_operation()
{
    static_assert(detail::is_valid(Mask), "SEMPOST requires at least one semaphore");
    return TT_OP_SEMPOST(hal::to_underlying(Mask));
}

/** @brief Encode a runtime-selected SEMPOST without issuing it. */
inline constexpr __attribute__((always_inline)) std::uint32_t post_operation(const SemaphoreMask mask)
{
    detail::require_valid_operand(detail::is_valid(mask), "SEMPOST requires at least one semaphore");
    return TT_OP_SEMPOST(hal::to_underlying(mask));
}

/** @brief Increment compile-time-selected semaphores through Tensix. */
template <SemaphoreMask Mask>
inline __attribute__((always_inline)) void post()
{
    (void)post_operation<Mask>();
    TTI_SEMPOST(hal::to_underlying(Mask));
}

/** @brief Increment runtime-selected semaphores through Tensix. */
inline __attribute__((always_inline)) void post(const SemaphoreMask mask)
{
    LLK_ASSERT(detail::is_valid(mask), "SEMPOST requires at least one semaphore");
    TT_SEMPOST(hal::to_underlying(mask));
}

/** @brief Increment one compile-time-selected semaphore through the chosen access path. */
template <Access A, Semaphore S>
inline __attribute__((always_inline)) void post()
{
    static_assert(detail::is_valid(A), "Semaphore access must be MMIO or Tensix");
    static_assert(detail::is_valid(S), "Semaphore index must be in [0, 7]");
    if constexpr (A == Access::MMIO)
    {
        ckernel::semaphore_post(hal::to_underlying(S));
    }
    else
    {
        TTI_SEMPOST(detail::semaphore_bit(S));
    }
}

/** @brief Increment one runtime-selected semaphore through the chosen access path. */
template <Access A>
inline __attribute__((always_inline)) void post(const Semaphore semaphore)
{
    static_assert(detail::is_valid(A), "Semaphore access must be MMIO or Tensix");
    LLK_ASSERT(detail::is_valid(semaphore), "Semaphore index must be in [0, 7]");
    if constexpr (A == Access::MMIO)
    {
        ckernel::semaphore_post(hal::to_underlying(semaphore));
    }
    else
    {
        TT_SEMPOST(detail::semaphore_bit(semaphore));
    }
}

/** @brief Encode SEMGET (atomic decrement) without issuing it. */
template <SemaphoreMask Mask>
inline constexpr std::uint32_t get_operation()
{
    static_assert(detail::is_valid(Mask), "SEMGET requires at least one semaphore");
    return TT_OP_SEMGET(hal::to_underlying(Mask));
}

/** @brief Encode a runtime-selected SEMGET without issuing it. */
inline constexpr __attribute__((always_inline)) std::uint32_t get_operation(const SemaphoreMask mask)
{
    detail::require_valid_operand(detail::is_valid(mask), "SEMGET requires at least one semaphore");
    return TT_OP_SEMGET(hal::to_underlying(mask));
}

/** @brief Decrement compile-time-selected semaphores through Tensix. */
template <SemaphoreMask Mask>
inline __attribute__((always_inline)) void get()
{
    (void)get_operation<Mask>();
    TTI_SEMGET(hal::to_underlying(Mask));
}

/** @brief Decrement runtime-selected semaphores through Tensix. */
inline __attribute__((always_inline)) void get(const SemaphoreMask mask)
{
    LLK_ASSERT(detail::is_valid(mask), "SEMGET requires at least one semaphore");
    TT_SEMGET(hal::to_underlying(mask));
}

/** @brief Decrement one compile-time-selected semaphore through the chosen access path. */
template <Access A, Semaphore S>
inline __attribute__((always_inline)) void get()
{
    static_assert(detail::is_valid(A), "Semaphore access must be MMIO or Tensix");
    static_assert(detail::is_valid(S), "Semaphore index must be in [0, 7]");
    if constexpr (A == Access::MMIO)
    {
        ckernel::semaphore_get(hal::to_underlying(S));
    }
    else
    {
        TTI_SEMGET(detail::semaphore_bit(S));
    }
}

/** @brief Decrement one runtime-selected semaphore through the chosen access path. */
template <Access A>
inline __attribute__((always_inline)) void get(const Semaphore semaphore)
{
    static_assert(detail::is_valid(A), "Semaphore access must be MMIO or Tensix");
    LLK_ASSERT(detail::is_valid(semaphore), "Semaphore index must be in [0, 7]");
    if constexpr (A == Access::MMIO)
    {
        ckernel::semaphore_get(hal::to_underlying(semaphore));
    }
    else
    {
        TT_SEMGET(detail::semaphore_bit(semaphore));
    }
}

/** @brief Read one semaphore value through the RISC MMIO window. */
template <Semaphore S>
inline __attribute__((always_inline)) std::uint8_t read()
{
    static_assert(detail::is_valid(S), "Semaphore index must be in [0, 7]");
    return ckernel::semaphore_read(hal::to_underlying(S));
}

/** @brief Read one runtime-selected semaphore value through the RISC MMIO window. */
inline __attribute__((always_inline)) std::uint8_t read(const Semaphore semaphore)
{
    LLK_ASSERT(detail::is_valid(semaphore), "Semaphore index must be in [0, 7]");
    return ckernel::semaphore_read(hal::to_underlying(semaphore));
}

} // namespace semaphore

namespace wait
{

/** @brief Encode STALLWAIT without issuing it. */
template <StallTarget Targets, StallCondition Conditions>
inline constexpr std::uint32_t stall_operation()
{
    static_assert(detail::is_valid(Targets), "STALLWAIT target mask must fit in nine bits");
    static_assert(detail::is_valid(Conditions), "Blackhole STALLWAIT condition mask must fit in 13 bits");
    return TT_OP_STALLWAIT(hal::to_underlying(Targets), hal::to_underlying(Conditions));
}

/** @brief Encode a runtime-selected STALLWAIT without issuing it. */
inline constexpr __attribute__((always_inline)) std::uint32_t stall_operation(const StallTarget targets, const StallCondition conditions)
{
    detail::require_valid_operand(detail::is_valid(targets), "STALLWAIT target mask must fit in nine bits");
    detail::require_valid_operand(detail::is_valid(conditions), "Blackhole STALLWAIT condition mask must fit in 13 bits");
    return TT_OP_STALLWAIT(hal::to_underlying(targets), hal::to_underlying(conditions));
}

/** @brief Install a compile-time STALLWAIT in the current thread's wait gate. */
template <StallTarget Targets, StallCondition Conditions>
inline __attribute__((always_inline)) void stall()
{
    (void)stall_operation<Targets, Conditions>();
    TTI_STALLWAIT(hal::to_underlying(Targets), hal::to_underlying(Conditions));
}

/** @brief Install a runtime STALLWAIT through the Tensix instruction buffer. */
inline __attribute__((always_inline)) void stall(const StallTarget targets, const StallCondition conditions)
{
    LLK_ASSERT(detail::is_valid(targets), "STALLWAIT target mask must fit in nine bits");
    LLK_ASSERT(detail::is_valid(conditions), "Blackhole STALLWAIT condition mask must fit in 13 bits");
    TT_STALLWAIT(hal::to_underlying(targets), hal::to_underlying(conditions));
}

/** @brief Encode SEMWAIT without issuing it. */
template <StallTarget Targets, SemaphoreMask Mask, SemaphoreCondition Conditions>
inline constexpr std::uint32_t semaphore_operation()
{
    static_assert(detail::is_valid(Targets), "SEMWAIT target mask must fit in nine bits");
    static_assert(detail::is_valid(Mask), "SEMWAIT requires at least one semaphore");
    static_assert(detail::is_valid(Conditions), "SEMWAIT requires WhileZero, WhileMaximum, or both");
    return TT_OP_SEMWAIT(hal::to_underlying(Targets), hal::to_underlying(Mask), hal::to_underlying(Conditions));
}

/** @brief Encode a runtime-selected SEMWAIT without issuing it. */
inline constexpr __attribute__((always_inline)) std::uint32_t semaphore_operation(
    const StallTarget targets, const SemaphoreMask mask, const SemaphoreCondition conditions)
{
    detail::require_valid_operand(detail::is_valid(targets), "SEMWAIT target mask must fit in nine bits");
    detail::require_valid_operand(detail::is_valid(mask), "SEMWAIT requires at least one semaphore");
    detail::require_valid_operand(detail::is_valid(conditions), "SEMWAIT requires WhileZero, WhileMaximum, or both");
    return TT_OP_SEMWAIT(hal::to_underlying(targets), hal::to_underlying(mask), hal::to_underlying(conditions));
}

/** @brief Install a compile-time SEMWAIT in the current thread's wait gate. */
template <StallTarget Targets, SemaphoreMask Mask, SemaphoreCondition Conditions>
inline __attribute__((always_inline)) void semaphore()
{
    (void)semaphore_operation<Targets, Mask, Conditions>();
    TTI_SEMWAIT(hal::to_underlying(Targets), hal::to_underlying(Mask), hal::to_underlying(Conditions));
}

/** @brief Install a runtime SEMWAIT through the Tensix instruction buffer. */
inline __attribute__((always_inline)) void semaphore(const StallTarget targets, const SemaphoreMask mask, const SemaphoreCondition conditions)
{
    LLK_ASSERT(detail::is_valid(targets), "SEMWAIT target mask must fit in nine bits");
    LLK_ASSERT(detail::is_valid(mask), "SEMWAIT requires at least one semaphore");
    LLK_ASSERT(detail::is_valid(conditions), "SEMWAIT requires WhileZero, WhileMaximum, or both");
    TT_SEMWAIT(hal::to_underlying(targets), hal::to_underlying(mask), hal::to_underlying(conditions));
}

} // namespace wait

// Descriptors encode one Tensix instruction. operation() is usable in constant
// expressions; invalid constants fail compilation and runtime operands use LLK_ASSERT.

/** @brief Encode ATGETM without acquiring the mutex. */
struct MutexAcquire
{
    Mutex selector;

    constexpr std::uint32_t operation() const
    {
        return mutex::acquire_operation(selector);
    }
};

/** @brief Encode ATRELM without releasing the mutex. */
struct MutexRelease
{
    Mutex selector;

    constexpr std::uint32_t operation() const
    {
        return mutex::release_operation(selector);
    }
};

/** @brief Encode SEMINIT without initializing the semaphores. */
struct SemaphoreInit
{
    SemaphoreMask mask;
    std::uint32_t initial;
    std::uint32_t maximum;

    constexpr std::uint32_t operation() const
    {
        return semaphore::init_operation(mask, initial, maximum);
    }
};

/** @brief Encode SEMPOST without incrementing the semaphores. */
struct SemaphorePost
{
    SemaphoreMask mask;

    constexpr std::uint32_t operation() const
    {
        return semaphore::post_operation(mask);
    }
};

/** @brief Encode SEMGET without decrementing the semaphores. */
struct SemaphoreGet
{
    SemaphoreMask mask;

    constexpr std::uint32_t operation() const
    {
        return semaphore::get_operation(mask);
    }
};

/** @brief Encode STALLWAIT without installing a wait gate. */
struct StallWait
{
    StallTarget targets;
    StallCondition conditions;

    constexpr std::uint32_t operation() const
    {
        return wait::stall_operation(targets, conditions);
    }
};

/** @brief Encode SEMWAIT without installing a wait gate. */
struct SemaphoreWait
{
    StallTarget targets;
    SemaphoreMask mask;
    SemaphoreCondition conditions;

    constexpr std::uint32_t operation() const
    {
        return wait::semaphore_operation(targets, mask, conditions);
    }
};

} // namespace hal::sync
