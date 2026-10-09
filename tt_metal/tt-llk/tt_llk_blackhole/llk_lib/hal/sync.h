// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

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

/**
 * @brief Select one of the eight physical Tensix semaphores by index.
 *
 * Tensix operations build the required bitmask internally. For example,
 * semaphore::get<Semaphore::S1, Semaphore::S3>() selects both atomically;
 * the runtime form is semaphore::get({first, second}). MMIO uses the index.
 */
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

// Accept a Semaphore or a braced list of Semaphores; retain only the encoded mask.
class SemaphoreSet
{
public:
    template <typename... Others>
    constexpr SemaphoreSet(const Semaphore first, const Others... others) : mask_(0)
    {
        static_assert((std::is_same_v<Others, Semaphore> && ...), "Semaphore selections must contain only Semaphore values");
        require_valid_operand(is_valid(first) && (is_valid(others) && ...), "Semaphore index must be in [0, 7]");
        mask_ = semaphore_bit(first) | (0u | ... | semaphore_bit(others));
    }

    constexpr std::uint32_t mask() const
    {
        return mask_;
    }

private:
    std::uint32_t mask_;
};

template <Semaphore First, Semaphore... Others>
constexpr std::uint32_t semaphore_mask()
{
    static_assert(is_valid(First) && (is_valid(Others) && ...), "Semaphore index must be in [0, 7]");
    return SemaphoreSet {First, Others...}.mask();
}

// These duplicate ckernel::semaphore_read, semaphore_post, and semaphore_get.
// TODO(njokovic) issue #58443: Remove ckernel:: implementations when HAL is applied to all kernels.

/** @brief Read a semaphore value from the shared PC-buffer MMIO window. */
inline __attribute__((always_inline)) std::uint8_t semaphore_read_mmio(const std::uint8_t index)
{
    LLK_ASSERT(index < ckernel::semaphore::NUM_SEMAPHORES, "Semaphore index out of bounds.");
    return ckernel::pc_buf_base[ckernel::PC_BUF_SEMAPHORE_BASE + index];
}

/** @brief Release one token with an atomic MMIO increment, capped at 15. */
inline __attribute__((always_inline)) void semaphore_post_mmio(const std::uint8_t index)
{
    LLK_ASSERT(index < ckernel::semaphore::NUM_SEMAPHORES, "Semaphore index out of bounds.");
    LLK_ASSERT(semaphore_read_mmio(index) < ckernel::semaphore::SEMAPHORE_MAX_VALUE, "Semaphore must not be already at max value.");
    ckernel::pc_buf_base[ckernel::PC_BUF_SEMAPHORE_BASE + index] = 0; // LSB clear selects SEMPOST.
}

/** @brief Acquire one token with an atomic MMIO decrement, floored at zero. */
inline __attribute__((always_inline)) void semaphore_get_mmio(const std::uint8_t index)
{
    LLK_ASSERT(index < ckernel::semaphore::NUM_SEMAPHORES, "Semaphore index out of bounds.");
    LLK_ASSERT(semaphore_read_mmio(index) > 0, "Semaphore must not be already at 0.");
    ckernel::pc_buf_base[ckernel::PC_BUF_SEMAPHORE_BASE + index] = 1; // LSB set selects SEMGET.
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
    TTI_INSN((acquire_operation<M>()));
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
    TTI_INSN((release_operation<M>()));
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

/** @brief Encode SEMINIT for the selected semaphores without issuing it. */
template <std::uint32_t Initial, std::uint32_t Maximum, Semaphore First, Semaphore... Others>
inline constexpr std::uint32_t init_operation()
{
    static_assert(Initial < 16u, "SEMINIT initial value must fit in four bits");
    static_assert(Maximum < 16u, "SEMINIT maximum value must fit in four bits");
    return TT_OP_SEMINIT(Maximum, Initial, (detail::semaphore_mask<First, Others...>()));
}

/** @brief Encode SEMINIT for a semaphore or braced list of semaphores without issuing it. */
inline constexpr __attribute__((always_inline)) std::uint32_t init_operation(
    const detail::SemaphoreSet semaphores, const std::uint32_t initial, const std::uint32_t maximum)
{
    detail::require_valid_operand(initial < 16u, "SEMINIT initial value must fit in four bits");
    detail::require_valid_operand(maximum < 16u, "SEMINIT maximum value must fit in four bits");
    return TT_OP_SEMINIT(maximum, initial, semaphores.mask());
}

/**
 * @brief Initialize compile-time-selected semaphores through Tensix.
 *
 * For example, init<0, 1, Semaphore::S1, Semaphore::S3>() initializes both
 * semaphores with one instruction. MMIO has no equivalent to SEMINIT.
 */
template <std::uint32_t Initial, std::uint32_t Maximum, Semaphore First, Semaphore... Others>
inline __attribute__((always_inline)) void init()
{
    TTI_INSN((init_operation<Initial, Maximum, First, Others...>()));
}

/** @brief Initialize a semaphore or braced list of semaphores through Tensix. */
inline __attribute__((always_inline)) void init(const detail::SemaphoreSet semaphores, const std::uint32_t initial, const std::uint32_t maximum)
{
    TT_INSN(init_operation(semaphores, initial, maximum));
}

/** @brief Encode SEMPOST for the selected semaphores without issuing it. */
template <Semaphore First, Semaphore... Others>
inline constexpr std::uint32_t post_operation()
{
    return TT_OP_SEMPOST((detail::semaphore_mask<First, Others...>()));
}

/** @brief Encode SEMPOST for a semaphore or braced list of semaphores without issuing it. */
inline constexpr __attribute__((always_inline)) std::uint32_t post_operation(const detail::SemaphoreSet semaphores)
{
    return TT_OP_SEMPOST(semaphores.mask());
}

/** @brief Increment compile-time-selected semaphores atomically through Tensix. */
template <Semaphore First, Semaphore... Others>
inline __attribute__((always_inline)) void post()
{
    TTI_INSN((post_operation<First, Others...>()));
}

/** @brief Increment a semaphore or braced list of semaphores atomically through Tensix. */
inline __attribute__((always_inline)) void post(const detail::SemaphoreSet semaphores)
{
    TT_INSN(post_operation(semaphores));
}

/** @brief Increment one compile-time-selected semaphore through the chosen access path. */
template <Access A, Semaphore S>
inline __attribute__((always_inline)) void post()
{
    static_assert(detail::is_valid(A), "Semaphore access must be MMIO or Tensix");
    static_assert(detail::is_valid(S), "Semaphore index must be in [0, 7]");
    if constexpr (A == Access::MMIO)
    {
        detail::semaphore_post_mmio(hal::to_underlying(S));
    }
    else
    {
        post<S>();
    }
}

/** @brief Increment one runtime-selected semaphore through the chosen access path. */
template <Access A>
inline __attribute__((always_inline)) void post(const Semaphore semaphore)
{
    static_assert(detail::is_valid(A), "Semaphore access must be MMIO or Tensix");
    if constexpr (A == Access::MMIO)
    {
        LLK_ASSERT(detail::is_valid(semaphore), "Semaphore index must be in [0, 7]");
        detail::semaphore_post_mmio(hal::to_underlying(semaphore));
    }
    else
    {
        post(semaphore);
    }
}

/** @brief Encode SEMGET for the selected semaphores without issuing it. */
template <Semaphore First, Semaphore... Others>
inline constexpr std::uint32_t get_operation()
{
    return TT_OP_SEMGET((detail::semaphore_mask<First, Others...>()));
}

/** @brief Encode SEMGET for a semaphore or braced list of semaphores without issuing it. */
inline constexpr __attribute__((always_inline)) std::uint32_t get_operation(const detail::SemaphoreSet semaphores)
{
    return TT_OP_SEMGET(semaphores.mask());
}

/** @brief Decrement compile-time-selected semaphores atomically through Tensix. */
template <Semaphore First, Semaphore... Others>
inline __attribute__((always_inline)) void get()
{
    TTI_INSN((get_operation<First, Others...>()));
}

/** @brief Decrement a semaphore or braced list of semaphores atomically through Tensix. */
inline __attribute__((always_inline)) void get(const detail::SemaphoreSet semaphores)
{
    TT_INSN(get_operation(semaphores));
}

/** @brief Decrement one compile-time-selected semaphore through the chosen access path. */
template <Access A, Semaphore S>
inline __attribute__((always_inline)) void get()
{
    static_assert(detail::is_valid(A), "Semaphore access must be MMIO or Tensix");
    static_assert(detail::is_valid(S), "Semaphore index must be in [0, 7]");
    if constexpr (A == Access::MMIO)
    {
        detail::semaphore_get_mmio(hal::to_underlying(S));
    }
    else
    {
        get<S>();
    }
}

/** @brief Decrement one runtime-selected semaphore through the chosen access path. */
template <Access A>
inline __attribute__((always_inline)) void get(const Semaphore semaphore)
{
    static_assert(detail::is_valid(A), "Semaphore access must be MMIO or Tensix");
    if constexpr (A == Access::MMIO)
    {
        LLK_ASSERT(detail::is_valid(semaphore), "Semaphore index must be in [0, 7]");
        detail::semaphore_get_mmio(hal::to_underlying(semaphore));
    }
    else
    {
        get(semaphore);
    }
}

/** @brief Read one semaphore value through the RISC MMIO window. */
template <Semaphore S>
inline __attribute__((always_inline)) std::uint8_t read()
{
    static_assert(detail::is_valid(S), "Semaphore index must be in [0, 7]");
    return detail::semaphore_read_mmio(hal::to_underlying(S));
}

/** @brief Read one runtime-selected semaphore value through the RISC MMIO window. */
inline __attribute__((always_inline)) std::uint8_t read(const Semaphore semaphore)
{
    LLK_ASSERT(detail::is_valid(semaphore), "Semaphore index must be in [0, 7]");
    return detail::semaphore_read_mmio(hal::to_underlying(semaphore));
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

/** @brief Encode SEMWAIT for the selected semaphores without issuing it. */
template <StallTarget Targets, SemaphoreCondition Conditions, Semaphore First, Semaphore... Others>
inline constexpr std::uint32_t semaphore_operation()
{
    static_assert(detail::is_valid(Targets), "SEMWAIT target mask must fit in nine bits");
    static_assert(detail::is_valid(Conditions), "SEMWAIT requires WhileZero, WhileMaximum, or both");
    return TT_OP_SEMWAIT(hal::to_underlying(Targets), (detail::semaphore_mask<First, Others...>()), hal::to_underlying(Conditions));
}

/** @brief Encode SEMWAIT for a semaphore or braced list of semaphores without issuing it. */
inline constexpr __attribute__((always_inline)) std::uint32_t semaphore_operation(
    const StallTarget targets, const detail::SemaphoreSet semaphores, const SemaphoreCondition conditions)
{
    detail::require_valid_operand(detail::is_valid(targets), "SEMWAIT target mask must fit in nine bits");
    detail::require_valid_operand(detail::is_valid(conditions), "SEMWAIT requires WhileZero, WhileMaximum, or both");
    return TT_OP_SEMWAIT(hal::to_underlying(targets), semaphores.mask(), hal::to_underlying(conditions));
}

/** @brief Install a compile-time SEMWAIT for the selected semaphores in the current thread's wait gate. */
template <StallTarget Targets, SemaphoreCondition Conditions, Semaphore First, Semaphore... Others>
inline __attribute__((always_inline)) void semaphore()
{
    TTI_INSN((semaphore_operation<Targets, Conditions, First, Others...>()));
}

/** @brief Install SEMWAIT for a semaphore or braced list of semaphores through Tensix. */
inline __attribute__((always_inline)) void semaphore(const StallTarget targets, const detail::SemaphoreSet semaphores, const SemaphoreCondition conditions)
{
    TT_INSN(semaphore_operation(targets, semaphores, conditions));
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
    detail::SemaphoreSet semaphores;
    std::uint32_t initial;
    std::uint32_t maximum;

    constexpr std::uint32_t operation() const
    {
        return semaphore::init_operation(semaphores, initial, maximum);
    }
};

/** @brief Encode SEMPOST without incrementing the semaphores. */
struct SemaphorePost
{
    detail::SemaphoreSet semaphores;

    constexpr std::uint32_t operation() const
    {
        return semaphore::post_operation(semaphores);
    }
};

/** @brief Encode SEMGET without decrementing the semaphores. */
struct SemaphoreGet
{
    detail::SemaphoreSet semaphores;

    constexpr std::uint32_t operation() const
    {
        return semaphore::get_operation(semaphores);
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
    detail::SemaphoreSet semaphores;
    SemaphoreCondition conditions;

    constexpr std::uint32_t operation() const
    {
        return wait::semaphore_operation(targets, semaphores, conditions);
    }
};

} // namespace hal::sync
