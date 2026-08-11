// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cfg.h"
#include "ckernel.h"

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
 * The M0/M2/M3/M4 names are physical indices. The semantic aliases preserve
 * both the ISA labels and the roles already used by LLK code.
 */
enum class Mutex : std::uint8_t
{
    M0 = 0,
    M2 = 2,
    M3 = 3,
    M4 = 4,

    Math      = M0,
    Unpacker0 = M2,
    Unpacker1 = M3,
    Packer0   = M4,

    RegisterRmw = M0,
    Sfpu        = M4
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
    S7 = 7,

    FpuSfpu           = S0,
    MathPack          = S1,
    UnpackToDest      = S2,
    UnpackOperandSync = S3,
    PackDone          = S4,
    UnpackSync        = S5,
    UnpackMathDone    = S6,
    MathDone          = S7
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
    All  = 0xffu,

    FpuSfpu           = S0,
    MathPack          = S1,
    UnpackToDest      = S2,
    UnpackOperandSync = S3,
    PackDone          = S4,
    UnpackSync        = S5,
    UnpackMathDone    = S6,
    MathDone          = S7
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

/** @brief Select one of the four STREAM_ID_SYNC thread-CFG entries. */
enum class StreamSlot : std::uint8_t
{
    S0,
    S1,
    S2,
    S3
};

/** @brief Select the STREAMWAIT threshold source. */
enum class StreamTarget : std::uint8_t
{
    Phase,
    MessagesReceived
};

/** @brief Identify a NoC overlay stream by its three-bit group and stream fields. */
struct StreamId
{
    std::uint8_t group;
    std::uint8_t stream;
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
inline constexpr std::uint32_t STREAM_TARGET_LOW_BITS = 10;
inline constexpr std::uint32_t STREAM_TARGET_LOW_MASK = (1u << STREAM_TARGET_LOW_BITS) - 1u;

constexpr std::uint32_t value(const Mutex mutex)
{
    return static_cast<std::uint8_t>(mutex);
}

constexpr std::uint32_t value(const Semaphore semaphore)
{
    return static_cast<std::uint8_t>(semaphore);
}

constexpr std::uint32_t value(const SemaphoreMask mask)
{
    return static_cast<std::uint8_t>(mask);
}

constexpr std::uint32_t value(const StallTarget targets)
{
    return static_cast<std::uint16_t>(targets);
}

constexpr std::uint32_t value(const StallCondition conditions)
{
    return static_cast<std::uint16_t>(conditions);
}

constexpr std::uint32_t value(const SemaphoreCondition conditions)
{
    return static_cast<std::uint8_t>(conditions);
}

constexpr std::uint32_t value(const StreamSlot slot)
{
    return static_cast<std::uint8_t>(slot);
}

constexpr std::uint32_t value(const StreamTarget target)
{
    return static_cast<std::uint8_t>(target);
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
    return value(semaphore) < 8u;
}

constexpr bool is_valid(const SemaphoreMask mask)
{
    return value(mask) != 0u;
}

constexpr bool is_valid(const StallTarget targets)
{
    return value(targets) <= value(StallTarget::All);
}

constexpr bool is_valid(const StallCondition conditions)
{
    return value(conditions) <= value(StallCondition::All);
}

constexpr bool is_valid(const SemaphoreCondition conditions)
{
    const std::uint32_t encoded = value(conditions);
    return encoded != 0u && encoded <= 0x3u;
}

constexpr bool is_valid(const StreamSlot slot)
{
    return value(slot) < 4u;
}

constexpr bool is_valid(const StreamTarget target)
{
    return target == StreamTarget::Phase || target == StreamTarget::MessagesReceived;
}

constexpr bool is_valid(const StreamId stream)
{
    return stream.group < 8u && stream.stream < 8u;
}

constexpr std::uint32_t encode(const StreamId stream)
{
    return (static_cast<std::uint32_t>(stream.group) << 3) | stream.stream;
}

constexpr std::uint32_t semaphore_bit(const Semaphore semaphore)
{
    return 1u << value(semaphore);
}

template <StreamSlot Slot>
inline constexpr cfg::Sec stream_section = static_cast<cfg::Sec>(value(Slot));

template <StreamTarget Target>
inline constexpr std::uint32_t max_stream_target = Target == StreamTarget::Phase ? ((1u << 20) - 1u) : ((1u << 17) - 1u);

} // namespace detail

namespace mutex
{

/** @brief Encode ATGETM without issuing it. */
template <Mutex M>
inline constexpr std::uint32_t acquire_operation()
{
    static_assert(detail::is_valid(M), "Blackhole mutex index must be 0 or in [2, 4]");
    return TT_OP_ATGETM(detail::value(M));
}

/** @brief Encode a runtime-selected ATGETM without issuing it. */
inline __attribute__((always_inline)) std::uint32_t acquire_operation(const Mutex mutex)
{
    LLK_ASSERT(detail::is_valid(mutex), "Blackhole mutex index must be 0 or in [2, 4]");
    return TT_OP_ATGETM(detail::value(mutex));
}

/** @brief Acquire a compile-time-selected mutex with one immediate ATGETM. */
template <Mutex M>
inline __attribute__((always_inline)) void acquire()
{
    (void)acquire_operation<M>();
    TTI_ATGETM(detail::value(M));
}

/** @brief Acquire a runtime-selected mutex through the Tensix instruction buffer. */
inline __attribute__((always_inline)) void acquire(const Mutex mutex)
{
    LLK_ASSERT(detail::is_valid(mutex), "Blackhole mutex index must be 0 or in [2, 4]");
    TT_ATGETM(detail::value(mutex));
}

/** @brief Encode ATRELM without issuing it. */
template <Mutex M>
inline constexpr std::uint32_t release_operation()
{
    static_assert(detail::is_valid(M), "Blackhole mutex index must be 0 or in [2, 4]");
    return TT_OP_ATRELM(detail::value(M));
}

/** @brief Encode a runtime-selected ATRELM without issuing it. */
inline __attribute__((always_inline)) std::uint32_t release_operation(const Mutex mutex)
{
    LLK_ASSERT(detail::is_valid(mutex), "Blackhole mutex index must be 0 or in [2, 4]");
    return TT_OP_ATRELM(detail::value(mutex));
}

/** @brief Release a compile-time-selected mutex with one immediate ATRELM. */
template <Mutex M>
inline __attribute__((always_inline)) void release()
{
    (void)release_operation<M>();
    TTI_ATRELM(detail::value(M));
}

/** @brief Release a runtime-selected mutex through the Tensix instruction buffer. */
inline __attribute__((always_inline)) void release(const Mutex mutex)
{
    LLK_ASSERT(detail::is_valid(mutex), "Blackhole mutex index must be 0 or in [2, 4]");
    TT_ATRELM(detail::value(mutex));
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
    return TT_OP_SEMINIT(Maximum, Initial, detail::value(Mask));
}

/** @brief Encode a runtime-selected SEMINIT without issuing it. */
inline __attribute__((always_inline)) std::uint32_t init_operation(const SemaphoreMask mask, const std::uint32_t initial, const std::uint32_t maximum)
{
    LLK_ASSERT(detail::is_valid(mask), "SEMINIT requires at least one semaphore");
    LLK_ASSERT(initial < 16u, "SEMINIT initial value must fit in four bits");
    LLK_ASSERT(maximum < 16u, "SEMINIT maximum value must fit in four bits");
    return TT_OP_SEMINIT(maximum, initial, detail::value(mask));
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
    TTI_SEMINIT(Maximum, Initial, detail::value(Mask));
}

/** @brief Initialize runtime-selected semaphores through the Tensix instruction buffer. */
inline __attribute__((always_inline)) void init(const SemaphoreMask mask, const std::uint32_t initial, const std::uint32_t maximum)
{
    LLK_ASSERT(detail::is_valid(mask), "SEMINIT requires at least one semaphore");
    LLK_ASSERT(initial < 16u, "SEMINIT initial value must fit in four bits");
    LLK_ASSERT(maximum < 16u, "SEMINIT maximum value must fit in four bits");
    TT_SEMINIT(maximum, initial, detail::value(mask));
}

/** @brief Encode SEMPOST without issuing it. */
template <SemaphoreMask Mask>
inline constexpr std::uint32_t post_operation()
{
    static_assert(detail::is_valid(Mask), "SEMPOST requires at least one semaphore");
    return TT_OP_SEMPOST(detail::value(Mask));
}

/** @brief Encode a runtime-selected SEMPOST without issuing it. */
inline __attribute__((always_inline)) std::uint32_t post_operation(const SemaphoreMask mask)
{
    LLK_ASSERT(detail::is_valid(mask), "SEMPOST requires at least one semaphore");
    return TT_OP_SEMPOST(detail::value(mask));
}

/** @brief Increment compile-time-selected semaphores through Tensix. */
template <SemaphoreMask Mask>
inline __attribute__((always_inline)) void post()
{
    (void)post_operation<Mask>();
    TTI_SEMPOST(detail::value(Mask));
}

/** @brief Increment runtime-selected semaphores through Tensix. */
inline __attribute__((always_inline)) void post(const SemaphoreMask mask)
{
    LLK_ASSERT(detail::is_valid(mask), "SEMPOST requires at least one semaphore");
    TT_SEMPOST(detail::value(mask));
}

/** @brief Increment one compile-time-selected semaphore through the chosen access path. */
template <Access A, Semaphore S>
inline __attribute__((always_inline)) void post()
{
    static_assert(detail::is_valid(A), "Semaphore access must be MMIO or Tensix");
    static_assert(detail::is_valid(S), "Semaphore index must be in [0, 7]");
    if constexpr (A == Access::MMIO)
    {
        ckernel::semaphore_post(detail::value(S));
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
        ckernel::semaphore_post(detail::value(semaphore));
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
    return TT_OP_SEMGET(detail::value(Mask));
}

/** @brief Encode a runtime-selected SEMGET without issuing it. */
inline __attribute__((always_inline)) std::uint32_t get_operation(const SemaphoreMask mask)
{
    LLK_ASSERT(detail::is_valid(mask), "SEMGET requires at least one semaphore");
    return TT_OP_SEMGET(detail::value(mask));
}

/** @brief Decrement compile-time-selected semaphores through Tensix. */
template <SemaphoreMask Mask>
inline __attribute__((always_inline)) void get()
{
    (void)get_operation<Mask>();
    TTI_SEMGET(detail::value(Mask));
}

/** @brief Decrement runtime-selected semaphores through Tensix. */
inline __attribute__((always_inline)) void get(const SemaphoreMask mask)
{
    LLK_ASSERT(detail::is_valid(mask), "SEMGET requires at least one semaphore");
    TT_SEMGET(detail::value(mask));
}

/** @brief Decrement one compile-time-selected semaphore through the chosen access path. */
template <Access A, Semaphore S>
inline __attribute__((always_inline)) void get()
{
    static_assert(detail::is_valid(A), "Semaphore access must be MMIO or Tensix");
    static_assert(detail::is_valid(S), "Semaphore index must be in [0, 7]");
    if constexpr (A == Access::MMIO)
    {
        ckernel::semaphore_get(detail::value(S));
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
        ckernel::semaphore_get(detail::value(semaphore));
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
    return ckernel::semaphore_read(detail::value(S));
}

/** @brief Read one runtime-selected semaphore value through the RISC MMIO window. */
inline __attribute__((always_inline)) std::uint8_t read(const Semaphore semaphore)
{
    LLK_ASSERT(detail::is_valid(semaphore), "Semaphore index must be in [0, 7]");
    return ckernel::semaphore_read(detail::value(semaphore));
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
    return TT_OP_STALLWAIT(detail::value(Targets), detail::value(Conditions));
}

/** @brief Encode a runtime-selected STALLWAIT without issuing it. */
inline __attribute__((always_inline)) std::uint32_t stall_operation(const StallTarget targets, const StallCondition conditions)
{
    LLK_ASSERT(detail::is_valid(targets), "STALLWAIT target mask must fit in nine bits");
    LLK_ASSERT(detail::is_valid(conditions), "Blackhole STALLWAIT condition mask must fit in 13 bits");
    return TT_OP_STALLWAIT(detail::value(targets), detail::value(conditions));
}

/** @brief Install a compile-time STALLWAIT in the current thread's wait gate. */
template <StallTarget Targets, StallCondition Conditions>
inline __attribute__((always_inline)) void stall()
{
    (void)stall_operation<Targets, Conditions>();
    TTI_STALLWAIT(detail::value(Targets), detail::value(Conditions));
}

/** @brief Install a runtime STALLWAIT through the Tensix instruction buffer. */
inline __attribute__((always_inline)) void stall(const StallTarget targets, const StallCondition conditions)
{
    LLK_ASSERT(detail::is_valid(targets), "STALLWAIT target mask must fit in nine bits");
    LLK_ASSERT(detail::is_valid(conditions), "Blackhole STALLWAIT condition mask must fit in 13 bits");
    TT_STALLWAIT(detail::value(targets), detail::value(conditions));
}

/** @brief Encode SEMWAIT without issuing it. */
template <StallTarget Targets, SemaphoreMask Mask, SemaphoreCondition Conditions>
inline constexpr std::uint32_t semaphore_operation()
{
    static_assert(detail::is_valid(Targets), "SEMWAIT target mask must fit in nine bits");
    static_assert(detail::is_valid(Mask), "SEMWAIT requires at least one semaphore");
    static_assert(detail::is_valid(Conditions), "SEMWAIT requires WhileZero, WhileMaximum, or both");
    return TT_OP_SEMWAIT(detail::value(Targets), detail::value(Mask), detail::value(Conditions));
}

/** @brief Encode a runtime-selected SEMWAIT without issuing it. */
inline __attribute__((always_inline)) std::uint32_t semaphore_operation(
    const StallTarget targets, const SemaphoreMask mask, const SemaphoreCondition conditions)
{
    LLK_ASSERT(detail::is_valid(targets), "SEMWAIT target mask must fit in nine bits");
    LLK_ASSERT(detail::is_valid(mask), "SEMWAIT requires at least one semaphore");
    LLK_ASSERT(detail::is_valid(conditions), "SEMWAIT requires WhileZero, WhileMaximum, or both");
    return TT_OP_SEMWAIT(detail::value(targets), detail::value(mask), detail::value(conditions));
}

/** @brief Install a compile-time SEMWAIT in the current thread's wait gate. */
template <StallTarget Targets, SemaphoreMask Mask, SemaphoreCondition Conditions>
inline __attribute__((always_inline)) void semaphore()
{
    (void)semaphore_operation<Targets, Mask, Conditions>();
    TTI_SEMWAIT(detail::value(Targets), detail::value(Mask), detail::value(Conditions));
}

/** @brief Install a runtime SEMWAIT through the Tensix instruction buffer. */
inline __attribute__((always_inline)) void semaphore(const StallTarget targets, const SemaphoreMask mask, const SemaphoreCondition conditions)
{
    LLK_ASSERT(detail::is_valid(targets), "SEMWAIT target mask must fit in nine bits");
    LLK_ASSERT(detail::is_valid(mask), "SEMWAIT requires at least one semaphore");
    LLK_ASSERT(detail::is_valid(conditions), "SEMWAIT requires WhileZero, WhileMaximum, or both");
    TT_SEMWAIT(detail::value(targets), detail::value(mask), detail::value(conditions));
}

/** @brief Program one STREAM_ID_SYNC slot with a compile-time NoC stream ID. */
template <StreamSlot Slot, std::uint32_t Group, std::uint32_t Stream>
inline __attribute__((always_inline)) void configure_stream()
{
    static_assert(detail::is_valid(Slot), "STREAMWAIT slot must be in [0, 3]");
    static_assert(Group < 8u, "NoC stream group must fit in three bits");
    static_assert(Stream < 8u, "NoC stream number must fit in three bits");
    constexpr std::uint32_t stream_id = (Group << 3) | Stream;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::StreamIdSync::BankSel, stream_id, detail::stream_section<Slot>>();
}

/** @brief Program one runtime-selected STREAM_ID_SYNC slot and NoC stream ID. */
inline __attribute__((always_inline)) void configure_stream(const StreamSlot slot, const StreamId stream)
{
    LLK_ASSERT(detail::is_valid(slot), "STREAMWAIT slot must be in [0, 3]");
    LLK_ASSERT(detail::is_valid(stream), "NoC stream group and number must each fit in three bits");
    const std::uint32_t stream_id = detail::encode(stream);
    switch (slot)
    {
        case StreamSlot::S0:
            cfg::write<cfg::Access::TensixCfgUnit, cfg::StreamIdSync::BankSel, cfg::Sec::S0>(stream_id);
            break;
        case StreamSlot::S1:
            cfg::write<cfg::Access::TensixCfgUnit, cfg::StreamIdSync::BankSel, cfg::Sec::S1>(stream_id);
            break;
        case StreamSlot::S2:
            cfg::write<cfg::Access::TensixCfgUnit, cfg::StreamIdSync::BankSel, cfg::Sec::S2>(stream_id);
            break;
        case StreamSlot::S3:
            cfg::write<cfg::Access::TensixCfgUnit, cfg::StreamIdSync::BankSel, cfg::Sec::S3>(stream_id);
            break;
        default:
            LLK_ASSERT(false, "STREAMWAIT slot must be in [0, 3]");
            break;
    }
}

/** @brief Program the high bits of a compile-time STREAMWAIT threshold. */
template <StreamTarget Target, std::uint32_t FullTarget>
inline __attribute__((always_inline)) void configure_stream_target()
{
    static_assert(detail::is_valid(Target), "STREAMWAIT target must be Phase or MessagesReceived");
    static_assert(FullTarget <= detail::max_stream_target<Target>, "STREAMWAIT target exceeds the selected counter width");
    if constexpr (Target == StreamTarget::Phase)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::StreamwaitPhaseHi::Val, (FullTarget >> detail::STREAM_TARGET_LOW_BITS)>();
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::StreamwaitNumMsgsHi::Val, (FullTarget >> detail::STREAM_TARGET_LOW_BITS)>();
    }
}

/** @brief Program the high bits of a runtime STREAMWAIT threshold. */
inline __attribute__((always_inline)) void configure_stream_target(const StreamTarget target, const std::uint32_t full_target)
{
    LLK_ASSERT(detail::is_valid(target), "STREAMWAIT target must be Phase or MessagesReceived");
    LLK_ASSERT(
        (target == StreamTarget::Phase && full_target <= detail::max_stream_target<StreamTarget::Phase>) ||
            (target == StreamTarget::MessagesReceived && full_target <= detail::max_stream_target<StreamTarget::MessagesReceived>),
        "STREAMWAIT target exceeds the selected counter width");
    if (target == StreamTarget::Phase)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::StreamwaitPhaseHi::Val>(full_target >> detail::STREAM_TARGET_LOW_BITS);
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::StreamwaitNumMsgsHi::Val>(full_target >> detail::STREAM_TARGET_LOW_BITS);
    }
}

/** @brief Encode one raw STREAMWAIT without issuing it. */
template <StallTarget Targets, StreamSlot Slot, StreamTarget Target, std::uint32_t TargetLow>
inline constexpr std::uint32_t stream_operation()
{
    static_assert(detail::is_valid(Targets), "STREAMWAIT target mask must fit in nine bits");
    static_assert(detail::is_valid(Slot), "STREAMWAIT slot must be in [0, 3]");
    static_assert(detail::is_valid(Target), "STREAMWAIT target must be Phase or MessagesReceived");
    static_assert(TargetLow <= detail::STREAM_TARGET_LOW_MASK, "STREAMWAIT low target must fit in ten bits");
    return TT_OP_STREAMWAIT(detail::value(Targets), TargetLow, detail::value(Target), detail::value(Slot));
}

/** @brief Encode a runtime-selected raw STREAMWAIT without issuing it. */
inline __attribute__((always_inline)) std::uint32_t stream_operation(
    const StallTarget targets, const StreamSlot slot, const StreamTarget target, const std::uint32_t target_low)
{
    LLK_ASSERT(detail::is_valid(targets), "STREAMWAIT target mask must fit in nine bits");
    LLK_ASSERT(detail::is_valid(slot), "STREAMWAIT slot must be in [0, 3]");
    LLK_ASSERT(detail::is_valid(target), "STREAMWAIT target must be Phase or MessagesReceived");
    LLK_ASSERT(target_low <= detail::STREAM_TARGET_LOW_MASK, "STREAMWAIT low target must fit in ten bits");
    return TT_OP_STREAMWAIT(detail::value(targets), target_low, detail::value(target), detail::value(slot));
}

/**
 * @brief Install one raw compile-time STREAMWAIT.
 *
 * Configure the selected STREAM_ID_SYNC slot and the chosen target-high field
 * before calling this one-instruction form.
 */
template <StallTarget Targets, StreamSlot Slot, StreamTarget Target, std::uint32_t TargetLow>
inline __attribute__((always_inline)) void stream()
{
    (void)stream_operation<Targets, Slot, Target, TargetLow>();
    TTI_STREAMWAIT(detail::value(Targets), TargetLow, detail::value(Target), detail::value(Slot));
}

/** @brief Install one raw runtime STREAMWAIT through the Tensix instruction buffer. */
inline __attribute__((always_inline)) void stream(const StallTarget targets, const StreamSlot slot, const StreamTarget target, const std::uint32_t target_low)
{
    LLK_ASSERT(detail::is_valid(targets), "STREAMWAIT target mask must fit in nine bits");
    LLK_ASSERT(detail::is_valid(slot), "STREAMWAIT slot must be in [0, 3]");
    LLK_ASSERT(detail::is_valid(target), "STREAMWAIT target must be Phase or MessagesReceived");
    LLK_ASSERT(target_low <= detail::STREAM_TARGET_LOW_MASK, "STREAMWAIT low target must fit in ten bits");
    TT_STREAMWAIT(detail::value(targets), target_low, detail::value(target), detail::value(slot));
}

/** @brief Configure a stream and full compile-time threshold, then install STREAMWAIT. */
template <StallTarget Targets, StreamSlot Slot, std::uint32_t Group, std::uint32_t Stream, StreamTarget Target, std::uint32_t FullTarget>
inline __attribute__((always_inline)) void configure_and_wait_stream()
{
    configure_stream<Slot, Group, Stream>();
    configure_stream_target<Target, FullTarget>();
    stall<StallTarget::Sync, StallCondition::ConfigUnitIdle>();
    stream<Targets, Slot, Target, FullTarget & detail::STREAM_TARGET_LOW_MASK>();
}

/** @brief Configure a stream and full runtime threshold, then install STREAMWAIT. */
inline __attribute__((always_inline)) void configure_and_wait_stream(
    const StallTarget targets, const StreamSlot slot, const StreamId stream_id, const StreamTarget target, const std::uint32_t full_target)
{
    configure_stream(slot, stream_id);
    configure_stream_target(target, full_target);
    stall(StallTarget::Sync, StallCondition::ConfigUnitIdle);
    stream(targets, slot, target, full_target & detail::STREAM_TARGET_LOW_MASK);
}

} // namespace wait

inline constexpr bool supports_stream_wait = true;

} // namespace hal::sync
