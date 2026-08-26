// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

namespace hal::unpack
{

/** @brief Select the Blackhole unpacker that executes an UNPACR instruction. */
enum class Engine : std::uint8_t
{
    Unpacker0 = 0,
    Unpacker1 = 1
};

/** @brief Select how an UNPACR data transfer resolves its configuration context. */
enum class ContextSource : std::uint8_t
{
    ThreadDefault,
    Explicit,
    Counter
};

/** @brief Select whether a transfer retains or hands off its source bank. */
enum class SourceHandoff : std::uint8_t
{
    Unset               = 0xFF,
    Keep                = 0,
    FlipAndSetDataValid = 1
};

/** @brief Select whether transferred datums retain their values or become zero. */
enum class DatumOverride : bool
{
    None = false,
    Zero = true
};

/** @brief Select datum-range or compressed-row-start search. */
enum class SearchMode : bool
{
    DatumRange = false,
    Row        = true
};

/** @brief Select whether the transfer continues or flushes the current accumulation task. */
enum class AccumulationAction : bool
{
    Continue = false,
    Flush    = true
};

/** @brief Select which compressed-row-start cache entries an UNPACR flush clears. */
enum class CacheScope : bool
{
    CurrentThread = false,
    AllEntries    = true
};

/** @brief Describe the per-channel address-counter increments applied after a transfer. */
struct AddressCounterPostIncrements
{
    /** @brief Describe the two-bit Y and Z increments supported by one UNPACR channel. */
    struct Channel
    {
        std::uint8_t y = 0;
        std::uint8_t z = 0;
    };

    Channel channel0 = {};
    Channel channel1 = {};
};

/**
 * @brief Describe how a data transfer selects configuration and address-counter contexts.
 *
 * The factories clear fields that are ignored by their selected context mode. Hand-written
 * aggregate values remain supported and receive the same range validation in the encoders.
 */
struct ContextSelection
{
    ContextSource source                 = ContextSource::Explicit;
    std::uint8_t configuration_context   = 0;
    std::uint8_t address_counter_context = 0;

    /** @brief Select configuration context zero and the issuing thread's address counters. */
    static constexpr ContextSelection thread_default();

    /**
     * @brief Select explicit configuration and address-counter contexts.
     *
     * @param configuration_context: Configuration context in [0, 7], or [0, 1] for unpacker 1.
     * @param address_counter_context: Address-counter context in [0, 2].
     */
    static constexpr ContextSelection explicit_context(std::uint8_t configuration_context = 0, std::uint8_t address_counter_context = 0);

    /**
     * @brief Select and advance the configuration context counter during a transfer.
     *
     * @param address_counter_context: Address-counter context in [0, 2].
     */
    static constexpr ContextSelection counter(std::uint8_t address_counter_context = 0);
};

constexpr ContextSelection ContextSelection::thread_default()
{
    return {ContextSource::ThreadDefault, 0, 0};
}

constexpr ContextSelection ContextSelection::explicit_context(const std::uint8_t configuration_context, const std::uint8_t address_counter_context)
{
    return {ContextSource::Explicit, configuration_context, address_counter_context};
}

constexpr ContextSelection ContextSelection::counter(const std::uint8_t address_counter_context)
{
    return {ContextSource::Counter, 0, address_counter_context};
}

/**
 * @brief Describe one UNPACR data-transfer operation.
 *
 * Data format, dimensions, L1 address, compression, and destination selection remain in
 * unpacker configuration and address-counter state. Source handoff has no implicit default;
 * select Keep or FlipAndSetDataValid at every call site.
 *
 * @note For unpacker 1 with an explicit or counter-selected context, the configured context
 *       offset is added to the selected ID; ensure the resolved configuration context,
 *       including configured offsets, remains below two.
 */
struct DataTransfer
{
    Engine engine;
    AddressCounterPostIncrements increments = {};
    ContextSelection context                = ContextSelection::explicit_context();
    SourceHandoff handoff                   = SourceHandoff::Unset;
    DatumOverride datum_override            = DatumOverride::None;
    SearchMode search                       = SearchMode::DatumRange;
    AccumulationAction accumulation         = AccumulationAction::Flush;
};

/** @brief Describe a canonical UNPACR configuration-context-counter increment. */
struct ContextCounterIncrement
{
    Engine engine;
};

/** @brief Describe a canonical UNPACR compressed-row-start-cache flush. */
struct RowStartCacheFlush
{
    Engine engine;
    CacheScope scope = CacheScope::CurrentThread;
};

namespace detail
{
constexpr bool is_valid(const Engine engine)
{
    return engine == Engine::Unpacker0 || engine == Engine::Unpacker1;
}

constexpr bool is_valid(const ContextSource source)
{
    return source == ContextSource::ThreadDefault || source == ContextSource::Explicit || source == ContextSource::Counter;
}

constexpr bool is_valid(const SourceHandoff handoff)
{
    return handoff == SourceHandoff::Keep || handoff == SourceHandoff::FlipAndSetDataValid;
}

constexpr bool is_valid(const DatumOverride override)
{
    return override == DatumOverride::None || override == DatumOverride::Zero;
}

constexpr bool is_valid(const SearchMode mode)
{
    return mode == SearchMode::DatumRange || mode == SearchMode::Row;
}

constexpr bool is_valid(const AccumulationAction action)
{
    return action == AccumulationAction::Continue || action == AccumulationAction::Flush;
}

constexpr bool is_valid(const CacheScope scope)
{
    return scope == CacheScope::CurrentThread || scope == CacheScope::AllEntries;
}

constexpr bool is_valid(const AddressCounterPostIncrements increments)
{
    return ckernel::is_valid(increments.channel0.y, 2) && ckernel::is_valid(increments.channel0.z, 2) && ckernel::is_valid(increments.channel1.y, 2) &&
           ckernel::is_valid(increments.channel1.z, 2);
}

constexpr bool is_valid(const ContextSelection context)
{
    return is_valid(context.source) && ckernel::is_valid(context.configuration_context, 3) && context.address_counter_context <= 2;
}

constexpr bool has_valid_explicit_context(const DataTransfer operation)
{
    return operation.engine != Engine::Unpacker1 || operation.context.source != ContextSource::Explicit || operation.context.configuration_context < 2;
}

constexpr bool is_valid(const DataTransfer operation)
{
    return is_valid(operation.engine) && is_valid(operation.increments) && is_valid(operation.context) && is_valid(operation.handoff) &&
           is_valid(operation.datum_override) && is_valid(operation.search) && is_valid(operation.accumulation) && has_valid_explicit_context(operation);
}

constexpr bool is_valid(const ContextCounterIncrement operation)
{
    return is_valid(operation.engine);
}

constexpr bool is_valid(const RowStartCacheFlush operation)
{
    return is_valid(operation.engine) && is_valid(operation.scope);
}

#ifdef ENABLE_LLK_ASSERT
inline __attribute__((always_inline)) void assert_valid(const DataTransfer operation)
{
    LLK_ASSERT(is_valid(operation.engine), "UNPACR engine must be Unpacker0 or Unpacker1");
    LLK_ASSERT(ckernel::is_valid(operation.increments.channel0.y, 2), "UNPACR channel 0 Y increment must be in [0, 3]");
    LLK_ASSERT(ckernel::is_valid(operation.increments.channel0.z, 2), "UNPACR channel 0 Z increment must be in [0, 3]");
    LLK_ASSERT(ckernel::is_valid(operation.increments.channel1.y, 2), "UNPACR channel 1 Y increment must be in [0, 3]");
    LLK_ASSERT(ckernel::is_valid(operation.increments.channel1.z, 2), "UNPACR channel 1 Z increment must be in [0, 3]");
    LLK_ASSERT(is_valid(operation.context.source), "UNPACR context source must be ThreadDefault, Explicit, or Counter");
    LLK_ASSERT(ckernel::is_valid(operation.context.configuration_context, 3), "UNPACR configuration context must be in [0, 7]");
    LLK_ASSERT(operation.context.address_counter_context <= 2u, "UNPACR address-counter context must be in [0, 2]");
    LLK_ASSERT(is_valid(operation.handoff), "UNPACR source handoff must be Keep or FlipAndSetDataValid");
    LLK_ASSERT(is_valid(operation.datum_override), "UNPACR datum override must be None or Zero");
    LLK_ASSERT(is_valid(operation.search), "UNPACR search mode must be DatumRange or Row");
    LLK_ASSERT(is_valid(operation.accumulation), "UNPACR accumulation action must be Continue or Flush");
    LLK_ASSERT(has_valid_explicit_context(operation), "UNPACR unpacker-1 explicit configuration context must be in [0, 1]");
}

inline __attribute__((always_inline)) void assert_valid(const ContextCounterIncrement operation)
{
    LLK_ASSERT(is_valid(operation.engine), "UNPACR context-counter engine must be Unpacker0 or Unpacker1");
}

inline __attribute__((always_inline)) void assert_valid(const RowStartCacheFlush operation)
{
    LLK_ASSERT(is_valid(operation.engine), "UNPACR row-start-cache engine must be Unpacker0 or Unpacker1");
    LLK_ASSERT(is_valid(operation.scope), "UNPACR row-start-cache scope must be CurrentThread or AllEntries");
}
#endif

constexpr std::uint32_t value(const Engine engine)
{
    return static_cast<std::uint32_t>(engine);
}

constexpr std::uint32_t value(const DatumOverride override)
{
    return static_cast<std::uint32_t>(override);
}

constexpr std::uint32_t value(const SearchMode mode)
{
    return static_cast<std::uint32_t>(mode);
}

constexpr std::uint32_t value(const AccumulationAction action)
{
    return static_cast<std::uint32_t>(action);
}

constexpr std::uint32_t value(const CacheScope scope)
{
    return static_cast<std::uint32_t>(scope);
}

constexpr std::uint32_t address_mode(const AddressCounterPostIncrements increments)
{
    return (static_cast<std::uint32_t>(increments.channel1.y) << 6) | (static_cast<std::uint32_t>(increments.channel1.z) << 4) |
           (static_cast<std::uint32_t>(increments.channel0.y) << 2) | static_cast<std::uint32_t>(increments.channel0.z);
}

constexpr std::uint32_t configuration_context(const ContextSelection context)
{
    return context.source == ContextSource::Explicit ? context.configuration_context : 0;
}

constexpr std::uint32_t address_counter_context(const ContextSelection context)
{
    return context.source == ContextSource::ThreadDefault ? 0 : context.address_counter_context;
}

constexpr std::uint32_t multi_context_mode(const ContextSelection context)
{
    return context.source == ContextSource::ThreadDefault ? 0 : 1;
}

constexpr std::uint32_t use_context_counter(const ContextSelection context)
{
    return context.source == ContextSource::Counter ? 1 : 0;
}

constexpr std::uint32_t set_data_valid(const SourceHandoff handoff)
{
    return handoff == SourceHandoff::FlipAndSetDataValid ? 1 : 0;
}

constexpr std::uint32_t get_operation(const DataTransfer operation)
{
    return TT_OP_UNPACR(
        value(operation.engine),
        address_mode(operation.increments),
        0,
        configuration_context(operation.context),
        address_counter_context(operation.context),
        multi_context_mode(operation.context),
        set_data_valid(operation.handoff),
        0,
        value(operation.datum_override),
        use_context_counter(operation.context),
        value(operation.search),
        0,
        value(operation.accumulation));
}

constexpr std::uint32_t get_operation(const ContextCounterIncrement operation)
{
    return TT_OP_UNPACR(value(operation.engine), 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

constexpr std::uint32_t get_operation(const RowStartCacheFlush operation)
{
    return TT_OP_UNPACR(value(operation.engine), 0, 0, 0, 0, value(operation.scope), 0, 0, 0, 0, 0, 1, 0);
}
} // namespace detail

/** @brief Return whether a data-transfer descriptor can be encoded without truncation. */
constexpr bool is_valid(const DataTransfer operation)
{
    return detail::is_valid(operation);
}

/** @brief Return whether a context-counter increment descriptor is valid. */
constexpr bool is_valid(const ContextCounterIncrement operation)
{
    return detail::is_valid(operation);
}

/** @brief Return whether a row-start-cache flush descriptor is valid. */
constexpr bool is_valid(const RowStartCacheFlush operation)
{
    return detail::is_valid(operation);
}

/**
 * @brief Encode a compile-time UNPACR data transfer without issuing it.
 *
 * @tparam Operation: Complete data-transfer description.
 * @note Use the result where another expander accepts an encoded operation, such as a MOP
 *       or replay configuration.
 */
template <DataTransfer Operation>
constexpr std::uint32_t get_operation()
{
    static_assert(is_valid(Operation), "invalid UNPACR data-transfer descriptor");

    return detail::get_operation(Operation);
}

/**
 * @brief Encode a runtime-selected UNPACR data transfer without issuing it.
 *
 * @param operation: Complete data-transfer description.
 * @note Use the result where another expander accepts an encoded operation, such as a MOP
 *       or replay configuration.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline constexpr __attribute__((always_inline)) std::uint32_t get_operation(const DataTransfer operation)
{
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(operation);
    }
#endif

    return detail::get_operation(operation);
}

/**
 * @brief Issue a compile-time UNPACR data transfer as one immediate instruction.
 *
 * @tparam Operation: Complete data-transfer description.
 */
template <DataTransfer Operation>
inline __attribute__((always_inline)) void run()
{
    constexpr std::uint32_t operation = get_operation<Operation>();
    INSTRUCTION_WORD(operation);
}

/**
 * @brief Encode and issue a runtime-selected UNPACR data transfer.
 *
 * @param operation: Complete data-transfer description.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline __attribute__((always_inline)) void run(const DataTransfer operation)
{
    ckernel::instrn_buffer[0] = get_operation(operation);
}

/**
 * @brief Encode a compile-time canonical UNPACR context-counter increment without issuing it.
 *
 * @tparam Operation: Unpacker whose per-thread configuration context counter advances.
 */
template <ContextCounterIncrement Operation>
constexpr std::uint32_t get_operation()
{
    static_assert(is_valid(Operation), "invalid UNPACR context-counter increment descriptor");
    return detail::get_operation(Operation);
}

/**
 * @brief Encode a runtime-selected canonical UNPACR context-counter increment without issuing it.
 *
 * @param operation: Unpacker whose per-thread configuration context counter advances.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline constexpr __attribute__((always_inline)) std::uint32_t get_operation(const ContextCounterIncrement operation)
{
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(operation);
    }
#endif
    return detail::get_operation(operation);
}

/**
 * @brief Issue a compile-time canonical UNPACR context-counter increment.
 *
 * @tparam Operation: Unpacker whose per-thread configuration context counter advances.
 */
template <ContextCounterIncrement Operation>
inline __attribute__((always_inline)) void run()
{
    constexpr std::uint32_t operation = get_operation<Operation>();
    INSTRUCTION_WORD(operation);
}

/**
 * @brief Encode and issue a runtime-selected UNPACR context-counter increment.
 *
 * @param operation: Unpacker whose per-thread configuration context counter advances.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline __attribute__((always_inline)) void run(const ContextCounterIncrement operation)
{
    ckernel::instrn_buffer[0] = get_operation(operation);
}

/**
 * @brief Encode a compile-time canonical UNPACR row-start-cache flush without issuing it.
 *
 * @tparam Operation: Unpacker and cache-entry scope to flush.
 */
template <RowStartCacheFlush Operation>
constexpr std::uint32_t get_operation()
{
    static_assert(is_valid(Operation), "invalid UNPACR row-start-cache flush descriptor");
    return detail::get_operation(Operation);
}

/**
 * @brief Encode a runtime-selected canonical UNPACR row-start-cache flush without issuing it.
 *
 * @param operation: Unpacker and cache-entry scope to flush.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline constexpr __attribute__((always_inline)) std::uint32_t get_operation(const RowStartCacheFlush operation)
{
#ifdef ENABLE_LLK_ASSERT
    if (!__builtin_is_constant_evaluated())
    {
        detail::assert_valid(operation);
    }
#endif
    return detail::get_operation(operation);
}

/**
 * @brief Issue a compile-time canonical UNPACR row-start-cache flush.
 *
 * @tparam Operation: Unpacker and cache-entry scope to flush.
 */
template <RowStartCacheFlush Operation>
inline __attribute__((always_inline)) void run()
{
    constexpr std::uint32_t operation = get_operation<Operation>();
    INSTRUCTION_WORD(operation);
}

/**
 * @brief Encode and issue a runtime-selected UNPACR row-start-cache flush.
 *
 * @param operation: Unpacker and cache-entry scope to flush.
 * @note Enable LLK assertions to diagnose invalid fields; disabled assertions add no runtime check.
 */
inline __attribute__((always_inline)) void run(const RowStartCacheFlush operation)
{
    ckernel::instrn_buffer[0] = get_operation(operation);
}

} // namespace hal::unpack
