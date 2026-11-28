// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "utils/helpers.h"
#include "utils/operation.h"
#include "utils/validate.h"

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
enum class SrcHandoff : std::uint8_t
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

/** @brief Select the value an UNPACR_NOP source clear writes to every datum of the bank. */
enum class ClearValue : std::uint8_t
{
    Zero             = 0,
    NegativeInfinity = 1, ///< SrcA only.
    One              = 2  ///< Written in the bank's implied format (INT8, FP16, TF32 or BF16 encoding).
};

/** @brief Select which source bank an UNPACR_NOP clear waits for before writing the unpacker's bank. */
enum class ClearWait : bool
{
    MatrixBank   = false, ///< The bank the Matrix Unit reads; not with FlipAndSetDataValid.
    UnpackerBank = true   ///< The unpacker's own bank, as UNPACR waits.
};

/** @brief Select which compressed-row-start cache entries an UNPACR flush clears. */
enum class CacheScope : bool
{
    CurrentThread = false,
    AllEntries    = true
};

/** @brief Describe the per-channel address-counter increments applied after a transfer. */
class AddressCounterPostIncrements
{
public:
    /** @brief Describe the two-bit Y and Z increments supported by one UNPACR channel. */
    class Channel
    {
    public:
        std::uint8_t y = 0; ///< Y increment in [0, 3].
        std::uint8_t z = 0; ///< Z increment in [0, 3].
    };

    Channel channel0 = {}; ///< Input-side (L1) increments.
    Channel channel1 = {}; ///< Output-side (SrcA, SrcB or Dst) increments.
};

/**
 * @brief Describe how a data transfer selects configuration and address-counter contexts.
 *
 * The factories clear fields that are ignored by their selected context mode. Hand-written
 * aggregate values remain supported and receive the same range validation in the encoders.
 */
class Context
{
public:
    ContextSource source                 = ContextSource::Explicit; ///< How the configuration context is resolved.
    std::uint8_t configuration_context   = 0;                       ///< Explicit configuration context; ignored otherwise.
    std::uint8_t address_counter_context = 0;                       ///< Address-counter context; ignored for ThreadDefault.

    /** @brief Select configuration context zero and the issuing thread's address counters. */
    static constexpr Context thread_default();

    /**
     * @brief Select explicit configuration and address-counter contexts.
     *
     * @param configuration_context: Configuration context in [0, 7], or [0, 1] for unpacker 1.
     * @param address_counter_context: Address-counter context in [0, 2].
     */
    static constexpr Context explicit_context(std::uint8_t configuration_context = 0, std::uint8_t address_counter_context = 0);

    /**
     * @brief Select and advance the configuration context counter during a transfer.
     *
     * @param address_counter_context: Address-counter context in [0, 2].
     */
    static constexpr Context counter(std::uint8_t address_counter_context = 0);
};

constexpr Context Context::thread_default()
{
    return {ContextSource::ThreadDefault, 0, 0};
}

constexpr Context Context::explicit_context(const std::uint8_t configuration_context, const std::uint8_t address_counter_context)
{
    return {ContextSource::Explicit, configuration_context, address_counter_context};
}

constexpr Context Context::counter(const std::uint8_t address_counter_context)
{
    return {ContextSource::Counter, 0, address_counter_context};
}

/**
 * @brief Describe the options of one UNPACR data transfer.
 *
 * Data format, dimensions, L1 address, compression, and destination selection remain in
 * unpacker configuration and address-counter state. Source handoff has no implicit default;
 * select Keep or FlipAndSetDataValid at every call site.
 *
 * @note For unpacker 1 with an explicit or counter-selected context, the configured context
 *       offset is added to the selected ID; ensure the resolved configuration context,
 *       including configured offsets, remains below two.
 */
class Transfer
{
public:
    AddressCounterPostIncrements increments = {};                          ///< Address-counter increments applied after the transfer.
    Context context                         = Context::explicit_context(); ///< Configuration and address-counter context selection.
    SrcHandoff handoff                      = SrcHandoff::Unset;           ///< Source-bank handoff; must be set at every call site.
    DatumOverride datum_override            = DatumOverride::None;         ///< Whether datums keep their values or become zero.
    SearchMode search                       = SearchMode::DatumRange;      ///< Datum-range or compressed-row-start search.
    AccumulationAction accumulation         = AccumulationAction::Flush;   ///< Whether the transfer continues or flushes the accumulation task.
};

/**
 * @brief Describe the options of one UNPACR_NOP clear of the unpacker's current source bank.
 *
 * The clear is queued behind earlier UNPACRs on the same unpacker, waits for the bank selected by
 * wait, and writes all 64 rows of the unpacker's bank regardless of the address-counter range.
 * Source handoff has no implicit default; select Keep or FlipAndSetDataValid at every call site.
 *
 * @note With FlipAndSetDataValid on unpacker 0, ensure SRCA_SET_Base is zero.
 * @note ClearValue::One on unpacker 0 leaves the hazard of @ref experimental::BothBanksClear armed
 *       until unpacker 0 next clears to Zero or NegativeInfinity.
 */
class Clear
{
public:
    ClearValue value   = ClearValue::Zero;        ///< Value written to every datum of the bank.
    SrcHandoff handoff = SrcHandoff::Unset;       ///< Source-bank handoff after the clear.
    ClearWait wait     = ClearWait::UnpackerBank; ///< Bank the clear waits for before writing.
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

constexpr bool is_valid(const SrcHandoff handoff)
{
    return handoff == SrcHandoff::Keep || handoff == SrcHandoff::FlipAndSetDataValid;
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

constexpr bool is_valid(const ClearValue value)
{
    return value == ClearValue::Zero || value == ClearValue::NegativeInfinity || value == ClearValue::One;
}

constexpr bool is_valid(const ClearWait wait)
{
    return wait == ClearWait::MatrixBank || wait == ClearWait::UnpackerBank;
}

// FlipAndSetDataValid after waiting for the Matrix Unit's bank hands over a bank the unpacker may not own.
constexpr bool has_contractual_handoff(const ClearWait wait, const SrcHandoff handoff)
{
    return wait == ClearWait::UnpackerBank || handoff != SrcHandoff::FlipAndSetDataValid;
}

constexpr bool is_valid(const AddressCounterPostIncrements increments)
{
    return hal::detail::is_valid(increments.channel0.y, 2) && hal::detail::is_valid(increments.channel0.z, 2) &&
           hal::detail::is_valid(increments.channel1.y, 2) && hal::detail::is_valid(increments.channel1.z, 2);
}

constexpr bool is_valid(const Context context)
{
    return is_valid(context.source) && hal::detail::is_valid(context.configuration_context, 3) && context.address_counter_context <= 2;
}

constexpr bool is_valid(const Transfer options)
{
    return is_valid(options.increments) && is_valid(options.context) && is_valid(options.handoff) && is_valid(options.datum_override) &&
           is_valid(options.search) && is_valid(options.accumulation);
}

constexpr bool is_valid(const Clear options)
{
    return is_valid(options.value) && is_valid(options.handoff) && is_valid(options.wait) && has_contractual_handoff(options.wait, options.handoff);
}

constexpr bool supports(const Engine engine, const Transfer options)
{
    return engine != Engine::Unpacker1 || options.context.source != ContextSource::Explicit || options.context.configuration_context < 2;
}

constexpr bool supports(const Engine engine, const ClearValue value)
{
    return engine != Engine::Unpacker1 || value != ClearValue::NegativeInfinity;
}

constexpr bool supports(const Engine engine, const Clear options)
{
    return supports(engine, options.value);
}

inline constexpr __attribute__((always_inline)) void validate(const Transfer options)
{
    hal::detail::require(hal::detail::is_valid(options.increments.channel0.y, 2), "UNPACR channel 0 Y increment must be in [0, 3]");
    hal::detail::require(hal::detail::is_valid(options.increments.channel0.z, 2), "UNPACR channel 0 Z increment must be in [0, 3]");
    hal::detail::require(hal::detail::is_valid(options.increments.channel1.y, 2), "UNPACR channel 1 Y increment must be in [0, 3]");
    hal::detail::require(hal::detail::is_valid(options.increments.channel1.z, 2), "UNPACR channel 1 Z increment must be in [0, 3]");
    hal::detail::require(is_valid(options.context.source), "UNPACR context source must be ThreadDefault, Explicit, or Counter");
    hal::detail::require(hal::detail::is_valid(options.context.configuration_context, 3), "UNPACR configuration context must be in [0, 7]");
    hal::detail::require(options.context.address_counter_context <= 2u, "UNPACR address-counter context must be in [0, 2]");
    hal::detail::require(is_valid(options.handoff), "UNPACR source handoff must be Keep or FlipAndSetDataValid");
    hal::detail::require(is_valid(options.datum_override), "UNPACR datum override must be None or Zero");
    hal::detail::require(is_valid(options.search), "UNPACR search mode must be DatumRange or Row");
    hal::detail::require(is_valid(options.accumulation), "UNPACR accumulation action must be Continue or Flush");
}

inline constexpr __attribute__((always_inline)) void validate(const Clear options)
{
    hal::detail::require(is_valid(options.value), "UNPACR_NOP source-clear value must be Zero, NegativeInfinity, or One");
    hal::detail::require(is_valid(options.handoff), "UNPACR_NOP source handoff must be Keep or FlipAndSetDataValid");
    hal::detail::require(is_valid(options.wait), "UNPACR_NOP source-clear wait must be MatrixBank or UnpackerBank");
    hal::detail::require(has_contractual_handoff(options.wait, options.handoff), "UNPACR_NOP FlipAndSetDataValid requires ClearWait::UnpackerBank");
}

inline constexpr __attribute__((always_inline)) void validate(const Engine engine)
{
    hal::detail::require(is_valid(engine), "UNPACR engine must be Unpacker0 or Unpacker1");
}

inline constexpr __attribute__((always_inline)) void validate(const Engine engine, const Transfer options)
{
    hal::detail::require(supports(engine, options), "UNPACR unpacker-1 explicit configuration context must be in [0, 1]");
}

inline constexpr __attribute__((always_inline)) void validate(const Engine engine, const Clear options)
{
    hal::detail::require(supports(engine, options), "UNPACR_NOP cannot clear SrcB to negative infinity");
}

constexpr std::uint32_t value(const Engine engine)
{
    return hal::to_underlying(engine);
}

constexpr std::uint32_t value(const DatumOverride override)
{
    return hal::to_underlying(override);
}

constexpr std::uint32_t value(const SearchMode mode)
{
    return hal::to_underlying(mode);
}

constexpr std::uint32_t value(const AccumulationAction action)
{
    return hal::to_underlying(action);
}

constexpr std::uint32_t value(const CacheScope scope)
{
    return hal::to_underlying(scope);
}

constexpr std::uint32_t value(const ClearValue clear_value)
{
    return hal::to_underlying(clear_value);
}

constexpr std::uint32_t value(const ClearWait wait)
{
    return hal::to_underlying(wait);
}

constexpr std::uint32_t address_mode(const AddressCounterPostIncrements increments)
{
    return (static_cast<std::uint32_t>(increments.channel1.y) << 6) | (static_cast<std::uint32_t>(increments.channel1.z) << 4) |
           (static_cast<std::uint32_t>(increments.channel0.y) << 2) | static_cast<std::uint32_t>(increments.channel0.z);
}

constexpr std::uint32_t configuration_context(const Context context)
{
    return context.source == ContextSource::Explicit ? context.configuration_context : 0;
}

constexpr std::uint32_t address_counter_context(const Context context)
{
    return context.source == ContextSource::ThreadDefault ? 0 : context.address_counter_context;
}

constexpr std::uint32_t multi_context_mode(const Context context)
{
    return context.source == ContextSource::ThreadDefault ? 0 : 1;
}

constexpr std::uint32_t use_context_counter(const Context context)
{
    return context.source == ContextSource::Counter ? 1 : 0;
}

constexpr std::uint32_t set_data_valid(const SrcHandoff handoff)
{
    return handoff == SrcHandoff::FlipAndSetDataValid ? 1 : 0;
}

/**
 * @brief UNPACR_NOP Unpack_Pop modes.
 *
 * The ISA marks Delay UnsupportedFunctionality; production still issues it as the unpack-template
 * filler (ckernel_unpack_template::lzA), and @ref hal::nop::unpacker exposes it the same way.
 */
enum class NopMode : std::uint32_t
{
    ClearSrc = 1,
    Delay    = 2
};

constexpr std::uint32_t transfer_operation(const Engine engine, const Transfer options)
{
    return TT_OP_UNPACR(
        value(engine),
        address_mode(options.increments),
        0 /*CfgContextCntInc*/,
        configuration_context(options.context),
        address_counter_context(options.context),
        multi_context_mode(options.context),
        set_data_valid(options.handoff),
        0 /*srcb_bcast*/,
        value(options.datum_override),
        use_context_counter(options.context),
        value(options.search),
        0 /*SearchCacheFlush*/,
        value(options.accumulation));
}

/** @brief UNPACR_NOP Bank_Clr_Ctrl values. */
enum class ClearBanks : std::uint32_t
{
    Current = 0,
    Both    = 1
};

constexpr std::uint32_t clear_operation(
    const Engine engine, const ClearValue clear_value, const SrcHandoff handoff, const ClearWait wait, const ClearBanks banks)
{
    return TT_OP_UNPACR_NOP(
        value(engine),
        0 /*Stream_Id*/,
        0 /*Msg_Clr_Cnt*/,
        set_data_valid(handoff),
        0 /*Clr_to1_fmt_Ctrl*/,
        value(wait),
        hal::to_underlying(banks),
        value(clear_value),
        hal::to_underlying(NopMode::ClearSrc));
}

constexpr std::uint32_t clear_operation(const Engine engine, const Clear options)
{
    return clear_operation(engine, options.value, options.handoff, options.wait, ClearBanks::Current);
}

constexpr std::uint32_t context_counter_increment_operation(const Engine engine)
{
    return TT_OP_UNPACR(
        value(engine),
        0 /*AddrMode*/,
        1 /*CfgContextCntInc*/,
        0 /*CfgContextId*/,
        0 /*AddrCntContextId*/,
        0 /*OvrdThreadId*/,
        0 /*SetDatValid*/,
        0 /*srcb_bcast*/,
        0 /*ZeroWrite2*/,
        0 /*AutoIncContextID*/,
        0 /*RowSearch*/,
        0 /*SearchCacheFlush*/,
        0 /*Last*/);
}

constexpr std::uint32_t row_start_cache_flush_operation(const Engine engine, const CacheScope scope)
{
    return TT_OP_UNPACR(
        value(engine),
        0 /*AddrMode*/,
        0 /*CfgContextCntInc*/,
        0 /*CfgContextId*/,
        0 /*AddrCntContextId*/,
        value(scope) /*OvrdThreadId*/,
        0 /*SetDatValid*/,
        0 /*srcb_bcast*/,
        0 /*ZeroWrite2*/,
        0 /*AutoIncContextID*/,
        0 /*RowSearch*/,
        1 /*SearchCacheFlush*/,
        0 /*Last*/);
}

} // namespace detail

/** @brief Return whether data-transfer options can be encoded without truncation on some unpacker. */
constexpr bool is_valid(const Transfer options)
{
    return detail::is_valid(options);
}

/** @brief Return whether source-clear options can be encoded on some unpacker. */
constexpr bool is_valid(const Clear options)
{
    return detail::is_valid(options);
}

/**
 * @brief Issue or encode the operations of one Blackhole unpacker.
 *
 * Unpacker 0 writes SrcA, or Dst when configured to; unpacker 1 writes SrcB. Options that do not
 * depend on the unpacker are checked at compile time in the template forms; checks that depend on
 * the unpacker fail constant evaluation, or LLK_ASSERT at runtime.
 *
 * Use @ref Unpacker0 and @ref Unpacker1; construct an Unpacker only when the engine is chosen by a value.
 *
 * @code
 * Unpacker0.transfer<Transfer {.increments = {.channel0 = {.z = 1}}, .handoff = SrcHandoff::FlipAndSetDataValid}>();
 * Unpacker1.clear<Clear {.handoff = SrcHandoff::FlipAndSetDataValid}>();
 *
 * constexpr std::uint32_t unpack_face = Unpacker0.operation().transfer<Transfer {.handoff = SrcHandoff::Keep}>();
 * @endcode
 */
class Unpacker
{
public:
    /** @brief Encode this unpacker's operations without issuing them, for MOP or replay use. */
    class Operations
    {
    public:
        Engine engine; ///< Unpacker that executes the operations.

        /**
         * @brief Encode a data transfer.
         *
         * @tparam Options: Data-transfer options.
         */
        template <Transfer Options>
        [[nodiscard]] constexpr hal::Operation transfer() const;

        /**
         * @brief Encode a runtime-selected data transfer.
         *
         * @param options: Data-transfer options.
         * @note Invalid options fail constant evaluation; at runtime, enable LLK assertions to diagnose them.
         */
        [[nodiscard]] constexpr hal::Operation transfer(Transfer options) const;

        /**
         * @brief Encode a source-bank clear.
         *
         * @tparam Options: Clear value, source handoff and wait.
         */
        template <Clear Options>
        [[nodiscard]] constexpr hal::Operation clear() const;

        /**
         * @brief Encode a runtime-selected source-bank clear.
         *
         * @param options: Clear value, source handoff and wait.
         * @note Invalid options fail constant evaluation; at runtime, enable LLK assertions to diagnose them.
         */
        [[nodiscard]] constexpr hal::Operation clear(Clear options) const;

        /** @brief Encode an increment of the issuing thread's configuration context counter. */
        [[nodiscard]] constexpr hal::Operation increment_context_counter() const;

        /**
         * @brief Encode a flush of the compressed-row-start cache.
         *
         * @tparam Scope: Cache entries to clear.
         */
        template <CacheScope Scope>
        [[nodiscard]] constexpr hal::Operation flush_row_start_cache() const;

        /**
         * @brief Encode a runtime-selected flush of the compressed-row-start cache.
         *
         * @param scope: Cache entries to clear.
         */
        [[nodiscard]] constexpr hal::Operation flush_row_start_cache(CacheScope scope) const;
    };

    Engine engine; ///< Unpacker that executes the operations.

    /** @brief Return an encoding view of this unpacker. */
    [[nodiscard]] constexpr Operations operation() const
    {
        return {engine};
    }

    /**
     * @brief Issue a data transfer.
     *
     * @tparam Options: Data-transfer options.
     */
    template <Transfer Options>
    inline __attribute__((always_inline)) void transfer() const
    {
        hal::run(operation().transfer<Options>());
    }

    /**
     * @brief Issue a runtime-selected data transfer.
     *
     * @param options: Data-transfer options.
     * @note Enable LLK assertions to diagnose invalid options; disabled assertions add no runtime check.
     */
    inline __attribute__((always_inline)) void transfer(const Transfer options) const
    {
        hal::run(operation().transfer(options));
    }

    /**
     * @brief Issue a source-bank clear.
     *
     * @tparam Options: Clear value, source handoff and wait.
     */
    template <Clear Options>
    inline __attribute__((always_inline)) void clear() const
    {
        hal::run(operation().clear<Options>());
    }

    /**
     * @brief Issue a runtime-selected source-bank clear.
     *
     * @param options: Clear value, source handoff and wait.
     * @note Enable LLK assertions to diagnose invalid options; disabled assertions add no runtime check.
     */
    inline __attribute__((always_inline)) void clear(const Clear options) const
    {
        hal::run(operation().clear(options));
    }

    /** @brief Increment the issuing thread's configuration context counter. */
    inline __attribute__((always_inline)) void increment_context_counter() const
    {
        hal::run(operation().increment_context_counter());
    }

    /**
     * @brief Flush the compressed-row-start cache.
     *
     * @tparam Scope: Cache entries to clear.
     */
    template <CacheScope Scope>
    inline __attribute__((always_inline)) void flush_row_start_cache() const
    {
        hal::run(operation().flush_row_start_cache<Scope>());
    }

    /**
     * @brief Flush the compressed-row-start cache with a runtime-selected scope.
     *
     * @param scope: Cache entries to clear.
     */
    inline __attribute__((always_inline)) void flush_row_start_cache(const CacheScope scope) const
    {
        hal::run(operation().flush_row_start_cache(scope));
    }
};

/** @brief Unpacker 0, which writes SrcA, or Dst when configured to. */
inline constexpr Unpacker Unpacker0 {Engine::Unpacker0};

/** @brief Unpacker 1, which writes SrcB. */
inline constexpr Unpacker Unpacker1 {Engine::Unpacker1};

template <Transfer Options>
constexpr hal::Operation Unpacker::Operations::transfer() const
{
    static_assert(is_valid(Options), "invalid UNPACR data-transfer options");
    detail::validate(engine);
    detail::validate(engine, Options);
    return hal::detail::make_operation(detail::transfer_operation(engine, Options));
}

constexpr hal::Operation Unpacker::Operations::transfer(const Transfer options) const
{
    detail::validate(engine);
    detail::validate(options);
    detail::validate(engine, options);
    return hal::detail::make_operation(detail::transfer_operation(engine, options));
}

template <Clear Options>
constexpr hal::Operation Unpacker::Operations::clear() const
{
    static_assert(is_valid(Options), "invalid UNPACR_NOP source-clear options");
    detail::validate(engine);
    detail::validate(engine, Options);
    return hal::detail::make_operation(detail::clear_operation(engine, Options));
}

constexpr hal::Operation Unpacker::Operations::clear(const Clear options) const
{
    detail::validate(engine);
    detail::validate(options);
    detail::validate(engine, options);
    return hal::detail::make_operation(detail::clear_operation(engine, options));
}

constexpr hal::Operation Unpacker::Operations::increment_context_counter() const
{
    detail::validate(engine);
    return hal::detail::make_operation(detail::context_counter_increment_operation(engine));
}

template <CacheScope Scope>
constexpr hal::Operation Unpacker::Operations::flush_row_start_cache() const
{
    static_assert(detail::is_valid(Scope), "UNPACR row-start-cache scope must be CurrentThread or AllEntries");
    detail::validate(engine);
    return hal::detail::make_operation(detail::row_start_cache_flush_operation(engine, Scope));
}

constexpr hal::Operation Unpacker::Operations::flush_row_start_cache(const CacheScope scope) const
{
    detail::validate(engine);
    hal::detail::require(detail::is_valid(scope), "UNPACR row-start-cache scope must be CurrentThread or AllEntries");
    return hal::detail::make_operation(detail::row_start_cache_flush_operation(engine, scope));
}

/**
 * @brief Operations the Blackhole ISA marks UnsupportedFunctionality but that experimental kernels rely on.
 *
 * Name the namespace at each call site; these operations have documented hardware behavior but no
 * contractual guarantee.
 */
namespace experimental
{
/**
 * @brief Describe an UNPACR_NOP clear of both banks of the unpacker's source register.
 *
 * Once the Matrix Unit's current bank belongs to the unpackers, every datum of both SrcA banks (unpacker 0)
 * or both SrcB banks (unpacker 1) is set to value. Bank ownership, the unpacker's current bank, and implied
 * formats are unchanged, so there is no source handoff. The clear always waits for the Matrix Unit's bank:
 * waiting for the unpacker's own bank could clear a bank the Matrix Unit is still reading.
 *
 * @note The unpackers share the both-banks select. STALLWAIT until both unpackers are idle before the
 *       clear, and until the clearing unpacker is idle before the other unpacker's next instruction.
 *       While unpacker 1 clears both SrcB banks, concurrent unpacker 0 writes can be lost if unpacker 0's
 *       last clear used ClearValue::One, or it has not cleared since reset.
 */
class BothBanksClear
{
public:
    ClearValue value = ClearValue::Zero; ///< Value written to every datum of both banks.
};

namespace detail
{
inline constexpr __attribute__((always_inline)) std::uint32_t clear_both_banks_operation(const Engine engine, const BothBanksClear options)
{
    unpack::detail::validate(engine);
    hal::detail::require(unpack::detail::is_valid(options.value), "UNPACR_NOP both-bank clear value must be Zero, NegativeInfinity, or One");
    hal::detail::require(unpack::detail::supports(engine, options.value), "UNPACR_NOP cannot clear SrcB to negative infinity");
    return unpack::detail::clear_operation(engine, options.value, SrcHandoff::Keep, ClearWait::MatrixBank, unpack::detail::ClearBanks::Both);
}
} // namespace detail

/**
 * @brief Encode a both-bank source clear without issuing it.
 *
 * @tparam Options: Clear value.
 * @param unpacker: Unpacker whose source register is cleared.
 */
template <BothBanksClear Options>
[[nodiscard]] constexpr hal::Operation clear_both_banks_operation(const Unpacker unpacker)
{
    static_assert(unpack::detail::is_valid(Options.value), "invalid UNPACR_NOP both-bank clear options");
    return hal::detail::make_operation(detail::clear_both_banks_operation(unpacker.engine, Options));
}

/**
 * @brief Encode a runtime-selected both-bank source clear without issuing it.
 *
 * @param unpacker: Unpacker whose source register is cleared.
 * @param options: Clear value.
 * @note Invalid options fail constant evaluation; at runtime, enable LLK assertions to diagnose them.
 */
[[nodiscard]] inline constexpr __attribute__((always_inline)) hal::Operation clear_both_banks_operation(const Unpacker unpacker, const BothBanksClear options)
{
    return hal::detail::make_operation(detail::clear_both_banks_operation(unpacker.engine, options));
}

/**
 * @brief Clear both banks of the unpacker's source register.
 *
 * @tparam Options: Clear value.
 * @param unpacker: Unpacker whose source register is cleared.
 */
template <BothBanksClear Options>
inline __attribute__((always_inline)) void clear_both_banks(const Unpacker unpacker)
{
    hal::run(clear_both_banks_operation<Options>(unpacker));
}

/**
 * @brief Clear both banks of the unpacker's source register with runtime-selected options.
 *
 * @param unpacker: Unpacker whose source register is cleared.
 * @param options: Clear value.
 * @note Enable LLK assertions to diagnose invalid options; disabled assertions add no runtime check.
 */
inline __attribute__((always_inline)) void clear_both_banks(const Unpacker unpacker, const BothBanksClear options)
{
    hal::run(clear_both_banks_operation(unpacker, options));
}
} // namespace experimental

} // namespace hal::unpack
