// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

namespace hal::mop
{

/**
 * @brief Select a MOP Expander configuration template.
 */
enum class MopTemplate : std::uint8_t
{
    Template0,
    Template1
};

/**
 * @brief Select one Template 1 MOP configuration field for an individual write.
 */
enum class Template1Field : std::uint8_t
{
    OuterLoopCount                 = 0, // ISA name: OuterCount
    InnerLoopCount                 = 1, // ISA name: InnerCount
    OuterLoopStartOp               = 2, // ISA name: StartOp
    OuterLoopEndOp                 = 3, // ISA name: EndOp0
    OuterLoopSecondEndOp           = 4, // ISA name: EndOp1
    InnerLoopBodyOp                = 5, // ISA name: LoopOp
    InnerLoopAlternatingOp         = 6, // ISA name: LoopOp1
    LastOpOnFinalOuterIteration    = 7, // ISA name: Loop0Last
    LastOpOnNonfinalOuterIteration = 8, // ISA name: Loop1Last
};

namespace detail
{
struct OptionalMopInstruction
{
    bool is_set         = false;
    std::uint32_t value = 0;

    constexpr OptionalMopInstruction() = default;

    constexpr OptionalMopInstruction(std::uint32_t instruction) : is_set(true), value(instruction)
    {
    }

    constexpr std::uint32_t value_or(const std::uint32_t fallback) const
    {
        return is_set ? value : fallback;
    }
};
} // namespace detail

/**
 * @brief Hold the three optional Template 0 middle operations.
 */
struct MidOps
{
    detail::OptionalMopInstruction op_a; // ISA name: InsnA1
    detail::OptionalMopInstruction op_b; // ISA name: InsnA2
    detail::OptionalMopInstruction op_c; // ISA name: InsnA3

    /**
     * @brief Report whether at least one middle operation was supplied.
     */
    constexpr bool any_set() const
    {
        return op_a.is_set || op_b.is_set || op_c.is_set;
    }

    /**
     * @brief Report whether all three middle operations were supplied.
     */
    constexpr bool all_set() const
    {
        return op_a.is_set && op_b.is_set && op_c.is_set;
    }
};

/**
 * @brief Describe one MOP Expander configuration.
 *
 * @tparam tmpl: MOP template to configure, values = <Template0/Template1>.
 * @note A constexpr configuration is structural and can be passed directly to @ref program
 *       as a non-type template argument.
 */
template <MopTemplate tmpl>
struct MopConfig;

template <>
struct MopConfig<MopTemplate::Template0>
{
    detail::OptionalMopInstruction start_op; // ISA name: InsnA0
    MidOps mid_ops;
    detail::OptionalMopInstruction end_op;          // ISA name: InsnB
    detail::OptionalMopInstruction start_op_shadow; // ISA name: SkipA0
    detail::OptionalMopInstruction end_op_shadow;   // ISA name: SkipB
};

/**
 * @brief Describe operations placed around each Template 1 outer-loop iteration.
 */
struct OuterLoop
{
    std::uint32_t count;                          // ISA name: OuterCount
    detail::OptionalMopInstruction start_op;      // ISA name: StartOp
    detail::OptionalMopInstruction end_op;        // ISA name: EndOp0
    detail::OptionalMopInstruction second_end_op; // ISA name: EndOp1
};

/**
 * @brief Describe operations placed in each Template 1 inner-loop iteration.
 */
struct InnerLoop
{
    std::uint32_t count;                                                // ISA name: InnerCount
    detail::OptionalMopInstruction body_op;                             // ISA name: LoopOp
    detail::OptionalMopInstruction alternating_op;                      // ISA name: LoopOp1
    detail::OptionalMopInstruction last_op_on_final_outer_iteration;    // ISA name: Loop0Last
    detail::OptionalMopInstruction last_op_on_nonfinal_outer_iteration; // ISA name: Loop1Last
};

/**
 * @brief Describe a Template 1 nested-loop expansion.
 *
 * @note Omit an instruction field to use its functional-model default. Counts are written as
 *       32-bit MopCfg values; the Template 1 functional model consumes their low seven bits.
 */
template <>
struct MopConfig<MopTemplate::Template1>
{
    OuterLoop outer_loop;
    InnerLoop inner_loop;
};

namespace detail
{
struct ResolvedTemplate1Config
{
    std::uint32_t outer_count;
    std::uint32_t inner_count;
    std::uint32_t start_op;
    std::uint32_t end_op0;
    std::uint32_t end_op1;
    std::uint32_t loop_op;
    std::uint32_t loop_op1;
    std::uint32_t loop0_last;
    std::uint32_t loop1_last;
};

constexpr bool is_plain_nop(const std::uint32_t instruction)
{
    return (instruction >> 24) == (TT_OP_NOP >> 24);
}

constexpr std::uint32_t effective_template1_count(const std::uint32_t count)
{
    return count & 127u;
}

constexpr ResolvedTemplate1Config resolve(const MopConfig<MopTemplate::Template1> &config)
{
    const std::uint32_t start_op        = config.outer_loop.start_op.value_or(TT_OP_NOP);
    const std::uint32_t end_op0         = config.outer_loop.end_op.value_or(TT_OP_NOP);
    const std::uint32_t end_op1         = is_plain_nop(end_op0) ? TT_OP_NOP : config.outer_loop.second_end_op.value_or(TT_OP_NOP);
    const std::uint32_t loop_op         = config.inner_loop.body_op.value_or(TT_OP_NOP);
    const std::uint32_t loop_op1        = config.inner_loop.alternating_op.value_or(TT_OP_NOP);
    const std::uint32_t default_last_op = is_plain_nop(loop_op1) ? loop_op : loop_op1;
    const std::uint32_t loop0_last      = config.inner_loop.last_op_on_final_outer_iteration.value_or(default_last_op);
    const std::uint32_t loop1_last      = config.inner_loop.last_op_on_nonfinal_outer_iteration.value_or(default_last_op);

    return {
        config.outer_loop.count,
        config.inner_loop.count,
        start_op,
        end_op0,
        end_op1,
        loop_op,
        loop_op1,
        loop0_last,
        loop1_last,
    };
}

constexpr bool triggers_outer_count_bug(const ResolvedTemplate1Config &config)
{
    return effective_template1_count(config.outer_count) == 1 && is_plain_nop(config.start_op) && effective_template1_count(config.inner_count) == 0 &&
           !is_plain_nop(config.end_op0);
}

inline __attribute__((always_inline)) void program(const ResolvedTemplate1Config &config)
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    ckernel::mop_sync();

    mop_cfg[0] = config.outer_count;
    mop_cfg[1] = config.inner_count;
    mop_cfg[2] = config.start_op;
    mop_cfg[3] = config.end_op0;
    mop_cfg[4] = config.end_op1;
    mop_cfg[5] = config.loop_op;
    mop_cfg[6] = config.loop_op1;
    mop_cfg[7] = config.loop0_last;
    mop_cfg[8] = config.loop1_last;
}
} // namespace detail

/**
 * @brief Validate and program this thread's compile-time Template 0 MOP configuration.
 *
 * @tparam Config: Structural Template 0 configuration value.
 * @note Call only after the preceding MOP expansion has completed; this function waits for
 *       that condition before replacing the write-only configuration.
 */
template <MopConfig<MopTemplate::Template0> Config>
inline __attribute__((always_inline)) void program()
{
    static_assert(Config.start_op.is_set, "Template 0 MOP requires start_op");
    static_assert(Config.start_op_shadow.is_set, "Template 0 MOP requires start_op_shadow");

    constexpr bool has_any_mid_op  = Config.mid_ops.any_set();
    constexpr bool has_all_mid_ops = Config.mid_ops.all_set();
    static_assert(!has_any_mid_op || has_all_mid_ops, "Template 0 MOP requires either all three mid ops or none");

    constexpr bool has_end_op        = Config.end_op.is_set;
    constexpr bool has_end_op_shadow = Config.end_op_shadow.is_set;
    static_assert(has_end_op == has_end_op_shadow, "Template 0 MOP requires end_op and end_op_shadow to be set together");

    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    ckernel::mop_sync();

    mop_cfg[1] = static_cast<std::uint32_t>(has_end_op) | (static_cast<std::uint32_t>(has_all_mid_ops) << 1);
    if constexpr (has_end_op)
    {
        mop_cfg[2] = Config.end_op.value;
    }
    mop_cfg[3] = Config.start_op.value;
    if constexpr (has_all_mid_ops)
    {
        mop_cfg[4] = Config.mid_ops.op_a.value;
        mop_cfg[5] = Config.mid_ops.op_b.value;
        mop_cfg[6] = Config.mid_ops.op_c.value;
    }
    mop_cfg[7] = Config.start_op_shadow.value;
    if constexpr (has_end_op)
    {
        mop_cfg[8] = Config.end_op_shadow.value;
    }
}

/**
 * @brief Program this thread's compile-time Template 1 MOP configuration.
 *
 * @tparam Config: Structural Template 1 configuration value.
 * @note Call only after the preceding MOP expansion has completed; this function waits for
 *       that condition before replacing the write-only configuration.
 * @note Configurations that trigger the OuterCount += 128 hardware bug are rejected.
 */
template <MopConfig<MopTemplate::Template1> Config>
inline __attribute__((always_inline)) void program()
{
    constexpr detail::ResolvedTemplate1Config resolved_config = detail::resolve(Config);
    static_assert(!detail::triggers_outer_count_bug(resolved_config), "Template 1 MOP configuration triggers the OuterCount += 128 hardware bug");

    detail::program(resolved_config);
}

/**
 * @brief Program this thread's runtime Template 1 MOP configuration.
 *
 * @param config: Template 1 configuration to resolve and program.
 * @note Call only after the preceding MOP expansion has completed; this function waits for
 *       that condition before replacing the write-only configuration.
 * @note Configurations that trigger the OuterCount += 128 hardware bug assert at runtime.
 */
inline __attribute__((always_inline)) void program(const MopConfig<MopTemplate::Template1> &config)
{
    const detail::ResolvedTemplate1Config resolved_config = detail::resolve(config);
    LLK_ASSERT(!detail::triggers_outer_count_bug(resolved_config), "Template 1 MOP configuration triggers the OuterCount += 128 hardware bug");

    detail::program(resolved_config);
}

/**
 * @brief Write one value into the selected Template 1 MOP configuration field.
 *
 * @tparam Field: Template 1 field to replace.
 * @param value: Raw 32-bit count or encoded instruction value.
 * @note Call @ref program with a complete Template 1 configuration before patching individual
 *       fields. This function waits for a preceding MOP expansion before writing the field.
 * @note Ensure the resulting complete configuration does not trigger the Template 1
 *       OuterCount hardware bug; write-only MOP state prevents validating it here.
 */
template <Template1Field Field>
inline __attribute__((always_inline)) void write(const std::uint32_t value)
{
    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);

    ckernel::mop_sync();
    mop_cfg[static_cast<std::uint32_t>(Field)] = value;
}

/**
 * @brief Issue a MOP expansion for the selected template.
 *
 * @tparam tmpl: MOP template to run, values = <Template0/Template1>.
 */
template <MopTemplate tmpl>
struct Runner;

/**
 * @brief Issue a Template 0 MOP expansion with compile-time or runtime arguments.
 */
template <>
struct Runner<MopTemplate::Template0>
{
private:
    static constexpr std::uint8_t COUNT_WIDTH   = 7;
    static constexpr std::uint8_t MASK_LO_WIDTH = 16;
    static constexpr std::uint32_t MASK_LO      = (1u << MASK_LO_WIDTH) - 1;

    static constexpr bool is_valid_count(const std::uint32_t count)
    {
        return ckernel::is_valid(count - 1, COUNT_WIDTH);
    }

    template <std::uint32_t Mask>
    inline __attribute__((always_inline)) static void configure_mask()
    {
        TTI_MOP_CFG(Mask >> MASK_LO_WIDTH);
    }

    inline __attribute__((always_inline)) static void configure_mask(const std::uint32_t mask)
    {
        TT_MOP_CFG(mask >> MASK_LO_WIDTH);
    }

public:
    /**
     * @brief Run the programmed Template 0 configuration with compile-time arguments.
     *
     * @tparam Count: Number of expansion iterations, valid values = <1-128>.
     * @tparam Mask: Per-iteration selector mask, consumed least-significant bit first.
     * @note Call @ref program with a Template 0 configuration before this function.
     */
    template <std::uint32_t Count, std::uint32_t Mask>
    inline __attribute__((always_inline)) static void run()
    {
        static_assert(is_valid_count(Count), "MOP encoded count must fit in 7 bits");

        if constexpr (!ckernel::is_valid(Mask, MASK_LO_WIDTH) || Count > MASK_LO_WIDTH)
        {
            configure_mask<Mask>();
        }
        TTI_MOP(0, Count - 1, Mask & MASK_LO);
    }

    /**
     * @brief Run the programmed Template 0 configuration with runtime count.
     *
     * @tparam Mask: Compile-time per-iteration selector mask, consumed least-significant bit first.
     * @param count: Number of expansion iterations, valid values = <1-128>.
     * @note Call @ref program with a Template 0 configuration before this function.
     */
    template <std::uint32_t Mask>
    inline __attribute__((always_inline)) static void run(const std::uint32_t count)
    {
        LLK_ASSERT(is_valid_count(count), "MOP encoded count must fit in 7 bits");

        configure_mask<Mask>();
        TT_MOP(0, count - 1, Mask & MASK_LO);
    }

    /**
     * @brief Run the programmed Template 0 configuration with runtime mask.
     *
     * @tparam Count: Compile-time number of expansion iterations, valid values = <1-128>.
     * @param mask: Per-iteration selector mask, consumed least-significant bit first.
     * @note Call @ref program with a Template 0 configuration before this function.
     */
    template <std::uint32_t Count>
    inline __attribute__((always_inline)) static void run_with_runtime_mask(const std::uint32_t mask)
    {
        static_assert(is_valid_count(Count), "MOP encoded count must fit in 7 bits");

        configure_mask(mask);
        TT_MOP(0, Count - 1, mask & MASK_LO);
    }

    /**
     * @brief Run the programmed Template 0 configuration with runtime arguments.
     *
     * @param count: Number of expansion iterations, valid values = <1-128>.
     * @param mask: Per-iteration selector mask, consumed least-significant bit first.
     * @note Call @ref program with a Template 0 configuration before this function.
     */
    inline __attribute__((always_inline)) static void run(const std::uint32_t count, const std::uint32_t mask)
    {
        LLK_ASSERT(is_valid_count(count), "MOP encoded count must fit in 7 bits");

        configure_mask(mask);
        TT_MOP(0, count - 1, mask & MASK_LO);
    }
};

/**
 * @brief Issue a Template 1 MOP expansion using its programmed loop counts.
 */
template <>
struct Runner<MopTemplate::Template1>
{
    /**
     * @brief Run the programmed Template 1 configuration.
     *
     * @note Call @ref program with a Template 1 configuration before this function.
     *       This function uses zero instruction-level count overrides.
     */
    inline __attribute__((always_inline)) static void run()
    {
        TTI_MOP(1, 0, 0);
    }
};

} // namespace hal::mop
