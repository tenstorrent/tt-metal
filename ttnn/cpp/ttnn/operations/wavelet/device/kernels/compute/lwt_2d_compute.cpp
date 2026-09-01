// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

// clang-format off
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/wavelet/device/protocol/lwt_2d_config.hpp"
#include "ttnn/operations/wavelet/planner/static_scheme.hpp"
#include "../sfpi/horizontal_stencil_sfpi.h"
#include "../sfpi/scale_sfpi.h"
#include "../sfpi/vertical_stencil_sfpi.h"

#if defined(LWT_2D_SCHEME_HEADER) && defined(LWT_2D_SCHEME_TYPE)
#define WAVELET_2D_ACTIVE_SCHEME_HEADER LWT_2D_SCHEME_HEADER
#define WAVELET_2D_ACTIVE_SCHEME_TYPE LWT_2D_SCHEME_TYPE
#elif defined(ILWT_2D_SCHEME_HEADER) && defined(ILWT_2D_SCHEME_TYPE)
#define WAVELET_2D_ACTIVE_SCHEME_HEADER ILWT_2D_SCHEME_HEADER
#define WAVELET_2D_ACTIVE_SCHEME_TYPE ILWT_2D_SCHEME_TYPE
#else
#error "LWT_2D_SCHEME_HEADER/TYPE or ILWT_2D_SCHEME_HEADER/TYPE must identify the generated lifting scheme"
#endif
#include WAVELET_2D_ACTIVE_SCHEME_HEADER
// clang-format on

namespace {

using Scheme = WAVELET_2D_ACTIVE_SCHEME_TYPE;

#ifdef ILWT_2D
constexpr bool kInverse = true;
#else
constexpr bool kInverse = false;
#endif

#if defined(ILWT_2D)
constexpr bool kCompactInverseCodegen = true;
#else
constexpr bool kCompactInverseCodegen = false;
#endif

#define WAVELET_2D_STENCIL_ATTRIBUTES __attribute__((noinline, noclone, optimize("Os")))
#define WAVELET_2D_AXIS_ATTRIBUTES __attribute__((noinline, noclone, optimize("Os")))

template <uint32_t Index = 0>
constexpr uint32_t first_predict_update_step_index() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return Scheme::num_steps;
    } else {
        using Step = ttnn::operations::wavelet::SchemeStep<Scheme, Index>;
        if constexpr (ttnn::operations::wavelet::is_predict_update_step(Step::type)) {
            return Index;
        } else {
            return first_predict_update_step_index<Index + 1>();
        }
    }
}

template <uint32_t End, uint32_t Index = 0>
constexpr uint32_t scale_count_before() noexcept {
    if constexpr (Index >= End) {
        return 0;
    } else {
        using Step = ttnn::operations::wavelet::SchemeStep<Scheme, Index>;
        return (ttnn::operations::wavelet::is_scale_step(Step::type) ? 1U : 0U) + scale_count_before<End, Index + 1>();
    }
}

template <uint32_t Index = 0>
constexpr uint32_t last_predict_update_step_index() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return Scheme::num_steps;
    } else {
        constexpr uint32_t later = last_predict_update_step_index<Index + 1>();
        using Step = ttnn::operations::wavelet::SchemeStep<Scheme, Index>;
        if constexpr (later < Scheme::num_steps) {
            return later;
        } else if constexpr (ttnn::operations::wavelet::is_predict_update_step(Step::type)) {
            return Index;
        } else {
            return Scheme::num_steps;
        }
    }
}

template <uint32_t Index>
constexpr uint32_t swap_count_from() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return 0;
    } else {
        using Step = ttnn::operations::wavelet::SchemeStep<Scheme, Index>;
        return (Step::type == ttnn::operations::wavelet::StepType::kSwap ? 1U : 0U) + swap_count_from<Index + 1>();
    }
}

constexpr ttnn::operations::wavelet::StepType inline_terminal_scale_type() noexcept {
    constexpr uint32_t last_step = last_predict_update_step_index();
    static_assert(last_step < Scheme::num_steps, "2D terminal scale fusion requires a predict/update step");
    using LastStep = ttnn::operations::wavelet::SchemeStep<Scheme, last_step>;
    constexpr bool target_even = LastStep::type == ttnn::operations::wavelet::StepType::kUpdate;
    constexpr bool swapped = (swap_count_from<last_step + 1>() & 1U) != 0;
    constexpr bool final_even = target_even != swapped;
    return final_even ? ttnn::operations::wavelet::StepType::kScaleEven
                      : ttnn::operations::wavelet::StepType::kScaleOdd;
}

template <ttnn::operations::wavelet::StepType ScaleType, uint32_t Index = 0>
constexpr uint32_t terminal_scale_bits() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return 0;
    } else {
        using Step = ttnn::operations::wavelet::SchemeStep<Scheme, Index>;
        if constexpr (Step::type == ScaleType) {
            static_assert(Step::k == 1, "2D terminal scale must contain one coefficient");
            return Step::coeff_bits[0];
        } else {
            return terminal_scale_bits<ScaleType, Index + 1>();
        }
    }
}

template <ttnn::operations::wavelet::StepType ScaleType>
constexpr uint32_t inverse_scale_bits() noexcept {
    constexpr uint32_t first_step = first_predict_update_step_index();
    static_assert(first_step < Scheme::num_steps, "2D inline inverse scaling requires a predict/update step");
    static_assert(scale_count_before<first_step>() == 2, "2D inline inverse scaling requires two leading scales");
    constexpr uint32_t bits = terminal_scale_bits<ScaleType>();
    static_assert(bits != 0, "2D inline inverse scaling could not find a reciprocal scale");
    return bits;
}

template <ttnn::operations::wavelet::StepType ScaleType>
constexpr uint32_t maybe_inverse_scale_bits() noexcept {
    if constexpr (kInverse) {
        return inverse_scale_bits<ScaleType>();
    }
    return 0U;
}

__attribute__((noinline)) void run_scale(
    const uint32_t tile_count, const uint32_t cb_source0, const uint32_t cb_output, const uint32_t coefficient) {
    CircularBuffer source0_buffer(cb_source0);
    CircularBuffer output_buffer(cb_output);
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        tile_regs_acquire();
        source0_buffer.wait_front(1);
        copy_tile_to_dst_init_short(cb_source0);
        copy_tile(cb_source0, 0, 0);
        source0_buffer.pop_front(1);
        scale_tile(0, coefficient);
        tile_regs_commit();
        tile_regs_wait();
        output_buffer.reserve_back(1);
        pack_tile(0, cb_output);
        output_buffer.push_back(1);
        tile_regs_release();
    }
}

struct CompactInverseScalePolicy {
    uint32_t source_scale_bits;
    uint32_t base_scale_bits;

    inline void scale_sources(const uint32_t source0, const uint32_t source1) const {
        if (source_scale_bits != 0) {
            scale_tile(source0, source_scale_bits);
            scale_tile(source1, source_scale_bits);
        }
    }

    inline void scale_base(const uint32_t base) const {
        if (base_scale_bits != 0) {
            scale_tile(base, base_scale_bits);
        }
    }
};

template <bool ScaleSource, bool ScaleBase, uint32_t SourceScaleBits, uint32_t BaseScaleBits>
struct SpecializedScalePolicy {
    inline void scale_sources(const uint32_t source0, const uint32_t source1) const {
        if constexpr (ScaleSource) {
            scale_tile(source0, SourceScaleBits);
            scale_tile(source1, SourceScaleBits);
        }
    }

    inline void scale_base(const uint32_t base) const {
        if constexpr (ScaleBase) {
            scale_tile(base, BaseScaleBits);
        }
    }
};

template <uint8_t K, bool Vertical, bool InlineTerminalScale, typename ScalePolicy>
WAVELET_2D_STENCIL_ATTRIBUTES void run_stencil(
    const uint32_t tile_count,
    const uint32_t cb_source0,
    const uint32_t cb_source1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const ScalePolicy scale_policy,
    const std::array<uint32_t, K> coefficients) {
    CircularBuffer source0_buffer(cb_source0);
    CircularBuffer source1_buffer(cb_source1);
    CircularBuffer base_buffer(cb_base);
    CircularBuffer output_buffer(cb_output);
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        tile_regs_acquire();
        source0_buffer.wait_front(1);
        copy_tile_to_dst_init_short(cb_source0);
        copy_tile(cb_source0, 0, 0);
        source0_buffer.pop_front(1);

        source1_buffer.wait_front(1);
        copy_tile_to_dst_init_short(cb_source1);
        copy_tile(cb_source1, 0, 1);
        source1_buffer.pop_front(1);

        base_buffer.wait_front(1);
        copy_tile_to_dst_init_short(cb_base);
        copy_tile(cb_base, 0, 2);
        base_buffer.pop_front(1);

        scale_policy.scale_sources(0, 1);
        scale_policy.scale_base(2);
        if constexpr (Vertical) {
            vstencil_init();
            vstencil_tile<K>(coefficients, 0, 1, 3, 2);
        } else {
            hstencil_init();
            hstencil_dense_tile<K>(coefficients, 0, 1, 2, 3);
        }
        if constexpr (InlineTerminalScale) {
            constexpr ttnn::operations::wavelet::StepType scale_type = inline_terminal_scale_type();
            constexpr uint32_t scale_bits = terminal_scale_bits<scale_type>();
            static_assert(scale_bits != 0, "2D terminal scale fusion could not find its scale coefficient");
            scale_tile(3, scale_bits);
        }
        tile_regs_commit();
        tile_regs_wait();
        output_buffer.reserve_back(1);
        pack_tile(3, cb_output);
        output_buffer.push_back(1);
        tile_regs_release();
    }
}

template <
    typename Step,
    bool Vertical,
    bool InlineTerminalScale,
    bool ScaleSource,
    bool ScaleBase,
    uint32_t SourceScaleBits,
    uint32_t BaseScaleBits>
inline void run_step(
    const uint32_t tile_count,
    const uint32_t cb_source0,
    const uint32_t cb_source1,
    const uint32_t cb_base,
    const uint32_t cb_output) {
    if constexpr (Step::type == ttnn::operations::wavelet::StepType::kSwap) {
        return;
    } else if constexpr (ttnn::operations::wavelet::is_scale_step(Step::type)) {
        if constexpr (kInverse || Step::type == inline_terminal_scale_type()) {
            // The terminal forward scale and both inverse reciprocal scales
            // are metadata-only. The companion forward stream remains an
            // executable scale route under the production planner contract.
            return;
        } else {
            static_assert(Step::k == 1, "2D scale route must contain exactly one coefficient");
            run_scale(tile_count, cb_source0, cb_output, Step::coeff_bits[0]);
        }
    } else {
        if constexpr (kCompactInverseCodegen) {
            run_stencil<Step::k, Vertical, InlineTerminalScale>(
                tile_count,
                cb_source0,
                cb_source1,
                cb_base,
                cb_output,
                CompactInverseScalePolicy{
                    .source_scale_bits = ScaleSource ? SourceScaleBits : 0,
                    .base_scale_bits = ScaleBase ? BaseScaleBits : 0,
                },
                Step::coeff_bits);
        } else {
            run_stencil<Step::k, Vertical, InlineTerminalScale>(
                tile_count,
                cb_source0,
                cb_source1,
                cb_base,
                cb_output,
                SpecializedScalePolicy<ScaleSource, ScaleBase, SourceScaleBits, BaseScaleBits>{},
                Step::coeff_bits);
        }
    }
}

template <
    bool Vertical,
    bool EvenNeedsScale,
    bool OddNeedsScale,
    uint32_t EvenScaleBits,
    uint32_t OddScaleBits,
    size_t StepIndex = 0>
WAVELET_2D_AXIS_ATTRIBUTES void run_axis(
    const uint32_t runtime_arg_base,
    const uint32_t route_offset,
    const uint32_t cb_source0,
    const uint32_t cb_source1,
    const uint32_t cb_base,
    const uint32_t cb_output) {
    if constexpr (StepIndex < Scheme::num_steps) {
        using Step = ttnn::operations::wavelet::SchemeStep<Scheme, StepIndex>;
        const uint32_t route_index = route_offset + StepIndex;
        const uint32_t packed_counts = get_arg_val<uint32_t>(runtime_arg_base + route_index / 4);
        const uint32_t tile_count = (packed_counts >> (8 * (route_index % 4))) & 0xFFU;
        constexpr bool predict = Step::type == ttnn::operations::wavelet::StepType::kPredict;
        constexpr bool scale_source = kInverse && ttnn::operations::wavelet::is_predict_update_step(Step::type) &&
                                      (predict ? EvenNeedsScale : OddNeedsScale);
        constexpr bool scale_base = kInverse && ttnn::operations::wavelet::is_predict_update_step(Step::type) &&
                                    (predict ? OddNeedsScale : EvenNeedsScale);
        constexpr uint32_t source_scale_bits = predict ? EvenScaleBits : OddScaleBits;
        constexpr uint32_t base_scale_bits = predict ? OddScaleBits : EvenScaleBits;
        constexpr bool inline_terminal_scale = !kInverse && StepIndex == last_predict_update_step_index();
        run_step<Step, Vertical, inline_terminal_scale, scale_source, scale_base, source_scale_bits, base_scale_bits>(
            tile_count, cb_source0, cb_source1, cb_base, cb_output);
        if constexpr (Step::type == ttnn::operations::wavelet::StepType::kSwap) {
            run_axis<Vertical, OddNeedsScale, EvenNeedsScale, OddScaleBits, EvenScaleBits, StepIndex + 1>(
                runtime_arg_base, route_offset, cb_source0, cb_source1, cb_base, cb_output);
        } else if constexpr (ttnn::operations::wavelet::is_predict_update_step(Step::type)) {
            run_axis<
                Vertical,
                predict ? EvenNeedsScale : false,
                predict ? false : OddNeedsScale,
                EvenScaleBits,
                OddScaleBits,
                StepIndex + 1>(runtime_arg_base, route_offset, cb_source0, cb_source1, cb_base, cb_output);
        } else {
            run_axis<Vertical, EvenNeedsScale, OddNeedsScale, EvenScaleBits, OddScaleBits, StepIndex + 1>(
                runtime_arg_base, route_offset, cb_source0, cb_source1, cb_base, cb_output);
        }
    } else {
        static_assert(
            !kInverse ||
                ((!EvenNeedsScale || EvenScaleBits == 0x3f800000U) && (!OddNeedsScale || OddScaleBits == 0x3f800000U)),
            "2D inline inverse scaling left a final stream unscaled");
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_source0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_source1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);
    init_sfpu(cb_base, cb_output);
    const uint32_t chunk_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t routes_per_axis = Scheme::num_steps;
    constexpr uint32_t routes_per_chunk = 4 * routes_per_axis;
    constexpr uint32_t packed_words_per_chunk = (routes_per_chunk + 3) / 4;
    constexpr uint32_t inverse_even_scale = maybe_inverse_scale_bits<ttnn::operations::wavelet::StepType::kScaleEven>();
    constexpr uint32_t inverse_odd_scale = maybe_inverse_scale_bits<ttnn::operations::wavelet::StepType::kScaleOdd>();

    for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
        const uint32_t runtime_arg_base = 1 + chunk * packed_words_per_chunk;
        if constexpr (kInverse) {
            run_axis<false, true, true, inverse_even_scale, inverse_odd_scale>(
                runtime_arg_base, 0, cb_source0, cb_source1, cb_base, cb_output);
            run_axis<false, true, true, inverse_even_scale, inverse_odd_scale>(
                runtime_arg_base, routes_per_axis, cb_source0, cb_source1, cb_base, cb_output);
            run_axis<true, true, true, inverse_even_scale, inverse_odd_scale>(
                runtime_arg_base, 2 * routes_per_axis, cb_source0, cb_source1, cb_base, cb_output);
            run_axis<true, true, true, inverse_even_scale, inverse_odd_scale>(
                runtime_arg_base, 3 * routes_per_axis, cb_source0, cb_source1, cb_base, cb_output);
        } else {
            run_axis<true, false, false, 0, 0>(runtime_arg_base, 0, cb_source0, cb_source1, cb_base, cb_output);
            run_axis<true, false, false, 0, 0>(
                runtime_arg_base, routes_per_axis, cb_source0, cb_source1, cb_base, cb_output);
            run_axis<false, false, false, 0, 0>(
                runtime_arg_base, 2 * routes_per_axis, cb_source0, cb_source1, cb_base, cb_output);
            run_axis<false, false, false, 0, 0>(
                runtime_arg_base, 3 * routes_per_axis, cb_source0, cb_source1, cb_base, cb_output);
        }
    }
}

#undef WAVELET_2D_STENCIL_ATTRIBUTES
#undef WAVELET_2D_AXIS_ATTRIBUTES
