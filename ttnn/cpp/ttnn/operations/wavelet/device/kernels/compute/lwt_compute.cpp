// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(LWT_SCHEME_HEADER) && defined(LWT_SCHEME_TYPE)
#define WAVELET_1D_ACTIVE_SCHEME_HEADER LWT_SCHEME_HEADER
#define WAVELET_1D_ACTIVE_SCHEME_TYPE LWT_SCHEME_TYPE
#elif defined(ILWT_SCHEME_HEADER) && defined(ILWT_SCHEME_TYPE)
#define WAVELET_1D_ACTIVE_SCHEME_HEADER ILWT_SCHEME_HEADER
#define WAVELET_1D_ACTIVE_SCHEME_TYPE ILWT_SCHEME_TYPE
#else
#error "LWT_SCHEME_HEADER/TYPE or ILWT_SCHEME_HEADER/TYPE must identify the generated lifting scheme"
#endif

#include <array>
#include <cstdint>

// clang-format off
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"
#include "internal/compute/tile_move_copy.h"
#include "ttnn/operations/wavelet/device/protocol/lwt_config.hpp"
#include "ttnn/operations/wavelet/planner/static_scheme.hpp"
#include "../sfpi/horizontal_stencil_sfpi.h"
#include WAVELET_1D_ACTIVE_SCHEME_HEADER
// clang-format on

#ifndef LWT_INLINE_TERMINAL_SCALE
#define LWT_INLINE_TERMINAL_SCALE 0
#endif

#ifndef ILWT_INLINE_INVERSE_SCALE
#define ILWT_INLINE_INVERSE_SCALE 0
#endif

namespace ttnn::operations::wavelet::kernels {

// Seven 32x16 FP32 tiles fit in the four 32x32 slots available under
// tile_regs_acquire(). Place outputs on even narrow tile indices so the
// standard packer can address them through W indices 0, 1, and 2.
constexpr uint32_t kDstBase0 = 0;
constexpr uint32_t kDstSource0 = 1;
constexpr uint32_t kDstBase1 = 2;
constexpr uint32_t kDstSource1 = 3;
constexpr uint32_t kDstBase2 = 4;
constexpr uint32_t kDstSource2 = 5;
constexpr uint32_t kDstSource3 = 6;
constexpr uint32_t kPackBase0 = 0;
constexpr uint32_t kPackBase1 = 1;
constexpr uint32_t kPackBase2 = 2;
constexpr uint32_t kScaleTileCount = 3;
constexpr bool kInlineTerminalScale = LWT_INLINE_TERMINAL_SCALE != 0;
constexpr bool kInlineInverseScale = ILWT_INLINE_INVERSE_SCALE != 0;

static_assert(
    kDstBase2 < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x16>(),
    "1D scale path exceeds the available FP32 narrow-tile destination capacity");

#if defined(TRISC_MATH) || defined(LWT_SCHEME_HEADER)
#define WAVELET_1D_STEP_ATTRIBUTES inline
#else
#define WAVELET_1D_STEP_ATTRIBUTES __attribute__((noinline, noclone))
#endif

template <typename Scheme, uint32_t Index = 0>
constexpr uint32_t last_predict_update_step_index() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return Scheme::num_steps;
    } else {
        constexpr uint32_t later = last_predict_update_step_index<Scheme, Index + 1>();
        using Step = SchemeStep<Scheme, Index>;
        if constexpr (later < Scheme::num_steps) {
            return later;
        } else if constexpr (Step::type == StepType::kPredict || Step::type == StepType::kUpdate) {
            return Index;
        } else {
            return Scheme::num_steps;
        }
    }
}

template <typename Scheme, uint32_t Index = 0>
constexpr uint32_t first_predict_update_step_index() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return Scheme::num_steps;
    } else {
        using Step = SchemeStep<Scheme, Index>;
        if constexpr (Step::type == StepType::kPredict || Step::type == StepType::kUpdate) {
            return Index;
        } else {
            return first_predict_update_step_index<Scheme, Index + 1>();
        }
    }
}

template <typename Scheme, uint32_t End, uint32_t Index = 0>
constexpr uint32_t scale_count_before() noexcept {
    if constexpr (Index >= End) {
        return 0;
    } else {
        using Step = SchemeStep<Scheme, Index>;
        return (Step::type == StepType::kScaleEven || Step::type == StepType::kScaleOdd ? 1U : 0U) +
               scale_count_before<Scheme, End, Index + 1>();
    }
}

template <typename Scheme, uint32_t Index>
constexpr uint32_t swap_count_from() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return 0;
    } else {
        using Step = SchemeStep<Scheme, Index>;
        return (Step::type == StepType::kSwap ? 1U : 0U) + swap_count_from<Scheme, Index + 1>();
    }
}

template <typename Scheme>
constexpr StepType inline_terminal_scale_type() noexcept {
    constexpr uint32_t last_step = last_predict_update_step_index<Scheme>();
    static_assert(last_step < Scheme::num_steps, "Inline terminal scaling requires a predict/update step");
    using LastStep = SchemeStep<Scheme, last_step>;
    constexpr bool target_even = LastStep::type == StepType::kUpdate;
    constexpr bool swapped = (swap_count_from<Scheme, last_step + 1>() & 1U) != 0;
    constexpr bool final_even = target_even != swapped;
    return final_even ? StepType::kScaleEven : StepType::kScaleOdd;
}

template <typename Scheme, StepType ScaleType, uint32_t Index = 0>
constexpr uint32_t terminal_scale_bits() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return 0;
    } else {
        using Step = SchemeStep<Scheme, Index>;
        if constexpr (Step::type == ScaleType) {
            static_assert(Step::k == 1, "Terminal scale must contain exactly one coefficient");
            return Step::coeff_bits[0];
        } else {
            return terminal_scale_bits<Scheme, ScaleType, Index + 1>();
        }
    }
}

template <typename Scheme, bool InlineInverseScale, StepType ScaleType>
constexpr uint32_t maybe_inverse_scale_bits() noexcept {
    if constexpr (!InlineInverseScale) {
        return 0U;
    } else {
        constexpr uint32_t first_step = first_predict_update_step_index<Scheme>();
        static_assert(first_step < Scheme::num_steps, "Inline inverse scaling requires a predict/update step");
        static_assert(
            scale_count_before<Scheme, first_step>() == 2, "Inline inverse scaling requires two leading scales");
        constexpr uint32_t bits = terminal_scale_bits<Scheme, ScaleType>();
        static_assert(bits != 0, "Inline inverse scaling could not find the required reciprocal scale");
        return bits;
    }
}

template <
    uint32_t K,
    bool InlineTerminalScale,
    uint32_t TerminalScalePacked,
    bool ScaleSource,
    bool ScaleBase,
    uint32_t SourceScalePacked,
    uint32_t BaseScalePacked>
WAVELET_1D_STEP_ATTRIBUTES void run_predict_update_step(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const std::array<uint32_t, K> h_coeffs,
    const uint32_t output_group_count) {
    static_assert(K > 0, "Predict/update steps must have at least one coefficient");
    static_assert(K <= device_protocol::kStepCoeffCapacity, "Step coefficient count exceeds device capacity");
    CircularBuffer input0_buffer(cb_input0);
    CircularBuffer input1_buffer(cb_input1);
    CircularBuffer base_buffer(cb_base);
    CircularBuffer output_buffer(cb_output);

    for (uint32_t group = 0; group < output_group_count; ++group) {
        tile_regs_acquire();

        input0_buffer.wait_front(2);
        copy_tile_to_dst_init_short(cb_input0);
        ckernel::internal::copy_tile_to_dst_32x16(cb_input0, 0, kDstSource0);
        ckernel::internal::copy_tile_to_dst_32x16(cb_input0, 1, kDstSource1);
        input0_buffer.pop_front(2);

        input1_buffer.wait_front(2);
        copy_tile_to_dst_init_short(cb_input1);
        ckernel::internal::copy_tile_to_dst_32x16(cb_input1, 0, kDstSource2);
        ckernel::internal::copy_tile_to_dst_32x16(cb_input1, 1, kDstSource3);
        input1_buffer.pop_front(2);

        base_buffer.wait_front(3);
        copy_tile_to_dst_init_short(cb_base);
        ckernel::internal::copy_tile_to_dst_32x16(cb_base, 0, kDstBase0);
        ckernel::internal::copy_tile_to_dst_32x16(cb_base, 1, kDstBase1);
        ckernel::internal::copy_tile_to_dst_32x16(cb_base, 2, kDstBase2);
        base_buffer.pop_front(3);

        hstencil_init();
        if constexpr (ScaleSource || ScaleBase) {
            hstencil_scaled_inputs_plus_base_narrow_tiles<
                K,
                ScaleSource,
                ScaleBase,
                SourceScalePacked,
                BaseScalePacked>(
                h_coeffs, kDstSource0, kDstSource1, kDstSource2, kDstSource3, kDstBase0, kDstBase1, kDstBase2);
        } else {
            hstencil_plus_base_narrow_tiles<K>(
                h_coeffs, kDstSource0, kDstSource1, kDstSource2, kDstSource3, kDstBase0, kDstBase1, kDstBase2);
        }

        if constexpr (InlineTerminalScale) {
            scale_narrow_tile(kDstBase0, TerminalScalePacked);
            scale_narrow_tile(kDstBase1, TerminalScalePacked);
            scale_narrow_tile(kDstBase2, TerminalScalePacked);
        }

        tile_regs_commit();
        tile_regs_wait();

        output_buffer.reserve_back(3);
        pack_tile(kPackBase0, cb_output, 0);
        pack_tile(kPackBase1, cb_output, 1);
        pack_tile(kPackBase2, cb_output, 2);
        output_buffer.push_back(3);

        tile_regs_release();
    }
}

inline void run_scale_step(
    const uint32_t cb_input,
    const uint32_t cb_output,
    const uint32_t scalar_packed,
    const uint32_t output_group_count) {
    CircularBuffer input_buffer(cb_input);
    CircularBuffer output_buffer(cb_output);
    for (uint32_t group = 0; group < output_group_count; ++group) {
        input_buffer.wait_front(kScaleTileCount);
        output_buffer.reserve_back(kScaleTileCount);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_input);
        ckernel::internal::copy_tile_to_dst_32x16(cb_input, 0, kDstBase0);
        ckernel::internal::copy_tile_to_dst_32x16(cb_input, 1, kDstBase1);
        ckernel::internal::copy_tile_to_dst_32x16(cb_input, 2, kDstBase2);
        scale_narrow_tile(kDstBase0, scalar_packed);
        scale_narrow_tile(kDstBase1, scalar_packed);
        scale_narrow_tile(kDstBase2, scalar_packed);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(kPackBase0, cb_output, 0);
        pack_tile(kPackBase1, cb_output, 1);
        pack_tile(kPackBase2, cb_output, 2);
        tile_regs_release();

        input_buffer.pop_front(kScaleTileCount);
        output_buffer.push_back(kScaleTileCount);
    }
}

template <
    typename Scheme,
    bool InlineTerminalScale,
    bool InlineInverseScale,
    bool EvenNeedsScale,
    bool OddNeedsScale,
    uint32_t EvenScalePacked,
    uint32_t OddScalePacked,
    uint32_t StepIndex,
    uint32_t ExecutableIndex>
inline void run_static_steps(
    const uint32_t cb_input0,
    const uint32_t cb_input1,
    const uint32_t cb_base,
    const uint32_t cb_output,
    const uint32_t runtime_arg_base) {
    if constexpr (StepIndex < Scheme::num_steps) {
        using Step = SchemeStep<Scheme, StepIndex>;
        if constexpr (Step::type == StepType::kSwap) {
            run_static_steps<
                Scheme,
                InlineTerminalScale,
                InlineInverseScale,
                OddNeedsScale,
                EvenNeedsScale,
                OddScalePacked,
                EvenScalePacked,
                StepIndex + 1,
                ExecutableIndex>(cb_input0, cb_input1, cb_base, cb_output, runtime_arg_base);
        } else if constexpr (Step::type == StepType::kPredict || Step::type == StepType::kUpdate) {
            const uint32_t output_group_count = get_arg_val<uint32_t>(runtime_arg_base + ExecutableIndex);
            constexpr uint32_t last_predict_update = last_predict_update_step_index<Scheme>();
            constexpr bool inline_terminal_scale = InlineTerminalScale && StepIndex == last_predict_update;
            constexpr bool predict = Step::type == StepType::kPredict;
            constexpr bool scale_source = InlineInverseScale && (predict ? EvenNeedsScale : OddNeedsScale);
            constexpr bool scale_base = InlineInverseScale && (predict ? OddNeedsScale : EvenNeedsScale);
            constexpr uint32_t source_scale_bits = predict ? EvenScalePacked : OddScalePacked;
            constexpr uint32_t base_scale_bits = predict ? OddScalePacked : EvenScalePacked;
            constexpr StepType scale_type = inline_terminal_scale_type<Scheme>();
            constexpr uint32_t scale_bits = terminal_scale_bits<Scheme, scale_type>();
            run_predict_update_step<
                Step::k,
                inline_terminal_scale,
                scale_bits,
                scale_source,
                scale_base,
                source_scale_bits,
                base_scale_bits>(cb_input0, cb_input1, cb_base, cb_output, Step::coeff_bits, output_group_count);
            run_static_steps<
                Scheme,
                InlineTerminalScale,
                InlineInverseScale,
                predict ? EvenNeedsScale : false,
                predict ? false : OddNeedsScale,
                EvenScalePacked,
                OddScalePacked,
                StepIndex + 1,
                ExecutableIndex + 1>(cb_input0, cb_input1, cb_base, cb_output, runtime_arg_base);
        } else if constexpr (Step::type == StepType::kScaleEven || Step::type == StepType::kScaleOdd) {
            constexpr uint32_t first_predict_update = first_predict_update_step_index<Scheme>();
            if constexpr (InlineInverseScale && StepIndex < first_predict_update) {
                run_static_steps<
                    Scheme,
                    InlineTerminalScale,
                    InlineInverseScale,
                    EvenNeedsScale,
                    OddNeedsScale,
                    EvenScalePacked,
                    OddScalePacked,
                    StepIndex + 1,
                    ExecutableIndex>(cb_input0, cb_input1, cb_base, cb_output, runtime_arg_base);
            } else if constexpr (InlineTerminalScale) {
                if constexpr (Step::type == inline_terminal_scale_type<Scheme>()) {
                    run_static_steps<
                        Scheme,
                        InlineTerminalScale,
                        InlineInverseScale,
                        EvenNeedsScale,
                        OddNeedsScale,
                        EvenScalePacked,
                        OddScalePacked,
                        StepIndex + 1,
                        ExecutableIndex>(cb_input0, cb_input1, cb_base, cb_output, runtime_arg_base);
                } else {
                    static_assert(Step::k == 1, "Scale steps must have exactly one coefficient");
                    const uint32_t output_group_count = get_arg_val<uint32_t>(runtime_arg_base + ExecutableIndex);
                    run_scale_step(cb_base, cb_output, Step::coeff_bits[0], output_group_count);
                    run_static_steps<
                        Scheme,
                        InlineTerminalScale,
                        InlineInverseScale,
                        EvenNeedsScale,
                        OddNeedsScale,
                        EvenScalePacked,
                        OddScalePacked,
                        StepIndex + 1,
                        ExecutableIndex + 1>(cb_input0, cb_input1, cb_base, cb_output, runtime_arg_base);
                }
            } else {
                static_assert(InlineInverseScale, "Every production scale route must use an inline scale policy");
            }
        }
    } else {
        static_assert(
            !InlineInverseScale || ((!EvenNeedsScale || EvenScalePacked == 0x3f800000U) &&
                                    (!OddNeedsScale || OddScalePacked == 0x3f800000U)),
            "Inline inverse scaling left a final stream unscaled");
    }
}

template <typename Scheme>
void lwt_compute() {
    constexpr uint32_t cb_input0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);
    static_assert(
        kInlineTerminalScale != kInlineInverseScale, "Exactly one scale policy must be selected for LWT or ILWT");
    constexpr uint32_t route_count =
        executable_step_count<Scheme>() - (kInlineTerminalScale ? 1U : 0U) - (kInlineInverseScale ? 2U : 0U);
    constexpr uint32_t inverse_even_scale =
        maybe_inverse_scale_bits<Scheme, kInlineInverseScale, StepType::kScaleEven>();
    constexpr uint32_t inverse_odd_scale = maybe_inverse_scale_bits<Scheme, kInlineInverseScale, StepType::kScaleOdd>();
    const uint32_t chunk_count = get_arg_val<uint32_t>(0);

    for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
        run_static_steps<
            Scheme,
            kInlineTerminalScale,
            kInlineInverseScale,
            kInlineInverseScale,
            kInlineInverseScale,
            inverse_even_scale,
            inverse_odd_scale,
            0,
            0>(cb_input0, cb_input1, cb_base, cb_output, 1 + chunk * route_count);
    }
}

}  // namespace ttnn::operations::wavelet::kernels

#undef WAVELET_1D_STEP_ATTRIBUTES

void kernel_main() {
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);
    init_sfpu(cb_base, cb_output);
    ttnn::operations::wavelet::kernels::lwt_compute<WAVELET_1D_ACTIVE_SCHEME_TYPE>();
}
