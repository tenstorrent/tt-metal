// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tt_stl/assert.hpp>
#include <tuple>
#include <vector>

#include "ttnn/operations/wavelet/common/boundary.hpp"
#include "ttnn/operations/wavelet/common/signal.hpp"
#include "ttnn/operations/wavelet/device/protocol/lwt_config.hpp"
#include "ttnn/operations/wavelet/planner/static_scheme.hpp"
#include "ttnn/operations/wavelet/planner/step.hpp"
#include "ttnn/operations/wavelet/planner/pad_split_layout.hpp"

namespace ttnn::operations::wavelet {

enum class StorageSlot : uint8_t {
    kA = 0,
    kB = 1,
    kScratch = 2,
};

struct StreamRef {
    StorageSlot slot{StorageSlot::kA};
};

enum class RouteOutputStorage : uint8_t {
    kWorkspaceSlot = 0,
    kFinalEvenDram = 1,
    kFinalOddDram = 2,
};

struct RouteOutputRef {
    RouteOutputStorage storage{RouteOutputStorage::kWorkspaceSlot};
    StorageSlot slot{StorageSlot::kA};
};

struct LiftingStepRoute {
    StepType type{StepType::kPredict};
    StreamRef source{};
    StreamRef base{};
    RouteOutputRef output{};
    size_t source_length{0};
    size_t base_length{0};
    size_t source_offset{0};
    size_t base_offset{0};
    uint32_t source_left_pad{0};
    size_t output_length{0};
};

struct LiftingActiveStreams {
    StreamRef even{.slot = StorageSlot::kA};
    StreamRef odd{.slot = StorageSlot::kB};
    StorageSlot free{StorageSlot::kScratch};
};

struct LiftingForwardPlan {
    PadSplit1DLayout preprocess_layout{};
    std::vector<LiftingStepRoute> routes;
    size_t final_even_length{0};
    size_t final_odd_length{0};
    int final_even_shift{0};
    int final_odd_shift{0};
    size_t output_length{0};
};

namespace detail {

struct StreamState {
    int shift{0};
    size_t length{0};
};

[[nodiscard]] constexpr RouteOutputRef workspace_output(const StorageSlot slot) noexcept {
    return RouteOutputRef{.storage = RouteOutputStorage::kWorkspaceSlot, .slot = slot};
}

[[nodiscard]] inline std::tuple<int, size_t, size_t, size_t> compute_step_geometry(
    const StreamState& source, const int kernel_shift, const size_t k, const StreamState& base) {
    const int conv_shift = source.shift + kernel_shift + static_cast<int>(std::min(source.length, k)) - 1;
    const size_t conv_length = source.length >= k ? source.length - k + 1 : 0;

    const int out_shift = std::max(base.shift, conv_shift);
    const int out_end =
        std::min(base.shift + static_cast<int>(base.length), conv_shift + static_cast<int>(conv_length));
    const size_t out_length = out_end > out_shift ? static_cast<size_t>(out_end - out_shift) : 0;

    const size_t source_offset = static_cast<size_t>(out_shift - conv_shift);
    const size_t base_offset = static_cast<size_t>(out_shift - base.shift);

    return {out_shift, out_length, source_offset, base_offset};
}

template <typename Scheme, size_t Index>
void append_forward_route(
    std::vector<LiftingStepRoute>& routes,
    LiftingActiveStreams& active,
    StreamState& even_state,
    StreamState& odd_state) {
    using Step = SchemeStep<Scheme, Index>;

    if constexpr (Step::type == StepType::kPredict) {
        static_assert(Step::k > 0, "Predict steps must have at least one coefficient");
        static_assert(
            Step::k <= device_protocol::kStepCoeffCapacity, "Predict step exceeds device coefficient capacity");
        const auto [out_shift, out_length, src_off, base_off] =
            compute_step_geometry(even_state, Step::shift, Step::k, odd_state);
        const StreamRef output{.slot = active.free};
        const StorageSlot released = active.odd.slot;
        routes.push_back(LiftingStepRoute{
            .type = Step::type,
            .source = active.even,
            .base = active.odd,
            .output = workspace_output(output.slot),
            .source_length = even_state.length,
            .base_length = odd_state.length,
            .source_offset = src_off,
            .base_offset = base_off,
            .source_left_pad = device_protocol::kStepCoeffCapacity - Step::k,
            .output_length = out_length,
        });
        odd_state = StreamState{.shift = out_shift, .length = out_length};
        active.odd = output;
        active.free = released;
    } else if constexpr (Step::type == StepType::kUpdate) {
        static_assert(Step::k > 0, "Update steps must have at least one coefficient");
        static_assert(
            Step::k <= device_protocol::kStepCoeffCapacity, "Update step exceeds device coefficient capacity");
        const auto [out_shift, out_length, src_off, base_off] =
            compute_step_geometry(odd_state, Step::shift, Step::k, even_state);
        const StreamRef output{.slot = active.free};
        const StorageSlot released = active.even.slot;
        routes.push_back(LiftingStepRoute{
            .type = Step::type,
            .source = active.odd,
            .base = active.even,
            .output = workspace_output(output.slot),
            .source_length = odd_state.length,
            .base_length = even_state.length,
            .source_offset = src_off,
            .base_offset = base_off,
            .source_left_pad = device_protocol::kStepCoeffCapacity - Step::k,
            .output_length = out_length,
        });
        even_state = StreamState{.shift = out_shift, .length = out_length};
        active.even = output;
        active.free = released;
    } else if constexpr (Step::type == StepType::kScaleOdd) {
        static_assert(Step::k == 1, "Scale odd steps must have exactly one coefficient");
        const StreamRef output{.slot = active.free};
        const StorageSlot released = active.odd.slot;
        routes.push_back(LiftingStepRoute{
            .type = Step::type,
            .source = active.odd,
            .base = active.odd,
            .output = workspace_output(output.slot),
            .source_length = odd_state.length,
            .base_length = odd_state.length,
            .source_offset = 0,
            .base_offset = 0,
            .source_left_pad = 0,
            .output_length = odd_state.length,
        });
        active.odd = output;
        active.free = released;
    } else if constexpr (Step::type == StepType::kScaleEven) {
        static_assert(Step::k == 1, "Scale even steps must have exactly one coefficient");
        const StreamRef output{.slot = active.free};
        const StorageSlot released = active.even.slot;
        routes.push_back(LiftingStepRoute{
            .type = Step::type,
            .source = active.even,
            .base = active.even,
            .output = workspace_output(output.slot),
            .source_length = even_state.length,
            .base_length = even_state.length,
            .source_offset = 0,
            .base_offset = 0,
            .source_left_pad = 0,
            .output_length = even_state.length,
        });
        active.even = output;
        active.free = released;
    } else {
        static_assert(Step::type == StepType::kSwap, "Unsupported static lifting step type");
        static_assert(Step::k == 0, "Swap steps must not have coefficients");
        routes.push_back(LiftingStepRoute{
            .type = Step::type,
            .source = active.even,
            .base = active.odd,
            .output = workspace_output(active.even.slot),
            .source_length = even_state.length,
            .base_length = odd_state.length,
            .source_offset = 0,
            .base_offset = 0,
            .source_left_pad = 0,
            .output_length = 0,
        });
        std::swap(active.even, active.odd);
        std::swap(even_state, odd_state);
    }
}

template <typename Scheme, size_t Index = 0>
void append_forward_routes(
    std::vector<LiftingStepRoute>& routes,
    LiftingActiveStreams& active,
    StreamState& even_state,
    StreamState& odd_state) {
    if constexpr (Index < Scheme::num_steps) {
        append_forward_route<Scheme, Index>(routes, active, even_state, odd_state);
        append_forward_routes<Scheme, Index + 1>(routes, active, even_state, odd_state);
    }
}

inline void route_terminal_scales_to_final_dram(std::vector<LiftingStepRoute>& routes) {
    TT_FATAL(routes.size() >= 2, "Lifting scheme must end in scale-even and scale-odd routes");

    const size_t terminal_begin = routes.size() - 2;
    for (size_t index = 0; index < terminal_begin; ++index) {
        TT_FATAL(!is_scale_step(routes[index].type), "Only the final two lifting routes may be scale routes");
    }

    auto& first = routes[terminal_begin];
    auto& second = routes[terminal_begin + 1];
    TT_FATAL(
        is_scale_step(first.type) && is_scale_step(second.type) && first.type != second.type,
        "Lifting scheme must end in one scale-even and one scale-odd route");

    for (auto* route : {&first, &second}) {
        route->output = RouteOutputRef{
            .storage = route->type == StepType::kScaleEven ? RouteOutputStorage::kFinalEvenDram
                                                           : RouteOutputStorage::kFinalOddDram,
            .slot = route->output.slot,
        };
    }
}

}  // namespace detail

template <typename Scheme>
[[nodiscard]] LiftingForwardPlan make_forward_lifting_plan(
    const SignalBuffer& input, const BoundaryMode boundary_mode = BoundaryMode::kSymmetric) {
    static_assert(Scheme::tap_size > 0, "Static lifting schemes must have a positive tap size");
    static_assert(Scheme::num_steps > 0, "Static lifting schemes must have at least one step");
    TT_FATAL(input.element_size_bytes == sizeof(float), "Forward lifting plan currently supports fp32 only");
    TT_FATAL(
        input.length <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        "Input length {} exceeds uint32_t runtime limits",
        input.length);

    const uint32_t wavelet_pad = static_cast<uint32_t>(Scheme::tap_size - 1);
    const PadSplit1DLayout preprocess_layout =
        make_pad_split_1d_layout(input, Pad1DConfig{.mode = boundary_mode, .left = wavelet_pad, .right = wavelet_pad});

    const SignalBuffer initial_even = preprocess_layout.output.even;
    const SignalBuffer initial_odd = preprocess_layout.output.odd;
    detail::StreamState even_state{.shift = Scheme::delay_even, .length = initial_even.length};
    detail::StreamState odd_state{.shift = Scheme::delay_odd, .length = initial_odd.length};

    std::vector<LiftingStepRoute> routes;
    routes.reserve(Scheme::num_steps);

    LiftingActiveStreams active{};
    detail::append_forward_routes<Scheme>(routes, active, even_state, odd_state);
    detail::route_terminal_scales_to_final_dram(routes);

    return LiftingForwardPlan{
        .preprocess_layout = preprocess_layout,
        .routes = std::move(routes),
        .final_even_length = even_state.length,
        .final_odd_length = odd_state.length,
        .final_even_shift = even_state.shift,
        .final_odd_shift = odd_state.shift,
        .output_length = (input.length + static_cast<size_t>(Scheme::tap_size) - 1) / size_t{2},
    };
}

}  // namespace ttnn::operations::wavelet
