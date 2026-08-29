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
#include <utility>
#include <vector>

#include "tt-metalium/math.hpp"
#include "ttnn/operations/wavelet/common/signal.hpp"
#include "ttnn/operations/wavelet/common/storage_contract.hpp"
#include "ttnn/operations/wavelet/device/protocol/lwt_config.hpp"
#include "ttnn/operations/wavelet/planner/plan.hpp"

namespace ttnn::operations::wavelet {

struct IndexInterval {
    size_t begin{0};
    size_t end{0};

    [[nodiscard]] constexpr bool empty() const noexcept { return begin == end; }
    [[nodiscard]] constexpr size_t length() const noexcept { return end - begin; }
};

struct AxisRequiredStreams {
    IndexInterval even{};
    IndexInterval odd{};
};

struct AxisRouteRequirement {
    StepType type{StepType::kPredict};
    AxisRequiredStreams before{};
    AxisRequiredStreams after{};
    IndexInterval source{};
    IndexInterval base{};
    IndexInterval output{};
};

struct AxisConePlan {
    IndexInterval final_even{};
    IndexInterval final_odd{};
    IndexInterval initial_even{};
    IndexInterval initial_odd{};
    std::vector<AxisRouteRequirement> routes;
};

struct LwtStepRoute {
    StepType type{StepType::kPredict};
    StreamRef source{};
    StreamRef base{};
    RouteOutputRef output{};
    size_t source_storage_length{0};
    size_t base_storage_length{0};
    size_t source_offset_elements{0};
    size_t base_offset_elements{0};
    uint32_t source_left_pad_elements{0};
    size_t output_length{0};
    size_t output_offset_elements{0};
};

struct LwtChunkPlan {
    IndexInterval initial_even{};
    IndexInterval initial_odd{};
    std::vector<LwtStepRoute> routes;
    size_t max_workspace_elements{0};
    double dependency_overhead{0.0};
};

enum class WorkspaceLayout : uint8_t {
    kRowMajor,
    kTileNative,
};

struct LwtExecutionPlan {
    LiftingForwardPlan full_plan{};
    std::vector<LwtChunkPlan> chunks;
    uint32_t groups_per_chunk{0};
    uint32_t workspace_elements{0};
    uint32_t max_workspace_elements{0};
    double max_dependency_overhead{0.0};
    WorkspaceLayout workspace_layout{WorkspaceLayout::kRowMajor};
};

namespace execution_detail {

using RequiredStreams = AxisRequiredStreams;

struct StoredStream {
    StorageSlot slot{StorageSlot::kA};
    IndexInterval storage{};
};

struct TerminalScaleInline {
    size_t predict_update_route_index{0};
    StepType scale_type{StepType::kScaleEven};
    RouteOutputStorage final_storage{RouteOutputStorage::kFinalEvenDram};
};

[[nodiscard]] constexpr bool contains(const IndexInterval outer, const IndexInterval inner) noexcept {
    return inner.empty() || (outer.begin <= inner.begin && inner.end <= outer.end);
}

[[nodiscard]] constexpr IndexInterval hull(const IndexInterval lhs, const IndexInterval rhs) noexcept {
    if (lhs.empty()) {
        return rhs;
    }
    if (rhs.empty()) {
        return lhs;
    }
    return IndexInterval{.begin = std::min(lhs.begin, rhs.begin), .end = std::max(lhs.end, rhs.end)};
}

[[nodiscard]] inline IndexInterval translated(
    const IndexInterval interval, const size_t left_offset, const size_t right_expansion = 0) {
    if (interval.empty()) {
        return interval;
    }
    TT_FATAL(
        interval.begin <= std::numeric_limits<size_t>::max() - left_offset &&
            interval.end <= std::numeric_limits<size_t>::max() - left_offset - right_expansion,
        "LWT dependency interval translation overflows size_t");
    return IndexInterval{
        .begin = interval.begin + left_offset,
        .end = interval.end + left_offset + right_expansion,
    };
}

inline void validate_interval(const IndexInterval interval, const size_t stream_length, const char* label) {
    TT_FATAL(interval.begin <= interval.end, "{} LWT interval is inverted", label);
    TT_FATAL(
        interval.end <= stream_length,
        "{} LWT interval [{}, {}) exceeds stream length {}",
        label,
        interval.begin,
        interval.end,
        stream_length);
}

[[nodiscard]] inline uint32_t coefficient_count(const LiftingStepRoute& route) {
    TT_FATAL(
        route.source_left_pad <= device_protocol::kStepCoeffCapacity,
        "Lifting route left pad {} exceeds coefficient capacity {}",
        route.source_left_pad,
        device_protocol::kStepCoeffCapacity);
    const uint32_t count = device_protocol::kStepCoeffCapacity - route.source_left_pad;
    TT_FATAL(count > 0, "Predict/update LWT route has no coefficients");
    return count;
}

[[nodiscard]] inline TerminalScaleInline terminal_scale_inline(const LiftingForwardPlan& plan) {
    TT_FATAL(plan.routes.size() >= 3, "LWT terminal-scale inline path requires a predict/update and two scales");

    size_t predict_update_route_index = plan.routes.size();
    for (size_t reverse_index = plan.routes.size() - 2; reverse_index > 0; --reverse_index) {
        const size_t route_index = reverse_index - 1;
        if (is_predict_update_step(plan.routes[route_index].type)) {
            predict_update_route_index = route_index;
            break;
        }
    }
    TT_FATAL(
        predict_update_route_index < plan.routes.size(),
        "LWT terminal-scale inline path could not find a predict/update route");

    bool final_even = plan.routes[predict_update_route_index].type == StepType::kUpdate;
    for (size_t route_index = predict_update_route_index + 1; route_index + 2 < plan.routes.size(); ++route_index) {
        TT_FATAL(
            plan.routes[route_index].type == StepType::kSwap,
            "Only metadata swaps may follow the final predict/update before terminal scales");
        final_even = !final_even;
    }

    const StepType scale_type = final_even ? StepType::kScaleEven : StepType::kScaleOdd;
    const auto final_storage = final_even ? RouteOutputStorage::kFinalEvenDram : RouteOutputStorage::kFinalOddDram;
    const bool scale_exists =
        std::any_of(plan.routes.end() - 2, plan.routes.end(), [scale_type](const LiftingStepRoute& route) {
            return route.type == scale_type;
        });
    TT_FATAL(scale_exists, "LWT inline terminal scale is missing from the forward plan");
    return TerminalScaleInline{
        .predict_update_route_index = predict_update_route_index,
        .scale_type = scale_type,
        .final_storage = final_storage,
    };
}

[[nodiscard]] inline std::vector<RequiredStreams> backpropagate_requirements(
    const LiftingForwardPlan& plan,
    const IndexInterval final_even,
    const IndexInterval final_odd,
    const size_t closure_extent = 0) {
    const auto close_interval = [closure_extent](const IndexInterval interval, const size_t stream_length) {
        if (closure_extent == 0 || interval.empty()) {
            return interval;
        }
        const size_t begin = tt::round_down(interval.begin, closure_extent);
        const size_t rounded_end = interval.end > std::numeric_limits<size_t>::max() - (closure_extent - 1)
                                       ? stream_length
                                       : tt::round_up(interval.end, closure_extent);
        return IndexInterval{.begin = begin, .end = std::min(rounded_end, stream_length)};
    };
    std::vector<RequiredStreams> required(plan.routes.size() + 1);

    size_t even_length = plan.final_even_length;
    size_t odd_length = plan.final_odd_length;
    validate_interval(final_even, even_length, "final even");
    validate_interval(final_odd, odd_length, "final odd");
    required.back() = RequiredStreams{
        .even = close_interval(final_even, even_length),
        .odd = close_interval(final_odd, odd_length),
    };

    for (size_t reverse_index = plan.routes.size(); reverse_index > 0; --reverse_index) {
        const size_t route_index = reverse_index - 1;
        const auto& route = plan.routes[route_index];
        const RequiredStreams after = required[route_index + 1];
        validate_interval(after.even, even_length, "required even after route");
        validate_interval(after.odd, odd_length, "required odd after route");

        RequiredStreams before{};
        switch (route.type) {
            case StepType::kPredict: {
                TT_FATAL(
                    even_length == route.source_length && odd_length == route.output_length,
                    "Predict route length state is inconsistent with the forward plan");
                const uint32_t k = coefficient_count(route);
                before.even = hull(after.even, translated(after.odd, route.source_offset, static_cast<size_t>(k - 1)));
                before.odd = translated(after.odd, route.base_offset);
                even_length = route.source_length;
                odd_length = route.base_length;
                break;
            }
            case StepType::kUpdate: {
                TT_FATAL(
                    odd_length == route.source_length && even_length == route.output_length,
                    "Update route length state is inconsistent with the forward plan");
                const uint32_t k = coefficient_count(route);
                before.even = translated(after.even, route.base_offset);
                before.odd = hull(after.odd, translated(after.even, route.source_offset, static_cast<size_t>(k - 1)));
                even_length = route.base_length;
                odd_length = route.source_length;
                break;
            }
            case StepType::kScaleEven: {
                TT_FATAL(even_length == route.output_length, "Scale-even output length is inconsistent");
                before.even = translated(after.even, route.source_offset);
                before.odd = after.odd;
                even_length = route.source_length;
                break;
            }
            case StepType::kScaleOdd: {
                TT_FATAL(odd_length == route.output_length, "Scale-odd output length is inconsistent");
                before.even = after.even;
                before.odd = translated(after.odd, route.source_offset);
                odd_length = route.source_length;
                break;
            }
            case StepType::kSwap: {
                TT_FATAL(
                    even_length == route.base_length && odd_length == route.source_length,
                    "Swap route length state is inconsistent with the forward plan");
                before.even = after.odd;
                before.odd = after.even;
                even_length = route.source_length;
                odd_length = route.base_length;
                break;
            }
        }

        before.even = close_interval(before.even, even_length);
        before.odd = close_interval(before.odd, odd_length);
        validate_interval(before.even, even_length, "required even before route");
        validate_interval(before.odd, odd_length, "required odd before route");
        required[route_index] = before;
    }

    TT_FATAL(
        even_length == plan.preprocess_layout.output.even.length &&
            odd_length == plan.preprocess_layout.output.odd.length,
        "Backward dependency propagation did not reach the initial split-stream lengths");
    return required;
}

[[nodiscard]] inline size_t local_offset(const IndexInterval storage, const IndexInterval required) {
    TT_FATAL(contains(storage, required), "LWT route requires data outside its local workspace");
    return required.empty() ? 0 : required.begin - storage.begin;
}

[[nodiscard]] inline LwtChunkPlan build_chunk(
    const LiftingForwardPlan& plan,
    const IndexInterval final_even,
    const IndexInterval final_odd,
    const size_t final_even_output_origin,
    const size_t final_odd_output_origin) {
    const std::vector<RequiredStreams> required = backpropagate_requirements(plan, final_even, final_odd);
    const TerminalScaleInline inline_scale = terminal_scale_inline(plan);
    StoredStream active_even{.slot = StorageSlot::kA, .storage = required.front().even};
    StoredStream active_odd{.slot = StorageSlot::kB, .storage = required.front().odd};
    StorageSlot free_slot = StorageSlot::kScratch;
    size_t max_workspace_elements = std::max(required.front().even.length(), required.front().odd.length());

    std::vector<LwtStepRoute> routes;
    routes.reserve(plan.routes.size());
    for (size_t route_index = 0; route_index < plan.routes.size(); ++route_index) {
        const auto& full_route = plan.routes[route_index];
        const RequiredStreams& after = required[route_index + 1];

        if (full_route.type == StepType::kSwap) {
            std::swap(active_even, active_odd);
            continue;
        }

        if (full_route.type == StepType::kPredict || full_route.type == StepType::kUpdate) {
            const bool predict = full_route.type == StepType::kPredict;
            const bool inline_scale_route = route_index == inline_scale.predict_update_route_index;
            const uint32_t k = coefficient_count(full_route);
            const IndexInterval output = predict ? after.odd : after.even;
            const IndexInterval source_required =
                translated(output, full_route.source_offset, static_cast<size_t>(k - 1));
            const IndexInterval base_required = translated(output, full_route.base_offset);
            const StoredStream& source = predict ? active_even : active_odd;
            const StoredStream& base = predict ? active_odd : active_even;
            const RouteOutputRef output_ref =
                inline_scale_route ? RouteOutputRef{.storage = inline_scale.final_storage, .slot = free_slot}
                                   : detail::workspace_output(free_slot);

            routes.push_back(LwtStepRoute{
                .type = full_route.type,
                .source = StreamRef{.slot = source.slot},
                .base = StreamRef{.slot = base.slot},
                .output = output_ref,
                .source_storage_length = source.storage.length(),
                .base_storage_length = base.storage.length(),
                .source_offset_elements = local_offset(source.storage, source_required),
                .base_offset_elements = local_offset(base.storage, base_required),
                .source_left_pad_elements = full_route.source_left_pad,
                .output_length = output.length(),
                .output_offset_elements = inline_scale_route && !output.empty() ? output.begin : 0,
            });

            const StoredStream replacement{.slot = free_slot, .storage = output};
            max_workspace_elements = std::max(max_workspace_elements, output.length());
            if (predict) {
                free_slot = active_odd.slot;
                active_odd = replacement;
            } else {
                free_slot = active_even.slot;
                active_even = replacement;
            }
            continue;
        }

        const bool scale_even = full_route.type == StepType::kScaleEven;
        TT_FATAL(scale_even || full_route.type == StepType::kScaleOdd, "Unsupported LWT route type");
        if (full_route.type == inline_scale.scale_type) {
            continue;
        }
        const StoredStream& source = scale_even ? active_even : active_odd;
        const IndexInterval output = scale_even ? after.even : after.odd;
        const auto final_storage = scale_even ? RouteOutputStorage::kFinalEvenDram : RouteOutputStorage::kFinalOddDram;
        routes.push_back(LwtStepRoute{
            .type = full_route.type,
            .source = StreamRef{.slot = source.slot},
            .base = StreamRef{.slot = source.slot},
            .output = RouteOutputRef{.storage = final_storage, .slot = source.slot},
            .source_storage_length = source.storage.length(),
            .base_storage_length = source.storage.length(),
            .source_offset_elements = local_offset(source.storage, output),
            .base_offset_elements = local_offset(source.storage, output),
            .source_left_pad_elements = 0,
            .output_length = output.length(),
            .output_offset_elements = output.empty() ? 0 : output.begin,
        });
    }

    for (auto& route : routes) {
        if (route.output.storage == RouteOutputStorage::kFinalEvenDram) {
            TT_FATAL(
                route.output_offset_elements >= final_even_output_origin,
                "LWT final-even route starts before the canonical interval");
            route.output_offset_elements -= final_even_output_origin;
        } else if (route.output.storage == RouteOutputStorage::kFinalOddDram) {
            TT_FATAL(
                route.output_offset_elements >= final_odd_output_origin,
                "LWT final-odd route starts before the canonical interval");
            route.output_offset_elements -= final_odd_output_origin;
        }
    }

    const IndexInterval initial_even = required.front().even;
    const IndexInterval initial_odd = required.front().odd;
    const size_t final_elements = final_even.length() + final_odd.length();
    const size_t dependency_elements = initial_even.length() + initial_odd.length();
    const double dependency_overhead =
        final_elements == 0 ? 0.0
                            : static_cast<double>(dependency_elements - std::min(dependency_elements, final_elements)) /
                                  static_cast<double>(final_elements);

    return LwtChunkPlan{
        .initial_even = initial_even,
        .initial_odd = initial_odd,
        .routes = std::move(routes),
        .max_workspace_elements = max_workspace_elements,
        .dependency_overhead = dependency_overhead,
    };
}

[[nodiscard]] inline std::vector<LwtChunkPlan> build_chunks(
    const LiftingForwardPlan& plan, const uint32_t requested_chunk_count) {
    TT_FATAL(requested_chunk_count > 0, "LWT chunk count must be non-zero");
    const int64_t canonical_start = static_cast<int64_t>(plan.preprocess_layout.pad_config.left + 1) / 2;
    const int64_t signed_even_origin = canonical_start - plan.final_even_shift;
    const int64_t signed_odd_origin = canonical_start - plan.final_odd_shift;
    TT_FATAL(signed_even_origin >= 0 && signed_odd_origin >= 0, "LWT canonical output requires a negative origin");
    const size_t final_even_origin = static_cast<size_t>(signed_even_origin);
    const size_t final_odd_origin = static_cast<size_t>(signed_odd_origin);
    TT_FATAL(
        final_even_origin + plan.output_length <= plan.final_even_length &&
            final_odd_origin + plan.output_length <= plan.final_odd_length,
        "LWT terminal streams do not cover the canonical output interval");
    const size_t max_final_length = plan.output_length;
    const size_t final_group_count =
        std::max(ceil_div(max_final_length, static_cast<size_t>(device_protocol::kLwtGroupOutputElements)), size_t{1});
    const size_t chunk_count = std::min(static_cast<size_t>(requested_chunk_count), final_group_count);
    const size_t base_groups = final_group_count / chunk_count;
    const size_t extra_groups = final_group_count % chunk_count;

    std::vector<LwtChunkPlan> chunks;
    chunks.reserve(chunk_count);
    size_t group_begin = 0;
    for (size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        const size_t group_count = base_groups + (chunk_index < extra_groups ? 1 : 0);
        const size_t begin = group_begin * device_protocol::kLwtGroupOutputElements;
        const size_t end =
            std::min((group_begin + group_count) * device_protocol::kLwtGroupOutputElements, max_final_length);
        chunks.push_back(build_chunk(
            plan,
            IndexInterval{.begin = begin + final_even_origin, .end = end + final_even_origin},
            IndexInterval{.begin = begin + final_odd_origin, .end = end + final_odd_origin},
            final_even_origin,
            final_odd_origin));
        group_begin += group_count;
    }
    TT_FATAL(group_begin == final_group_count, "LWT chunks do not cover every final output group");
    return chunks;
}

}  // namespace execution_detail

[[nodiscard]] inline AxisConePlan build_axis_cone(
    const LiftingForwardPlan& plan,
    const IndexInterval final_even,
    const IndexInterval final_odd,
    const size_t closure_extent = 0) {
    const std::vector<AxisRequiredStreams> required =
        execution_detail::backpropagate_requirements(plan, final_even, final_odd, closure_extent);

    std::vector<AxisRouteRequirement> routes;
    routes.reserve(plan.routes.size());
    for (size_t route_index = 0; route_index < plan.routes.size(); ++route_index) {
        const LiftingStepRoute& route = plan.routes[route_index];
        const AxisRequiredStreams before = required[route_index];
        const AxisRequiredStreams after = required[route_index + 1];
        AxisRouteRequirement requirement{
            .type = route.type,
            .before = before,
            .after = after,
        };

        switch (route.type) {
            case StepType::kPredict: {
                const uint32_t k = execution_detail::coefficient_count(route);
                requirement.output = after.odd;
                requirement.source =
                    execution_detail::translated(requirement.output, route.source_offset, static_cast<size_t>(k - 1));
                requirement.base = execution_detail::translated(requirement.output, route.base_offset);
                break;
            }
            case StepType::kUpdate: {
                const uint32_t k = execution_detail::coefficient_count(route);
                requirement.output = after.even;
                requirement.source =
                    execution_detail::translated(requirement.output, route.source_offset, static_cast<size_t>(k - 1));
                requirement.base = execution_detail::translated(requirement.output, route.base_offset);
                break;
            }
            case StepType::kScaleEven:
                requirement.output = after.even;
                requirement.source = execution_detail::translated(requirement.output, route.source_offset);
                requirement.base = requirement.source;
                break;
            case StepType::kScaleOdd:
                requirement.output = after.odd;
                requirement.source = execution_detail::translated(requirement.output, route.source_offset);
                requirement.base = requirement.source;
                break;
            case StepType::kSwap: break;
        }
        routes.push_back(requirement);
    }

    return AxisConePlan{
        .final_even = required.back().even,
        .final_odd = required.back().odd,
        .initial_even = required.front().even,
        .initial_odd = required.front().odd,
        .routes = std::move(routes),
    };
}

[[nodiscard]] inline LwtExecutionPlan make_lwt_execution_plan(
    LiftingForwardPlan full_plan,
    const uint32_t core_limit,
    const uint32_t l1_signal_budget_bytes,
    const WorkspaceLayout workspace_layout = WorkspaceLayout::kRowMajor) {
    TT_FATAL(core_limit > 0, "LWT requires at least one worker core");
    TT_FATAL(l1_signal_budget_bytes >= 3 * device_protocol::kStickBytes, "LWT L1 budget is too small");

    const size_t max_final_length = full_plan.output_length;
    const uint32_t final_group_count = static_cast<uint32_t>(
        std::max(ceil_div(max_final_length, static_cast<size_t>(device_protocol::kLwtGroupOutputElements)), size_t{1}));
    uint32_t chunk_count = std::min(final_group_count, core_limit);
    std::vector<LwtChunkPlan> chunks;
    uint32_t workspace_elements = 0;
    uint32_t max_workspace_elements = 0;

    const auto build_candidate = [&](const uint32_t candidate_chunk_count) {
        auto candidate_chunks = execution_detail::build_chunks(full_plan, candidate_chunk_count);
        size_t candidate_max_workspace_elements = 0;
        for (const auto& chunk : candidate_chunks) {
            candidate_max_workspace_elements = std::max(candidate_max_workspace_elements, chunk.max_workspace_elements);
        }
        const size_t workspace_alignment = workspace_layout == WorkspaceLayout::kTileNative
                                               ? static_cast<size_t>(device_protocol::kLwtGroupOutputElements)
                                               : static_cast<size_t>(kStickWidth);
        const size_t aligned_workspace = tt::round_up(candidate_max_workspace_elements, workspace_alignment);
        TT_FATAL(
            aligned_workspace <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
            "LWT workspace length {} overflows uint32_t",
            aligned_workspace);
        return std::tuple{
            std::move(candidate_chunks),
            static_cast<uint32_t>(aligned_workspace),
            static_cast<uint32_t>(candidate_max_workspace_elements)};
    };

    for (;;) {
        auto candidate = build_candidate(chunk_count);
        chunks = std::move(std::get<0>(candidate));
        workspace_elements = std::get<1>(candidate);
        max_workspace_elements = std::get<2>(candidate);
        const uint64_t workspace_bytes_per_core = uint64_t{3} * workspace_elements * sizeof(float);
        if (workspace_bytes_per_core <= l1_signal_budget_bytes) {
            break;
        }
        TT_FATAL(
            chunk_count < final_group_count,
            "One-group LWT workspace requires {} bytes/core, exceeding the {}-byte L1 signal budget",
            workspace_bytes_per_core,
            l1_signal_budget_bytes);
        chunk_count =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(final_group_count), uint64_t{2} * chunk_count));
    }

    const uint32_t groups_per_chunk =
        static_cast<uint32_t>(ceil_div(static_cast<size_t>(final_group_count), chunks.size()));
    const auto max_dependency =
        std::max_element(chunks.begin(), chunks.end(), [](const LwtChunkPlan& lhs, const LwtChunkPlan& rhs) {
            return lhs.dependency_overhead < rhs.dependency_overhead;
        });

    return LwtExecutionPlan{
        .full_plan = std::move(full_plan),
        .chunks = std::move(chunks),
        .groups_per_chunk = groups_per_chunk,
        .workspace_elements = workspace_elements,
        .max_workspace_elements = max_workspace_elements,
        .max_dependency_overhead = max_dependency->dependency_overhead,
        .workspace_layout = workspace_layout,
    };
}

}  // namespace ttnn::operations::wavelet
