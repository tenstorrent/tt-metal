// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include "tt-metalium/math.hpp"
#include "ttnn/operations/wavelet/common/storage_contract.hpp"
#include "ttnn/operations/wavelet/planner/inverse_plan.hpp"
#include "ttnn/operations/wavelet/planner/plan_2d.hpp"

namespace ttnn::operations::wavelet {

struct Ilwt2DExecutionPlan {
    LiftingInversePlan y_plan{};
    LiftingInversePlan x_plan{};
    Lwt2DTilingContract tiling{};
    size_t output_height{0};
    size_t output_width{0};
    size_t band_height{0};
    size_t band_width{0};
    std::vector<Lwt2DChunkPlan> chunks;
    std::array<uint32_t, 5> allocated_plane_widths_elements{};
    std::array<uint64_t, 5> allocated_plane_slot_bytes{};
    uint64_t allocated_l1_bytes{0};
};

namespace inverse_2d_detail {

[[nodiscard]] inline IndexInterval reconstructed_parity_interval(
    const IndexInterval output, const size_t pad, const bool even) {
    TT_FATAL(output.end <= std::numeric_limits<size_t>::max() - pad, "2D ILWT padded output interval overflows");
    const size_t padded_begin = output.begin + pad;
    const size_t padded_end = output.end + pad;
    return even ? IndexInterval{.begin = ceil_div(padded_begin, size_t{2}), .end = ceil_div(padded_end, size_t{2})}
                : IndexInterval{.begin = padded_begin / 2, .end = padded_end / 2};
}

[[nodiscard]] inline AxisConePlan build_axis_cone(
    const LiftingInversePlan& inverse_plan, const IndexInterval target_even, const IndexInterval target_odd) {
    const LiftingForwardPlan& forward = inverse_plan.forward_trace;
    const std::vector<AxisRequiredStreams> required =
        inverse_detail::propagate_requirements(forward, target_even, target_odd);

    AxisConePlan cone{
        .final_even = target_even,
        .final_odd = target_odd,
        .initial_even = required.back().even,
        .initial_odd = required.back().odd,
        .routes = {},
    };
    cone.routes.reserve(forward.routes.size());

    for (size_t reverse_index = forward.routes.size(); reverse_index > 0; --reverse_index) {
        const size_t forward_index = reverse_index - 1;
        const LiftingStepRoute& route = forward.routes[forward_index];
        const AxisRequiredStreams current = required[forward_index + 1];
        const AxisRequiredStreams reconstructed = required[forward_index];
        AxisRouteRequirement requirement{
            .type = route.type,
            .before = current,
            .after = reconstructed,
        };

        if (route.type == StepType::kSwap || is_scale_step(route.type)) {
            cone.routes.push_back(requirement);
            continue;
        }

        const bool predict = route.type == StepType::kPredict;
        TT_FATAL(predict || route.type == StepType::kUpdate, "Unsupported 2D inverse route type");
        const IndexInterval output = predict ? reconstructed.odd : reconstructed.even;
        const IndexInterval base =
            inverse_detail::subtract_offset(output, route.base_offset, "2D inverse predict/update");
        const uint32_t k = execution_detail::coefficient_count(route);
        requirement.source = execution_detail::translated(base, route.source_offset, static_cast<size_t>(k - 1));
        requirement.base = base;
        requirement.output = output;
        cone.routes.push_back(requirement);
    }
    return cone;
}

inline void append_axis_routes(
    const AxisConePlan& cone,
    const Lwt2DAxis axis,
    const IndexInterval transverse,
    plan_2d_detail::AxisPairSlots& slots,
    std::vector<Lwt2DRoutePlan>& routes) {
    const size_t route_count = cone.routes.size();
    for (size_t inverse_index = 0; inverse_index < route_count; ++inverse_index) {
        const AxisRouteRequirement& requirement = cone.routes[inverse_index];
        const size_t forward_index = route_count - inverse_index - 1;
        if (requirement.type == StepType::kSwap) {
            routes.push_back(Lwt2DRoutePlan{
                .axis = axis,
                .axis_route_index = forward_index,
                .type = requirement.type,
                .source_slot = slots.even,
                .base_slot = slots.odd,
                .output_slot = slots.even,
                .source = plan_2d_detail::axis_rectangle(axis, requirement.before.even, transverse),
                .base = plan_2d_detail::axis_rectangle(axis, requirement.before.odd, transverse),
                .output = {},
            });
            std::swap(slots.even, slots.odd);
            continue;
        }
        if (is_scale_step(requirement.type)) {
            const bool even = requirement.type == StepType::kScaleEven;
            const Lwt2DPlaneSlot slot = even ? slots.even : slots.odd;
            routes.push_back(Lwt2DRoutePlan{
                .axis = axis,
                .axis_route_index = forward_index,
                .type = requirement.type,
                .source_slot = slot,
                .base_slot = slot,
                .output_slot = slot,
                .source = plan_2d_detail::axis_rectangle(
                    axis, even ? requirement.before.even : requirement.before.odd, transverse),
                .base = {},
                .output = {},
            });
            continue;
        }

        const bool predict = requirement.type == StepType::kPredict;
        const Lwt2DPlaneSlot source_slot = predict ? slots.even : slots.odd;
        const Lwt2DPlaneSlot base_slot = predict ? slots.odd : slots.even;
        const Lwt2DPlaneSlot output_slot = slots.free;
        routes.push_back(Lwt2DRoutePlan{
            .axis = axis,
            .axis_route_index = forward_index,
            .type = requirement.type,
            .source_slot = source_slot,
            .base_slot = base_slot,
            .output_slot = output_slot,
            .source = plan_2d_detail::axis_rectangle(axis, requirement.source, transverse),
            .base = plan_2d_detail::axis_rectangle(axis, requirement.base, transverse),
            .output = plan_2d_detail::axis_rectangle(axis, requirement.output, transverse),
        });
        if (predict) {
            slots.free = slots.odd;
            slots.odd = output_slot;
        } else {
            slots.free = slots.even;
            slots.even = output_slot;
        }
    }
}

[[nodiscard]] inline std::pair<std::vector<Lwt2DRoutePlan>, Lwt2DBandSlots> build_route_schedule(
    const AxisConePlan& y_cone, const AxisConePlan& x_cone) {
    std::vector<Lwt2DRoutePlan> routes;
    routes.reserve(2 * x_cone.routes.size() + 2 * y_cone.routes.size());

    // LL/LH -> Le/Lo.
    plan_2d_detail::AxisPairSlots vertical_low{
        .even = Lwt2DPlaneSlot::kP0,
        .odd = Lwt2DPlaneSlot::kP1,
        .free = Lwt2DPlaneSlot::kScratch,
    };
    append_axis_routes(x_cone, Lwt2DAxis::kHorizontal, y_cone.initial_even, vertical_low, routes);

    // HL/HH -> He/Ho.
    plan_2d_detail::AxisPairSlots vertical_high{
        .even = Lwt2DPlaneSlot::kP2,
        .odd = Lwt2DPlaneSlot::kP3,
        .free = vertical_low.free,
    };
    append_axis_routes(x_cone, Lwt2DAxis::kHorizontal, y_cone.initial_odd, vertical_high, routes);

    // Le/He -> EE/OE.
    plan_2d_detail::AxisPairSlots horizontal_even{
        .even = vertical_low.even,
        .odd = vertical_high.even,
        .free = vertical_high.free,
    };
    append_axis_routes(y_cone, Lwt2DAxis::kVertical, x_cone.final_even, horizontal_even, routes);

    // Lo/Ho -> EO/OO.
    plan_2d_detail::AxisPairSlots horizontal_odd{
        .even = vertical_low.odd,
        .odd = vertical_high.odd,
        .free = horizontal_even.free,
    };
    append_axis_routes(y_cone, Lwt2DAxis::kVertical, x_cone.final_odd, horizontal_odd, routes);

    const Lwt2DBandSlots parity{
        .ll = horizontal_even.even,
        .lh = horizontal_odd.even,
        .hl = horizontal_even.odd,
        .hh = horizontal_odd.odd,
    };
    std::array<Lwt2DPlaneSlot, 4> slots = {parity.ll, parity.lh, parity.hl, parity.hh};
    std::sort(slots.begin(), slots.end());
    TT_FATAL(
        std::adjacent_find(slots.begin(), slots.end()) == slots.end(),
        "2D ILWT route schedule aliases two final parity planes");
    return {std::move(routes), parity};
}

[[nodiscard]] inline Lwt2DChunkPlan build_chunk(
    const LiftingInversePlan& y_plan, const LiftingInversePlan& x_plan, const IndexRectangle output) {
    const size_t pad_y = y_plan.forward_trace.preprocess_layout.pad_config.left;
    const size_t pad_x = x_plan.forward_trace.preprocess_layout.pad_config.left;
    const AxisConePlan y_cone = build_axis_cone(
        y_plan,
        reconstructed_parity_interval(output.y, pad_y, true),
        reconstructed_parity_interval(output.y, pad_y, false));
    const AxisConePlan x_cone = build_axis_cone(
        x_plan,
        reconstructed_parity_interval(output.x, pad_x, true),
        reconstructed_parity_interval(output.x, pad_x, false));

    const PolyphaseDependencyRectangles initial{
        .ee = interval_product(y_cone.initial_even, x_cone.initial_even),
        .eo = interval_product(y_cone.initial_even, x_cone.initial_odd),
        .oe = interval_product(y_cone.initial_odd, x_cone.initial_even),
        .oo = interval_product(y_cone.initial_odd, x_cone.initial_odd),
    };
    auto [routes, parity_slots] = build_route_schedule(y_cone, x_cone);
    const Lwt2DResourceModel resources = plan_2d_detail::make_resource_model(initial, routes);
    const Lwt2DBandSourceRectangles parity_sources{
        .ll = interval_product(y_cone.final_even, x_cone.final_even),
        .lh = interval_product(y_cone.final_even, x_cone.final_odd),
        .hl = interval_product(y_cone.final_odd, x_cone.final_even),
        .hh = interval_product(y_cone.final_odd, x_cone.final_odd),
    };
    const uint64_t output_elements = output.area();
    const uint64_t dependency_elements = initial.total_area();
    const double dependency_overhead =
        output_elements == 0
            ? 0.0
            : static_cast<double>(dependency_elements - std::min(dependency_elements, output_elements)) /
                  static_cast<double>(output_elements);

    return Lwt2DChunkPlan{
        .final_band_rect = output,
        .execution_band_rect =
            IndexRectangle{
                .y =
                    IndexInterval{
                        .begin = tt::round_down(output.y.begin, kTileHeight2D),
                        .end = tt::round_up(output.y.end, kTileHeight2D),
                    },
                .x =
                    IndexInterval{
                        .begin = tt::round_down(output.x.begin, kTileWidth2D),
                        .end = tt::round_up(output.x.end, kTileWidth2D),
                    },
            },
        .initial = initial,
        .routes = std::move(routes),
        .final_bands = parity_slots,
        .final_band_sources = parity_sources,
        .resources = resources,
        .dependency_overhead = dependency_overhead,
    };
}

[[nodiscard]] inline std::vector<Lwt2DChunkPlan> build_chunks(
    const LiftingInversePlan& y_plan,
    const LiftingInversePlan& x_plan,
    const uint32_t chunk_tiles_y,
    const uint32_t chunk_tiles_x) {
    const size_t chunk_height = static_cast<size_t>(chunk_tiles_y) * kTileHeight2D;
    const size_t chunk_width = static_cast<size_t>(chunk_tiles_x) * kTileWidth2D;
    std::vector<Lwt2DChunkPlan> chunks;
    const size_t chunk_rows = ceil_div(y_plan.original_length, chunk_height);
    const size_t chunk_columns = ceil_div(x_plan.original_length, chunk_width);
    chunks.reserve(plan_2d_detail::checked_area(chunk_rows, chunk_columns, "2D ILWT chunk grid"));
    for (size_t y = 0; y < y_plan.original_length; y += chunk_height) {
        for (size_t x = 0; x < x_plan.original_length; x += chunk_width) {
            chunks.push_back(build_chunk(
                y_plan,
                x_plan,
                IndexRectangle{
                    .y = IndexInterval{.begin = y, .end = std::min(y + chunk_height, y_plan.original_length)},
                    .x = IndexInterval{.begin = x, .end = std::min(x + chunk_width, x_plan.original_length)},
                }));
        }
    }
    return chunks;
}

[[nodiscard]] inline plan_2d_detail::AxisConeSignature make_inverse_axis_signature(
    const LiftingInversePlan& plan, const IndexInterval output, const size_t tile_extent) {
    const size_t pad = plan.forward_trace.preprocess_layout.pad_config.left;
    const AxisConePlan cone = build_axis_cone(
        plan, reconstructed_parity_interval(output, pad, true), reconstructed_parity_interval(output, pad, false));
    return plan_2d_detail::make_axis_cone_signature(output, cone, nullptr, tile_extent);
}

}  // namespace inverse_2d_detail

[[nodiscard]] inline Ilwt2DExecutionPlan make_ilwt_2d_execution_plan(
    LiftingInversePlan y_plan,
    LiftingInversePlan x_plan,
    const uint32_t core_limit,
    const uint64_t l1_budget_bytes,
    const uint64_t inverse_coordination_penalty_cycles_per_core = 0) {
    TT_FATAL(core_limit > 0, "2D ILWT requires at least one worker core");
    TT_FATAL(y_plan.original_length > 0 && x_plan.original_length > 0, "2D ILWT output shape must be positive");
    TT_FATAL(
        y_plan.original_length <= kMax2DLogicalExtent && x_plan.original_length <= kMax2DLogicalExtent,
        "2D ILWT output dimensions {}x{} exceed the signed device-coordinate limit {}",
        y_plan.original_length,
        x_plan.original_length,
        kMax2DLogicalExtent);

    const uint32_t output_tiles_y =
        plan_2d_detail::checked_u32(ceil_div(y_plan.original_length, kTileHeight2D), "2D ILWT output tile rows");
    const uint32_t output_tiles_x =
        plan_2d_detail::checked_u32(ceil_div(x_plan.original_length, kTileWidth2D), "2D ILWT output tile columns");
    const uint64_t maximum_tile_area = plan_2d_detail::maximum_candidate_tile_area(l1_budget_bytes);
    const uint32_t maximum_tiles_y = static_cast<uint32_t>(std::min<uint64_t>(output_tiles_y, maximum_tile_area));
    const uint32_t maximum_tiles_x = static_cast<uint32_t>(std::min<uint64_t>(output_tiles_x, maximum_tile_area));
    const auto y_candidate_classes = plan_2d_detail::make_axis_candidate_classes(
        y_plan.original_length, maximum_tiles_y, kTileHeight2D, [&](const IndexInterval output) {
            return inverse_2d_detail::make_inverse_axis_signature(y_plan, output, kTileHeight2D);
        });
    const auto x_candidate_classes = plan_2d_detail::make_axis_candidate_classes(
        x_plan.original_length, maximum_tiles_x, kTileWidth2D, [&](const IndexInterval output) {
            return inverse_2d_detail::make_inverse_axis_signature(x_plan, output, kTileWidth2D);
        });
    plan_2d_detail::Candidate best{};
    bool found = false;
    for (uint32_t tiles_y = 1; tiles_y <= maximum_tiles_y; ++tiles_y) {
        const uint32_t candidate_tiles_x = static_cast<uint32_t>(
            std::min<uint64_t>(maximum_tiles_x, maximum_tile_area / static_cast<uint64_t>(tiles_y)));
        for (uint32_t tiles_x = 1; tiles_x <= candidate_tiles_x; ++tiles_x) {
            std::optional<plan_2d_detail::Candidate> candidate = plan_2d_detail::evaluate_candidate(
                y_candidate_classes[tiles_y],
                x_candidate_classes[tiles_x],
                tiles_y,
                tiles_x,
                core_limit,
                l1_budget_bytes,
                [&](const IndexRectangle output) { return inverse_2d_detail::build_chunk(y_plan, x_plan, output); },
                [&](const Lwt2DChunkPlan& chunk) {
                    return plan_2d_detail::estimate_chunk_latency_cycles(
                        chunk, y_plan.forward_trace, x_plan.forward_trace, true);
                },
                inverse_coordination_penalty_cycles_per_core);
            if (!candidate.has_value()) {
                continue;
            }
            if (!found || plan_2d_detail::is_better_candidate(*candidate, best, true)) {
                best = std::move(*candidate);
                found = true;
            }
        }
    }
    TT_FATAL(found, "No 2D ILWT chunk fits the {}-byte L1 budget", l1_budget_bytes);
    best.chunks = inverse_2d_detail::build_chunks(y_plan, x_plan, best.chunk_tiles_y, best.chunk_tiles_x);

    std::array<uint32_t, 5> heights{};
    std::array<uint32_t, 5> widths{};
    std::array<uint64_t, 5> slot_bytes{};
    uint64_t workspace_bytes = 0;
    for (size_t slot = 0; slot < slot_bytes.size(); ++slot) {
        for (const Lwt2DChunkPlan& chunk : best.chunks) {
            heights[slot] = std::max(heights[slot], chunk.resources.plane_heights_elements[slot]);
            widths[slot] = std::max(widths[slot], chunk.resources.plane_widths_elements[slot]);
        }
        slot_bytes[slot] = plan_2d_detail::checked_bytes(
            plan_2d_detail::checked_area(heights[slot], widths[slot], "2D ILWT plane"), "2D ILWT plane");
        workspace_bytes += slot_bytes[slot];
    }
    constexpr uint64_t fixed_bytes =
        plan_2d_detail::kCircularBufferBytes + plan_2d_detail::kMetadataBytes + plan_2d_detail::kSynchronizationBytes;
    const uint64_t total_l1_bytes = workspace_bytes + fixed_bytes;
    TT_FATAL(total_l1_bytes <= l1_budget_bytes, "2D ILWT uniform L1 allocation exceeds its budget");

    const Lwt2DTilingContract tiling{
        .input = TiledShape2D::from_logical(Shape2D{.height = y_plan.original_length, .width = x_plan.original_length}),
        .band = TiledShape2D::from_logical(
            Shape2D{.height = y_plan.coefficient_length, .width = x_plan.coefficient_length}),
    };
    return Ilwt2DExecutionPlan{
        .y_plan = std::move(y_plan),
        .x_plan = std::move(x_plan),
        .tiling = tiling,
        .output_height = tiling.input.logical.height,
        .output_width = tiling.input.logical.width,
        .band_height = tiling.band.logical.height,
        .band_width = tiling.band.logical.width,
        .chunks = std::move(best.chunks),
        .allocated_plane_widths_elements = widths,
        .allocated_plane_slot_bytes = slot_bytes,
        .allocated_l1_bytes = total_l1_bytes,
    };
}

template <typename Scheme>
[[nodiscard]] Ilwt2DExecutionPlan make_ilwt_2d_execution_plan(
    const size_t output_height,
    const size_t output_width,
    const uint32_t core_limit,
    const uint64_t l1_budget_bytes,
    const BoundaryMode boundary_mode = BoundaryMode::kSymmetric,
    const uint64_t inverse_coordination_penalty_cycles_per_core = 0) {
    TT_FATAL(output_height > 0 && output_width > 0, "2D ILWT output dimensions must be positive");
    TT_FATAL(
        output_height <= kMax2DLogicalExtent && output_width <= kMax2DLogicalExtent,
        "2D ILWT output dimensions {}x{} exceed the signed device-coordinate limit {}",
        output_height,
        output_width,
        kMax2DLogicalExtent);
    const SignalBuffer y_signal{
        .length = output_height, .stick_width = kStickWidth, .element_size_bytes = sizeof(float)};
    const SignalBuffer x_signal{
        .length = output_width, .stick_width = kStickWidth, .element_size_bytes = sizeof(float)};
    LiftingForwardPlan y_forward = make_forward_lifting_plan<Scheme>(y_signal, boundary_mode);
    LiftingForwardPlan x_forward = make_forward_lifting_plan<Scheme>(x_signal, boundary_mode);
    const size_t y_coefficients = y_forward.output_length;
    const size_t x_coefficients = x_forward.output_length;
    return make_ilwt_2d_execution_plan(
        LiftingInversePlan{
            .forward_trace = std::move(y_forward),
            .original_length = output_height,
            .coefficient_length = y_coefficients,
        },
        LiftingInversePlan{
            .forward_trace = std::move(x_forward),
            .original_length = output_width,
            .coefficient_length = x_coefficients,
        },
        core_limit,
        l1_budget_bytes,
        inverse_coordination_penalty_cycles_per_core);
}

[[nodiscard]] inline std::vector<uint32_t> build_ilwt_2d_chunk_config_words(const Ilwt2DExecutionPlan& plan) {
    TT_FATAL(!plan.chunks.empty(), "2D ILWT chunk protocol requires at least one chunk");
    std::vector<uint32_t> words(plan.chunks.size() * device_protocol::kLwt2DChunkConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const Lwt2DChunkPlan& chunk = plan.chunks[chunk_index];
        const size_t offset = chunk_index * device_protocol::kLwt2DChunkConfigWordCount;
        words[offset + device_protocol::kLwt2DFinalYBegin] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.y.begin, "2D ILWT final y begin");
        words[offset + device_protocol::kLwt2DFinalYLength] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.height(), "2D ILWT final height");
        words[offset + device_protocol::kLwt2DFinalXBegin] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.x.begin, "2D ILWT final x begin");
        words[offset + device_protocol::kLwt2DFinalXLength] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.width(), "2D ILWT final width");
        words[offset + device_protocol::kLwt2DExecutionTileYBegin] =
            plan_2d_detail::checked_u32(chunk.execution_band_rect.y.begin / kTileHeight2D, "2D ILWT execution tile y");
        words[offset + device_protocol::kLwt2DExecutionTileYCount] =
            plan_2d_detail::checked_u32(chunk.execution_band_rect.height() / kTileHeight2D, "2D ILWT execution rows");
        words[offset + device_protocol::kLwt2DExecutionTileXBegin] =
            plan_2d_detail::checked_u32(chunk.execution_band_rect.x.begin / kTileWidth2D, "2D ILWT execution tile x");
        words[offset + device_protocol::kLwt2DExecutionTileXCount] =
            plan_2d_detail::checked_u32(chunk.execution_band_rect.width() / kTileWidth2D, "2D ILWT execution columns");
        plan_2d_detail::write_protocol_rectangle(words, offset + device_protocol::kLwt2DInitialEe, chunk.initial.ee);
        plan_2d_detail::write_protocol_rectangle(words, offset + device_protocol::kLwt2DInitialEo, chunk.initial.eo);
        plan_2d_detail::write_protocol_rectangle(words, offset + device_protocol::kLwt2DInitialOe, chunk.initial.oe);
        plan_2d_detail::write_protocol_rectangle(words, offset + device_protocol::kLwt2DInitialOo, chunk.initial.oo);
    }
    return words;
}

[[nodiscard]] inline std::vector<uint32_t> build_ilwt_2d_route_config_words(const Ilwt2DExecutionPlan& plan) {
    TT_FATAL(!plan.chunks.empty(), "2D ILWT route protocol requires at least one chunk");
    const size_t route_count = plan.chunks.front().routes.size();
    TT_FATAL(
        route_count == 2 * plan.y_plan.forward_trace.routes.size() + 2 * plan.x_plan.forward_trace.routes.size(),
        "2D ILWT route protocol has an unexpected route count");
    std::vector<uint32_t> words(plan.chunks.size() * route_count * device_protocol::kLwt2DRouteConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const Lwt2DChunkPlan& chunk = plan.chunks[chunk_index];
        TT_FATAL(chunk.routes.size() == route_count, "2D ILWT chunks have inconsistent route counts");
        for (size_t route_index = 0; route_index < route_count; ++route_index) {
            const Lwt2DRoutePlan& route = chunk.routes[route_index];
            const size_t offset =
                (chunk_index * route_count + route_index) * device_protocol::kLwt2DRouteConfigWordCount;
            words[offset + device_protocol::kLwt2DRouteAxis] = static_cast<uint32_t>(route.axis);
            words[offset + device_protocol::kLwt2DRouteType] = static_cast<uint32_t>(route.type);
            words[offset + device_protocol::kLwt2DRouteSourceSlot] = static_cast<uint32_t>(route.source_slot);
            words[offset + device_protocol::kLwt2DRouteBaseSlot] = static_cast<uint32_t>(route.base_slot);
            words[offset + device_protocol::kLwt2DRouteOutputSlot] = static_cast<uint32_t>(route.output_slot);
            plan_2d_detail::write_protocol_rectangle(
                words, offset + device_protocol::kLwt2DRouteSourceRect, route.source);
            plan_2d_detail::write_protocol_rectangle(words, offset + device_protocol::kLwt2DRouteBaseRect, route.base);
            plan_2d_detail::write_protocol_rectangle(
                words, offset + device_protocol::kLwt2DRouteOutputRect, route.output);
            uint32_t flags = 0;
            if (route.output.empty()) {
                flags |= device_protocol::kLwt2DRouteFlagMetadataOnly;
            }
            if (is_scale_step(route.type)) {
                flags |= device_protocol::kLwt2DRouteFlagScale;
            }
            words[offset + device_protocol::kLwt2DRouteFlags] = flags;
            words[offset + device_protocol::kLwt2DRouteAxisStepIndex] =
                plan_2d_detail::checked_u32(route.axis_route_index, "2D ILWT axis route index");
        }
    }
    return words;
}

[[nodiscard]] inline std::vector<uint32_t> build_ilwt_2d_band_config_words(const Ilwt2DExecutionPlan& plan) {
    TT_FATAL(!plan.chunks.empty(), "2D ILWT terminal protocol requires at least one chunk");
    std::vector<uint32_t> words(plan.chunks.size() * device_protocol::kLwt2DBandConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const Lwt2DChunkPlan& chunk = plan.chunks[chunk_index];
        const size_t offset = chunk_index * device_protocol::kLwt2DBandConfigWordCount;
        words[offset + device_protocol::kLwt2DBandFinalYBegin] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.y.begin, "2D ILWT output y begin");
        words[offset + device_protocol::kLwt2DBandFinalYLength] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.height(), "2D ILWT output height");
        words[offset + device_protocol::kLwt2DBandFinalXBegin] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.x.begin, "2D ILWT output x begin");
        words[offset + device_protocol::kLwt2DBandFinalXLength] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.width(), "2D ILWT output width");
        const auto write_parity =
            [&](const size_t parity_offset, const Lwt2DPlaneSlot slot, const IndexRectangle source) {
                words[offset + parity_offset + device_protocol::kLwt2DBandSourceSlot] = static_cast<uint32_t>(slot);
                plan_2d_detail::write_protocol_rectangle(
                    words, offset + parity_offset + device_protocol::kLwt2DBandSourceRect, source);
            };
        write_parity(device_protocol::kLwt2DBandLl, chunk.final_bands.ll, chunk.final_band_sources.ll);
        write_parity(device_protocol::kLwt2DBandLh, chunk.final_bands.lh, chunk.final_band_sources.lh);
        write_parity(device_protocol::kLwt2DBandHl, chunk.final_bands.hl, chunk.final_band_sources.hl);
        write_parity(device_protocol::kLwt2DBandHh, chunk.final_bands.hh, chunk.final_band_sources.hh);
    }
    return words;
}

}  // namespace ttnn::operations::wavelet
