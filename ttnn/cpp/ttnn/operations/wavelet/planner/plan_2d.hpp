// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tt_stl/assert.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/operations/wavelet/common/signal.hpp"
#include "ttnn/operations/wavelet/common/storage_contract.hpp"
#include "ttnn/operations/wavelet/common/tiling_2d.hpp"
#include "ttnn/operations/wavelet/device/protocol/lwt_2d_config.hpp"
#include "ttnn/operations/wavelet/planner/execution_plan.hpp"
#include "ttnn/operations/wavelet/planner/plan.hpp"

namespace ttnn::operations::wavelet {

struct IndexRectangle {
    IndexInterval y{};
    IndexInterval x{};

    [[nodiscard]] constexpr bool empty() const noexcept { return y.empty() || x.empty(); }
    [[nodiscard]] constexpr size_t height() const noexcept { return y.length(); }
    [[nodiscard]] constexpr size_t width() const noexcept { return x.length(); }
    [[nodiscard]] constexpr size_t area() const noexcept { return empty() ? 0 : height() * width(); }
};

[[nodiscard]] constexpr IndexRectangle interval_product(const IndexInterval y, const IndexInterval x) noexcept {
    return IndexRectangle{.y = y, .x = x};
}

struct PolyphaseDependencyRectangles {
    IndexRectangle ee{};
    IndexRectangle eo{};
    IndexRectangle oe{};
    IndexRectangle oo{};

    [[nodiscard]] constexpr size_t total_area() const noexcept { return ee.area() + eo.area() + oe.area() + oo.area(); }
};

enum class Lwt2DWorkspacePolicy : uint8_t {
    kFivePlaneGeneric,
    kFourPlaneAligned,
};

enum class Lwt2DRouteDomainPolicy : uint8_t {
    kExact,
    kTileClosed,
};

enum class Lwt2DAxis : uint8_t {
    kVertical,
    kHorizontal,
};

enum class Lwt2DPlaneSlot : uint8_t {
    kP0,
    kP1,
    kP2,
    kP3,
    kScratch,
};

struct Lwt2DRoutePlan {
    Lwt2DAxis axis{Lwt2DAxis::kVertical};
    size_t axis_route_index{0};
    StepType type{StepType::kPredict};
    Lwt2DPlaneSlot source_slot{Lwt2DPlaneSlot::kP0};
    Lwt2DPlaneSlot base_slot{Lwt2DPlaneSlot::kP0};
    Lwt2DPlaneSlot output_slot{Lwt2DPlaneSlot::kP0};
    IndexRectangle source{};
    IndexRectangle base{};
    IndexRectangle output{};
    bool inline_terminal_scale{false};
};

struct Lwt2DBandSlots {
    Lwt2DPlaneSlot ll{Lwt2DPlaneSlot::kP0};
    Lwt2DPlaneSlot lh{Lwt2DPlaneSlot::kP1};
    Lwt2DPlaneSlot hl{Lwt2DPlaneSlot::kP2};
    Lwt2DPlaneSlot hh{Lwt2DPlaneSlot::kP3};
};

struct Lwt2DBandSourceRectangles {
    IndexRectangle ll{};
    IndexRectangle lh{};
    IndexRectangle hl{};
    IndexRectangle hh{};
};

struct Lwt2DResourceModel {
    std::array<uint32_t, 5> plane_heights_elements{};
    std::array<uint32_t, 5> plane_widths_elements{};
    uint64_t total_l1_bytes{0};
};

struct Lwt2DChunkPlan {
    IndexRectangle final_band_rect{};
    IndexRectangle execution_band_rect{};
    PolyphaseDependencyRectangles initial{};
    std::vector<Lwt2DRoutePlan> routes;
    Lwt2DBandSlots final_bands{};
    Lwt2DBandSourceRectangles final_band_sources{};
    Lwt2DResourceModel resources{};
    double dependency_overhead{0.0};
};

struct Lwt2DExecutionPlan {
    LiftingForwardPlan y_plan{};
    LiftingForwardPlan x_plan{};
    Lwt2DTilingContract tiling{};
    size_t input_height{0};
    size_t input_width{0};
    std::vector<Lwt2DChunkPlan> chunks;
    std::array<uint32_t, 5> allocated_plane_widths_elements{};
    std::array<uint64_t, 5> allocated_plane_slot_bytes{};
    uint64_t allocated_l1_bytes{0};
};

namespace plan_2d_detail {

using execution_detail::TerminalScaleInline;

constexpr uint32_t kTileHeight = static_cast<uint32_t>(kTileHeight2D);
constexpr uint32_t kTileWidth = static_cast<uint32_t>(kTileWidth2D);
constexpr uint64_t kFullFp32TileBytes = static_cast<uint64_t>(kTileHeight) * kTileWidth * sizeof(float);
constexpr uint64_t kCircularBufferBytes = 9 * kFullFp32TileBytes;
constexpr uint64_t kMetadataBytes = device_protocol::kLwt2DChunkConfigPageBytes +
                                    2 * device_protocol::kLwt2DRouteConfigPageBytes +
                                    device_protocol::kLwt2DBandConfigPageBytes;
constexpr uint64_t kSynchronizationBytes = 64 + device_protocol::kLwt2DSplitScratchBytes;

[[nodiscard]] inline size_t checked_area(const size_t height, const size_t width, const char* label) {
    TT_FATAL(width == 0 || height <= std::numeric_limits<size_t>::max() / width, "{} area overflows size_t", label);
    return height * width;
}

[[nodiscard]] inline size_t checked_add(const size_t lhs, const size_t rhs, const char* label) {
    TT_FATAL(lhs <= std::numeric_limits<size_t>::max() - rhs, "{} count overflows size_t", label);
    return lhs + rhs;
}

[[nodiscard]] inline uint64_t checked_bytes(const size_t elements, const char* label) {
    TT_FATAL(
        elements <= static_cast<size_t>(std::numeric_limits<uint64_t>::max() / sizeof(float)),
        "{} byte count overflows uint64_t",
        label);
    return static_cast<uint64_t>(elements) * sizeof(float);
}

[[nodiscard]] inline uint32_t checked_u32(const size_t value, const char* label) {
    TT_FATAL(value <= std::numeric_limits<uint32_t>::max(), "{} exceeds uint32_t", label);
    return static_cast<uint32_t>(value);
}

[[nodiscard]] inline IndexInterval canonical_to_stream_interval(
    const IndexInterval canonical,
    const int stream_shift,
    const size_t stream_length,
    const size_t canonical_start,
    const char* label) {
    const int64_t offset = static_cast<int64_t>(canonical_start) - static_cast<int64_t>(stream_shift);
    const int64_t begin = static_cast<int64_t>(canonical.begin) + offset;
    const int64_t end = static_cast<int64_t>(canonical.end) + offset;
    TT_FATAL(
        begin >= 0 && end >= begin && static_cast<uint64_t>(end) <= stream_length,
        "{} canonical interval [{}, {}) maps to invalid stream interval [{}, {}) with shift {}, "
        "canonical start {}, and stream length {}",
        label,
        canonical.begin,
        canonical.end,
        begin,
        end,
        stream_shift,
        canonical_start,
        stream_length);
    return IndexInterval{.begin = static_cast<size_t>(begin), .end = static_cast<size_t>(end)};
}

[[nodiscard]] constexpr size_t slot_index(const Lwt2DPlaneSlot slot) noexcept { return static_cast<size_t>(slot); }

[[nodiscard]] inline size_t aligned_interval_span(
    const IndexInterval interval, const size_t tile_extent, const char* label) {
    if (interval.empty()) {
        return 0;
    }
    TT_FATAL(
        interval.end <= std::numeric_limits<size_t>::max() - (tile_extent - 1),
        "{} interval end cannot be rounded to a tile",
        label);
    const size_t begin = (interval.begin / tile_extent) * tile_extent;
    const size_t end = round_up(interval.end, tile_extent);
    return end - begin;
}

inline void account_plane_use(
    const Lwt2DPlaneSlot slot,
    const IndexRectangle rectangle,
    std::array<size_t, 5>& heights,
    std::array<size_t, 5>& widths) {
    if (rectangle.empty()) {
        return;
    }
    const size_t index = slot_index(slot);
    heights[index] = std::max(heights[index], aligned_interval_span(rectangle.y, kTileHeight, "2D plane height"));
    widths[index] = std::max(widths[index], aligned_interval_span(rectangle.x, kTileWidth, "2D plane width"));
}

[[nodiscard]] inline Lwt2DResourceModel make_resource_model(
    const PolyphaseDependencyRectangles& initial,
    const std::vector<Lwt2DRoutePlan>& routes,
    const Lwt2DWorkspacePolicy workspace_policy) {
    const uint32_t plane_count = workspace_policy == Lwt2DWorkspacePolicy::kFourPlaneAligned ? 4U : 5U;
    std::array<size_t, 5> heights{};
    std::array<size_t, 5> widths{};
    account_plane_use(Lwt2DPlaneSlot::kP0, initial.ee, heights, widths);
    account_plane_use(Lwt2DPlaneSlot::kP1, initial.eo, heights, widths);
    account_plane_use(Lwt2DPlaneSlot::kP2, initial.oe, heights, widths);
    account_plane_use(Lwt2DPlaneSlot::kP3, initial.oo, heights, widths);
    for (const Lwt2DRoutePlan& route : routes) {
        account_plane_use(route.source_slot, route.source, heights, widths);
        account_plane_use(route.base_slot, route.base, heights, widths);
        account_plane_use(route.output_slot, route.output, heights, widths);
    }

    std::array<uint32_t, 5> plane_heights{};
    std::array<uint32_t, 5> plane_widths{};
    uint64_t workspace_bytes = 0;
    for (size_t slot = 0; slot < plane_count; ++slot) {
        TT_FATAL(heights[slot] > 0 && widths[slot] > 0, "2D LWT workspace plane {} is unused", slot);
        TT_FATAL(
            heights[slot] <= std::numeric_limits<uint32_t>::max() &&
                widths[slot] <= std::numeric_limits<uint32_t>::max(),
            "2D LWT workspace plane {} geometry {}x{} exceeds uint32_t",
            slot,
            heights[slot],
            widths[slot]);
        const uint64_t bytes = checked_bytes(
            checked_area(heights[slot], widths[slot], "2D LWT workspace plane"), "2D LWT workspace plane");
        TT_FATAL(
            workspace_bytes <= std::numeric_limits<uint64_t>::max() - bytes,
            "2D LWT workspace byte count overflows uint64_t");
        plane_heights[slot] = static_cast<uint32_t>(heights[slot]);
        plane_widths[slot] = static_cast<uint32_t>(widths[slot]);
        workspace_bytes += bytes;
    }
    const uint64_t fixed_bytes = kCircularBufferBytes + kMetadataBytes + kSynchronizationBytes;
    TT_FATAL(
        workspace_bytes <= std::numeric_limits<uint64_t>::max() - fixed_bytes,
        "2D LWT total L1 byte count overflows uint64_t");
    const uint64_t total_l1_bytes = workspace_bytes + fixed_bytes;
    return Lwt2DResourceModel{
        .plane_heights_elements = plane_heights,
        .plane_widths_elements = plane_widths,
        .total_l1_bytes = total_l1_bytes,
    };
}

struct AxisPairSlots {
    Lwt2DPlaneSlot even{Lwt2DPlaneSlot::kP0};
    Lwt2DPlaneSlot odd{Lwt2DPlaneSlot::kP1};
    Lwt2DPlaneSlot free{Lwt2DPlaneSlot::kScratch};
};

[[nodiscard]] constexpr IndexRectangle axis_rectangle(
    const Lwt2DAxis axis, const IndexInterval active, const IndexInterval transverse) noexcept {
    return axis == Lwt2DAxis::kVertical ? interval_product(active, transverse) : interval_product(transverse, active);
}

inline void append_axis_routes(
    const AxisConePlan& cone,
    const Lwt2DAxis axis,
    const IndexInterval transverse,
    const Lwt2DWorkspacePolicy workspace_policy,
    const TerminalScaleInline* terminal_scale,
    AxisPairSlots& slots,
    std::vector<Lwt2DRoutePlan>& routes) {
    const bool aligned_in_place = workspace_policy == Lwt2DWorkspacePolicy::kFourPlaneAligned;
    for (size_t route_index = 0; route_index < cone.routes.size(); ++route_index) {
        const AxisRouteRequirement& requirement = cone.routes[route_index];
        if (requirement.type == StepType::kSwap) {
            routes.push_back(Lwt2DRoutePlan{
                .axis = axis,
                .axis_route_index = route_index,
                .type = requirement.type,
                .source_slot = slots.even,
                .base_slot = slots.odd,
                .output_slot = slots.even,
                .source = axis_rectangle(axis, requirement.before.even, transverse),
                .base = axis_rectangle(axis, requirement.before.odd, transverse),
                .output = {},
            });
            std::swap(slots.even, slots.odd);
            continue;
        }

        const bool predict = requirement.type == StepType::kPredict;
        const bool update = requirement.type == StepType::kUpdate;
        const bool scale_even = requirement.type == StepType::kScaleEven;
        const bool scale_odd = requirement.type == StepType::kScaleOdd;
        TT_FATAL(predict || update || scale_even || scale_odd, "Unsupported 2D LWT route type");
        const bool fused_scale_route = terminal_scale != nullptr &&
                                       route_index != terminal_scale->predict_update_route_index &&
                                       requirement.type == terminal_scale->scale_type;
        const bool inline_terminal_scale =
            terminal_scale != nullptr && route_index == terminal_scale->predict_update_route_index;

        const Lwt2DPlaneSlot source_slot = predict      ? slots.even
                                           : update     ? slots.odd
                                           : scale_even ? slots.even
                                                        : slots.odd;
        const Lwt2DPlaneSlot base_slot = predict ? slots.odd : update ? slots.even : source_slot;
        const Lwt2DPlaneSlot output_slot =
            predict || update ? (aligned_in_place ? base_slot : slots.free) : source_slot;
        routes.push_back(Lwt2DRoutePlan{
            .axis = axis,
            .axis_route_index = route_index,
            .type = requirement.type,
            .source_slot = source_slot,
            .base_slot = base_slot,
            .output_slot = output_slot,
            .source = axis_rectangle(axis, requirement.source, transverse),
            .base = axis_rectangle(axis, requirement.base, transverse),
            .output = fused_scale_route ? IndexRectangle{} : axis_rectangle(axis, requirement.output, transverse),
            .inline_terminal_scale = inline_terminal_scale,
        });

        if (!predict && !update) {
            continue;
        }
        if (aligned_in_place) {
            continue;
        }
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
    const AxisConePlan& y_cone,
    const AxisConePlan& x_cone,
    const Lwt2DWorkspacePolicy workspace_policy,
    const TerminalScaleInline* y_terminal_scale,
    const TerminalScaleInline* x_terminal_scale) {
    std::vector<Lwt2DRoutePlan> routes;
    routes.reserve(2 * y_cone.routes.size() + 2 * x_cone.routes.size());

    AxisPairSlots x_even_pair{
        .even = Lwt2DPlaneSlot::kP0,
        .odd = Lwt2DPlaneSlot::kP2,
        .free = Lwt2DPlaneSlot::kScratch,
    };
    append_axis_routes(
        y_cone, Lwt2DAxis::kVertical, x_cone.initial_even, workspace_policy, y_terminal_scale, x_even_pair, routes);

    AxisPairSlots x_odd_pair{
        .even = Lwt2DPlaneSlot::kP1,
        .odd = Lwt2DPlaneSlot::kP3,
        .free = x_even_pair.free,
    };
    append_axis_routes(
        y_cone, Lwt2DAxis::kVertical, x_cone.initial_odd, workspace_policy, y_terminal_scale, x_odd_pair, routes);

    AxisPairSlots vertical_low_pair{
        .even = x_even_pair.even,
        .odd = x_odd_pair.even,
        .free = x_odd_pair.free,
    };
    append_axis_routes(
        x_cone,
        Lwt2DAxis::kHorizontal,
        y_cone.final_even,
        workspace_policy,
        x_terminal_scale,
        vertical_low_pair,
        routes);

    AxisPairSlots vertical_high_pair{
        .even = x_even_pair.odd,
        .odd = x_odd_pair.odd,
        .free = vertical_low_pair.free,
    };
    append_axis_routes(
        x_cone,
        Lwt2DAxis::kHorizontal,
        y_cone.final_odd,
        workspace_policy,
        x_terminal_scale,
        vertical_high_pair,
        routes);

    const Lwt2DBandSlots bands{
        .ll = vertical_low_pair.even,
        .lh = vertical_low_pair.odd,
        .hl = vertical_high_pair.even,
        .hh = vertical_high_pair.odd,
    };
    std::array<Lwt2DPlaneSlot, 4> final_slots = {
        bands.ll,
        bands.lh,
        bands.hl,
        bands.hh,
    };
    std::sort(final_slots.begin(), final_slots.end());
    TT_FATAL(
        std::adjacent_find(final_slots.begin(), final_slots.end()) == final_slots.end(),
        "2D LWT route schedule aliases two final bands to one plane");
    if (workspace_policy == Lwt2DWorkspacePolicy::kFourPlaneAligned) {
        TT_FATAL(
            std::none_of(
                final_slots.begin(),
                final_slots.end(),
                [](const Lwt2DPlaneSlot slot) { return slot == Lwt2DPlaneSlot::kScratch; }),
            "Aligned four-plane 2D LWT schedule used the scratch plane");
    }
    return {std::move(routes), bands};
}

[[nodiscard]] inline Lwt2DChunkPlan build_chunk(
    const LiftingForwardPlan& y_plan,
    const LiftingForwardPlan& x_plan,
    const IndexRectangle final_band_rect,
    const bool fuse_terminal_scale,
    const Lwt2DRouteDomainPolicy route_domain) {
    const size_t y_tap_size = static_cast<size_t>(y_plan.preprocess_layout.pad_config.left) + 1;
    const size_t x_tap_size = static_cast<size_t>(x_plan.preprocess_layout.pad_config.left) + 1;
    const IndexInterval final_y_even = canonical_to_stream_interval(
        final_band_rect.y, y_plan.final_even_shift, y_plan.final_even_length, y_tap_size / 2, "vertical even");
    const IndexInterval final_y_odd = canonical_to_stream_interval(
        final_band_rect.y, y_plan.final_odd_shift, y_plan.final_odd_length, y_tap_size / 2, "vertical odd");
    const IndexInterval final_x_even = canonical_to_stream_interval(
        final_band_rect.x, x_plan.final_even_shift, x_plan.final_even_length, x_tap_size / 2, "horizontal even");
    const IndexInterval final_x_odd = canonical_to_stream_interval(
        final_band_rect.x, x_plan.final_odd_shift, x_plan.final_odd_length, x_tap_size / 2, "horizontal odd");

    const AxisConePlan exact_y_cone = build_axis_cone(y_plan, final_y_even, final_y_odd);
    const AxisConePlan exact_x_cone = build_axis_cone(x_plan, final_x_even, final_x_odd);
    AxisConePlan y_cone = route_domain == Lwt2DRouteDomainPolicy::kTileClosed
                              ? build_axis_cone(y_plan, final_y_even, final_y_odd, kTileHeight)
                              : exact_y_cone;
    AxisConePlan x_cone = route_domain == Lwt2DRouteDomainPolicy::kTileClosed
                              ? build_axis_cone(x_plan, final_x_even, final_x_odd, kTileWidth)
                              : exact_x_cone;
    const PolyphaseDependencyRectangles initial{
        .ee = interval_product(y_cone.initial_even, x_cone.initial_even),
        .eo = interval_product(y_cone.initial_even, x_cone.initial_odd),
        .oe = interval_product(y_cone.initial_odd, x_cone.initial_even),
        .oo = interval_product(y_cone.initial_odd, x_cone.initial_odd),
    };

    // The correctness-first production path always uses a separate scratch
    // plane. Four-plane route aliasing remains a future optimization.
    constexpr Lwt2DWorkspacePolicy workspace_policy = Lwt2DWorkspacePolicy::kFivePlaneGeneric;
    const TerminalScaleInline y_terminal_scale = execution_detail::terminal_scale_inline(y_plan);
    const TerminalScaleInline x_terminal_scale = execution_detail::terminal_scale_inline(x_plan);
    auto [routes, final_bands] = build_route_schedule(
        y_cone,
        x_cone,
        workspace_policy,
        fuse_terminal_scale ? &y_terminal_scale : nullptr,
        fuse_terminal_scale ? &x_terminal_scale : nullptr);
    const Lwt2DResourceModel resources = make_resource_model(initial, routes, workspace_policy);
    const Lwt2DBandSourceRectangles final_band_sources{
        .ll = interval_product(exact_y_cone.final_even, exact_x_cone.final_even),
        .lh = interval_product(exact_y_cone.final_even, exact_x_cone.final_odd),
        .hl = interval_product(exact_y_cone.final_odd, exact_x_cone.final_even),
        .hh = interval_product(exact_y_cone.final_odd, exact_x_cone.final_odd),
    };
    const size_t final_area =
        checked_area(final_band_rect.height(), final_band_rect.width(), "2D LWT final band rectangle");
    TT_FATAL(
        final_area <= std::numeric_limits<size_t>::max() / 4, "2D LWT four-band output element count overflows size_t");
    const size_t final_elements = 4 * final_area;
    const size_t ee_dependencies = checked_area(initial.ee.height(), initial.ee.width(), "2D LWT EE dependency");
    const size_t eo_dependencies = checked_area(initial.eo.height(), initial.eo.width(), "2D LWT EO dependency");
    const size_t oe_dependencies = checked_area(initial.oe.height(), initial.oe.width(), "2D LWT OE dependency");
    const size_t oo_dependencies = checked_area(initial.oo.height(), initial.oo.width(), "2D LWT OO dependency");
    const size_t even_dependencies = checked_add(ee_dependencies, eo_dependencies, "2D LWT dependency");
    const size_t odd_dependencies = checked_add(oe_dependencies, oo_dependencies, "2D LWT dependency");
    const size_t dependency_elements = checked_add(even_dependencies, odd_dependencies, "2D LWT dependency");
    const double dependency_overhead =
        final_elements == 0 ? 0.0
                            : static_cast<double>(dependency_elements - std::min(dependency_elements, final_elements)) /
                                  static_cast<double>(final_elements);
    return Lwt2DChunkPlan{
        .final_band_rect = final_band_rect,
        .execution_band_rect =
            IndexRectangle{
                .y =
                    IndexInterval{
                        .begin = (final_band_rect.y.begin / kTileHeight) * kTileHeight,
                        .end = round_up(final_band_rect.y.end, static_cast<size_t>(kTileHeight)),
                    },
                .x =
                    IndexInterval{
                        .begin = (final_band_rect.x.begin / kTileWidth) * kTileWidth,
                        .end = round_up(final_band_rect.x.end, static_cast<size_t>(kTileWidth)),
                    },
            },
        .initial = initial,
        .routes = std::move(routes),
        .final_bands = final_bands,
        .final_band_sources = final_band_sources,
        .resources = resources,
        .dependency_overhead = dependency_overhead,
    };
}

[[nodiscard]] inline std::vector<Lwt2DChunkPlan> build_chunks(
    const LiftingForwardPlan& y_plan,
    const LiftingForwardPlan& x_plan,
    const uint32_t chunk_tiles_y,
    const uint32_t chunk_tiles_x,
    const bool fuse_terminal_scale,
    const Lwt2DRouteDomainPolicy route_domain) {
    TT_FATAL(chunk_tiles_y > 0 && chunk_tiles_x > 0, "2D LWT chunk tile dimensions must be positive");
    const size_t chunk_height = static_cast<size_t>(chunk_tiles_y) * kTileHeight;
    const size_t chunk_width = static_cast<size_t>(chunk_tiles_x) * kTileWidth;
    std::vector<Lwt2DChunkPlan> chunks;
    const size_t chunk_rows = ceil_div(y_plan.output_length, chunk_height);
    const size_t chunk_cols = ceil_div(x_plan.output_length, chunk_width);
    chunks.reserve(checked_area(chunk_rows, chunk_cols, "2D LWT chunk grid"));
    for (size_t y = 0; y < y_plan.output_length;) {
        const size_t y_end = y + std::min(chunk_height, y_plan.output_length - y);
        for (size_t x = 0; x < x_plan.output_length;) {
            const size_t x_end = x + std::min(chunk_width, x_plan.output_length - x);
            chunks.push_back(build_chunk(
                y_plan,
                x_plan,
                IndexRectangle{
                    .y = IndexInterval{.begin = y, .end = y_end},
                    .x = IndexInterval{.begin = x, .end = x_end},
                },
                fuse_terminal_scale,
                route_domain));
            x = x_end;
        }
        y = y_end;
    }
    return chunks;
}

struct Candidate {
    uint32_t chunk_tiles_y{0};
    uint32_t chunk_tiles_x{0};
    uint32_t active_core_count{0};
    double max_dependency_overhead{0.0};
    uint64_t estimated_latency_cycles{0};
    std::vector<Lwt2DChunkPlan> chunks;
};

enum class AlignmentCostClass : uint8_t {
    kExact,
    kOneAxisShifted,
    kGeneric,
};

[[nodiscard]] inline int64_t signed_tile_modulo(const int64_t value) noexcept {
    const int64_t remainder = value % static_cast<int64_t>(kTileHeight);
    return remainder < 0 ? remainder + static_cast<int64_t>(kTileHeight) : remainder;
}

[[nodiscard]] inline AlignmentCostClass alignment_cost_class(
    const IndexRectangle stored, const int64_t requested_y, const int64_t requested_x) noexcept {
    const bool inside = requested_y >= static_cast<int64_t>(stored.y.begin) &&
                        requested_x >= static_cast<int64_t>(stored.x.begin) &&
                        requested_y + static_cast<int64_t>(kTileHeight) <= static_cast<int64_t>(stored.y.end) &&
                        requested_x + static_cast<int64_t>(kTileWidth) <= static_cast<int64_t>(stored.x.end);
    if (!inside) {
        return AlignmentCostClass::kGeneric;
    }
    const bool y_aligned = signed_tile_modulo(requested_y) == 0;
    const bool x_aligned = signed_tile_modulo(requested_x) == 0;
    if (y_aligned && x_aligned) {
        return AlignmentCostClass::kExact;
    }
    if (y_aligned != x_aligned) {
        return AlignmentCostClass::kOneAxisShifted;
    }
    return AlignmentCostClass::kGeneric;
}

[[nodiscard]] constexpr uint64_t staging_class_cycles(const AlignmentCostClass tile_class) noexcept {
    switch (tile_class) {
        case AlignmentCostClass::kExact: return 900;
        case AlignmentCostClass::kOneAxisShifted: return 7'000;
        case AlignmentCostClass::kGeneric: return 9'000;
    }
    return 70'000;
}

[[nodiscard]] inline uint64_t estimate_chunk_latency_cycles(
    const Lwt2DChunkPlan& chunk,
    const LiftingForwardPlan& y_plan,
    const LiftingForwardPlan& x_plan,
    const bool inverse = false) {
    constexpr uint64_t route_config_and_sync_cycles = 3'700;
    constexpr uint64_t full_tile_persistence_cycles = 1'200;
    constexpr uint64_t fragmented_terminal_tile_cycles = 80'000;
    constexpr uint64_t interleaved_terminal_tile_cycles = 80'000;
    constexpr uint64_t tiled_terminal_tile_cycles = 1'200;
    uint64_t cycles = chunk.initial.total_area() * 12;
    std::array<IndexRectangle, 5> stored = {
        chunk.initial.ee,
        chunk.initial.eo,
        chunk.initial.oe,
        chunk.initial.oo,
        IndexRectangle{},
    };
    for (const Lwt2DRoutePlan& route : chunk.routes) {
        cycles += route_config_and_sync_cycles;
        if (route.output.empty()) {
            continue;
        }
        const LiftingStepRoute& axis_route = route.axis == Lwt2DAxis::kVertical ? y_plan.routes[route.axis_route_index]
                                                                                : x_plan.routes[route.axis_route_index];
        const uint32_t k = is_predict_update_step(route.type) ? execution_detail::coefficient_count(axis_route) : 1U;
        const int64_t output_y_origin = static_cast<int64_t>((route.output.y.begin / kTileHeight) * kTileHeight);
        const int64_t output_x_origin = static_cast<int64_t>((route.output.x.begin / kTileWidth) * kTileWidth);
        const size_t tile_rows = round_up(route.output.y.end, static_cast<size_t>(kTileHeight)) / kTileHeight -
                                 route.output.y.begin / kTileHeight;
        const size_t tile_columns = round_up(route.output.x.end, static_cast<size_t>(kTileWidth)) / kTileWidth -
                                    route.output.x.begin / kTileWidth;
        for (size_t tile_y = 0; tile_y < tile_rows; ++tile_y) {
            for (size_t tile_x = 0; tile_x < tile_columns; ++tile_x) {
                const int64_t output_tile_y = output_y_origin + static_cast<int64_t>(tile_y * kTileHeight);
                const int64_t output_tile_x = output_x_origin + static_cast<int64_t>(tile_x * kTileWidth);
                const auto requested_origin = [&](const IndexRectangle rectangle) {
                    return std::pair{
                        static_cast<int64_t>(rectangle.y.begin) + output_tile_y -
                            static_cast<int64_t>(route.output.y.begin),
                        static_cast<int64_t>(rectangle.x.begin) + output_tile_x -
                            static_cast<int64_t>(route.output.x.begin),
                    };
                };
                if (is_predict_update_step(route.type)) {
                    const auto [base_y, base_x] = requested_origin(route.base);
                    cycles +=
                        staging_class_cycles(alignment_cost_class(stored[slot_index(route.base_slot)], base_y, base_x));
                    const auto [source_y, source_x] = requested_origin(route.source);
                    for (uint32_t source_tile = 0; source_tile < 2; ++source_tile) {
                        const int64_t requested_y =
                            source_y +
                            (route.axis == Lwt2DAxis::kVertical ? static_cast<int64_t>(source_tile * kTileHeight) : 0);
                        const int64_t requested_x = source_x + (route.axis == Lwt2DAxis::kHorizontal
                                                                    ? static_cast<int64_t>(source_tile * kTileWidth) -
                                                                          static_cast<int64_t>(17 - k)
                                                                    : 0);
                        cycles += staging_class_cycles(
                            alignment_cost_class(stored[slot_index(route.source_slot)], requested_y, requested_x));
                    }
                    cycles += route.axis == Lwt2DAxis::kVertical ? 16'000 + 2'500 * k : 12'000 + 1'800 * k;
                } else {
                    const auto [source_y, source_x] = requested_origin(route.source);
                    cycles += staging_class_cycles(
                        alignment_cost_class(stored[slot_index(route.source_slot)], source_y, source_x));
                    cycles += 8'000;
                }
                cycles += full_tile_persistence_cycles;
            }
        }
        stored[slot_index(route.output_slot)] = route.output;
    }

    const bool full_terminal_tiles =
        chunk.final_band_rect.y.begin % kTileHeight == 0 && chunk.final_band_rect.x.begin % kTileWidth == 0 &&
        chunk.final_band_rect.height() % kTileHeight == 0 && chunk.final_band_rect.width() % kTileWidth == 0;
    const uint64_t terminal_tiles =
        static_cast<uint64_t>(ceil_div(chunk.final_band_rect.height(), static_cast<size_t>(kTileHeight))) *
        ceil_div(chunk.final_band_rect.width(), static_cast<size_t>(kTileWidth));
    if (inverse) {
        cycles += terminal_tiles * interleaved_terminal_tile_cycles;
    } else {
        cycles +=
            4 * terminal_tiles * (full_terminal_tiles ? tiled_terminal_tile_cycles : fragmented_terminal_tile_cycles);
    }
    return cycles;
}

[[nodiscard]] inline uint64_t estimate_candidate_latency_cycles(
    const std::vector<Lwt2DChunkPlan>& chunks,
    const uint32_t active_core_count,
    const LiftingForwardPlan& y_plan,
    const LiftingForwardPlan& x_plan,
    const bool inverse = false,
    const uint64_t inverse_coordination_penalty_cycles_per_core = 0) {
    std::vector<uint64_t> chunk_costs;
    chunk_costs.reserve(chunks.size());
    for (const Lwt2DChunkPlan& chunk : chunks) {
        chunk_costs.push_back(estimate_chunk_latency_cycles(chunk, y_plan, x_plan, inverse));
    }
    const size_t base = chunks.size() / active_core_count;
    const size_t extra = chunks.size() % active_core_count;
    size_t begin = 0;
    uint64_t maximum = 0;
    for (uint32_t core = 0; core < active_core_count; ++core) {
        const size_t count = base + (core < extra ? 1U : 0U);
        uint64_t core_cycles = 30'000;
        for (size_t index = 0; index < count; ++index) {
            core_cycles += chunk_costs[begin + index];
        }
        maximum = std::max(maximum, core_cycles);
        begin += count;
    }
    if (inverse && inverse_coordination_penalty_cycles_per_core > 0 && active_core_count > 64) {
        maximum += static_cast<uint64_t>(active_core_count - 64) * inverse_coordination_penalty_cycles_per_core;
    }
    return maximum;
}

[[nodiscard]] inline bool is_better_candidate(
    const Candidate& candidate, const Candidate& best, const bool latency_oriented) noexcept {
    if (latency_oriented && candidate.estimated_latency_cycles != best.estimated_latency_cycles) {
        if (candidate.active_core_count < best.active_core_count) {
            return static_cast<long double>(candidate.estimated_latency_cycles) <
                   0.90L * static_cast<long double>(best.estimated_latency_cycles);
        }
        if (candidate.active_core_count > best.active_core_count) {
            return static_cast<long double>(candidate.estimated_latency_cycles) <=
                   1.10L * static_cast<long double>(best.estimated_latency_cycles);
        }
        return candidate.estimated_latency_cycles < best.estimated_latency_cycles;
    }
    if (candidate.active_core_count != best.active_core_count) {
        return candidate.active_core_count > best.active_core_count;
    }
    if (candidate.max_dependency_overhead != best.max_dependency_overhead) {
        return candidate.max_dependency_overhead < best.max_dependency_overhead;
    }
    const uint64_t candidate_area = static_cast<uint64_t>(candidate.chunk_tiles_y) * candidate.chunk_tiles_x;
    const uint64_t best_area = static_cast<uint64_t>(best.chunk_tiles_y) * best.chunk_tiles_x;
    if (candidate_area != best_area) {
        return candidate_area > best_area;
    }
    const uint32_t candidate_aspect = candidate.chunk_tiles_y > candidate.chunk_tiles_x
                                          ? candidate.chunk_tiles_y - candidate.chunk_tiles_x
                                          : candidate.chunk_tiles_x - candidate.chunk_tiles_y;
    const uint32_t best_aspect = best.chunk_tiles_y > best.chunk_tiles_x ? best.chunk_tiles_y - best.chunk_tiles_x
                                                                         : best.chunk_tiles_x - best.chunk_tiles_y;
    return candidate_aspect < best_aspect;
}

}  // namespace plan_2d_detail

[[nodiscard]] inline Lwt2DExecutionPlan make_lwt_2d_execution_plan(
    LiftingForwardPlan y_plan,
    LiftingForwardPlan x_plan,
    const uint32_t core_limit,
    const uint64_t l1_budget_bytes,
    const bool fuse_terminal_scale = false,
    const bool latency_oriented_planner = false,
    const Lwt2DRouteDomainPolicy route_domain = Lwt2DRouteDomainPolicy::kExact) {
    TT_FATAL(core_limit > 0, "2D LWT requires at least one worker core");
    TT_FATAL(y_plan.preprocess_layout.input.length > 0, "2D LWT input height must be positive");
    TT_FATAL(x_plan.preprocess_layout.input.length > 0, "2D LWT input width must be positive");
    TT_FATAL(
        y_plan.preprocess_layout.pad_config.mode == x_plan.preprocess_layout.pad_config.mode &&
            is_supported_lwt_boundary_mode(y_plan.preprocess_layout.pad_config.mode),
        "2D LWT requires one supported extension mode shared by both axes");
    TT_FATAL(
        !boundary_mode_requires_multiple_samples(y_plan.preprocess_layout.pad_config.mode) ||
            (y_plan.preprocess_layout.input.length > 1 && x_plan.preprocess_layout.input.length > 1),
        "2D reflect and antireflect extension require both input dimensions to exceed one");
    TT_FATAL(
        l1_budget_bytes > plan_2d_detail::kCircularBufferBytes + plan_2d_detail::kMetadataBytes +
                              plan_2d_detail::kSynchronizationBytes,
        "2D LWT L1 budget {} is too small for fixed kernel resources",
        l1_budget_bytes);

    const size_t band_tiles_y_size = ceil_div(y_plan.output_length, static_cast<size_t>(plan_2d_detail::kTileHeight));
    const size_t band_tiles_x_size = ceil_div(x_plan.output_length, static_cast<size_t>(plan_2d_detail::kTileWidth));
    TT_FATAL(
        band_tiles_y_size <= std::numeric_limits<uint32_t>::max() &&
            band_tiles_x_size <= std::numeric_limits<uint32_t>::max(),
        "2D LWT band tile grid {}x{} exceeds uint32_t planner geometry",
        band_tiles_y_size,
        band_tiles_x_size);
    const uint32_t band_tiles_y = static_cast<uint32_t>(band_tiles_y_size);
    const uint32_t band_tiles_x = static_cast<uint32_t>(band_tiles_x_size);

    plan_2d_detail::Candidate best{};
    bool found = false;
    for (uint32_t tiles_y = 1; tiles_y <= band_tiles_y; ++tiles_y) {
        for (uint32_t tiles_x = 1; tiles_x <= band_tiles_x; ++tiles_x) {
            std::vector<Lwt2DChunkPlan> chunks =
                plan_2d_detail::build_chunks(y_plan, x_plan, tiles_y, tiles_x, fuse_terminal_scale, route_domain);
            double max_dependency_overhead = 0.0;
            bool fits = true;
            for (const Lwt2DChunkPlan& chunk : chunks) {
                max_dependency_overhead = std::max(max_dependency_overhead, chunk.dependency_overhead);
                fits = fits && chunk.resources.total_l1_bytes <= l1_budget_bytes;
            }
            if (!fits) {
                continue;
            }

            plan_2d_detail::Candidate candidate{
                .chunk_tiles_y = tiles_y,
                .chunk_tiles_x = tiles_x,
                .active_core_count = static_cast<uint32_t>(std::min(chunks.size(), static_cast<size_t>(core_limit))),
                .max_dependency_overhead = max_dependency_overhead,
                .estimated_latency_cycles = plan_2d_detail::estimate_candidate_latency_cycles(
                    chunks,
                    static_cast<uint32_t>(std::min(chunks.size(), static_cast<size_t>(core_limit))),
                    y_plan,
                    x_plan),
                .chunks = std::move(chunks),
            };
            if (!found || plan_2d_detail::is_better_candidate(candidate, best, latency_oriented_planner)) {
                best = std::move(candidate);
                found = true;
            }
        }
    }

    TT_FATAL(
        found,
        "No 2D LWT band-tile chunk fits the {}-byte L1 budget for input {}x{}",
        l1_budget_bytes,
        y_plan.preprocess_layout.input.length,
        x_plan.preprocess_layout.input.length);
    const size_t input_height = y_plan.preprocess_layout.input.length;
    const size_t input_width = x_plan.preprocess_layout.input.length;
    const size_t band_height = y_plan.output_length;
    const size_t band_width = x_plan.output_length;
    const Lwt2DTilingContract tiling{
        .input = TiledShape2D::from_logical(Shape2D{.height = input_height, .width = input_width}),
        .band = TiledShape2D::from_logical(Shape2D{.height = band_height, .width = band_width}),
        .padding_precedes_split = true,
    };
    validate_lwt_2d_tiling_contract(tiling);
    std::array<uint32_t, 5> allocated_plane_heights{};
    std::array<uint32_t, 5> allocated_plane_widths{};
    std::array<uint64_t, 5> allocated_plane_bytes{};
    uint64_t allocated_workspace_bytes = 0;
    for (size_t slot = 0; slot < allocated_plane_bytes.size(); ++slot) {
        for (const Lwt2DChunkPlan& chunk : best.chunks) {
            allocated_plane_heights[slot] =
                std::max(allocated_plane_heights[slot], chunk.resources.plane_heights_elements[slot]);
            allocated_plane_widths[slot] =
                std::max(allocated_plane_widths[slot], chunk.resources.plane_widths_elements[slot]);
        }
        const size_t elements = plan_2d_detail::checked_area(
            allocated_plane_heights[slot], allocated_plane_widths[slot], "2D allocated workspace plane");
        allocated_plane_bytes[slot] = plan_2d_detail::checked_bytes(elements, "2D allocated workspace plane");
        TT_FATAL(
            allocated_workspace_bytes <= std::numeric_limits<uint64_t>::max() - allocated_plane_bytes[slot],
            "2D allocated workspace byte count overflows uint64_t");
        allocated_workspace_bytes += allocated_plane_bytes[slot];
    }
    constexpr uint64_t fixed_l1_bytes =
        plan_2d_detail::kCircularBufferBytes + plan_2d_detail::kMetadataBytes + plan_2d_detail::kSynchronizationBytes;
    TT_FATAL(
        allocated_workspace_bytes <= std::numeric_limits<uint64_t>::max() - fixed_l1_bytes,
        "2D allocated L1 byte count overflows uint64_t");
    const uint64_t allocated_l1_bytes = allocated_workspace_bytes + fixed_l1_bytes;
    TT_FATAL(
        allocated_l1_bytes <= l1_budget_bytes,
        "2D uniform workspace allocation requires {} bytes per core, exceeding the {}-byte L1 budget",
        allocated_l1_bytes,
        l1_budget_bytes);
    return Lwt2DExecutionPlan{
        .y_plan = std::move(y_plan),
        .x_plan = std::move(x_plan),
        .tiling = tiling,
        .input_height = input_height,
        .input_width = input_width,
        .chunks = std::move(best.chunks),
        .allocated_plane_widths_elements = allocated_plane_widths,
        .allocated_plane_slot_bytes = allocated_plane_bytes,
        .allocated_l1_bytes = allocated_l1_bytes,
    };
}

template <typename Scheme>
[[nodiscard]] Lwt2DExecutionPlan make_lwt_2d_execution_plan(
    const size_t input_height,
    const size_t input_width,
    const uint32_t core_limit,
    const uint64_t l1_budget_bytes,
    const BoundaryMode boundary_mode = BoundaryMode::kSymmetric,
    const bool fuse_terminal_scale = false,
    const bool latency_oriented_planner = false,
    const Lwt2DRouteDomainPolicy route_domain = Lwt2DRouteDomainPolicy::kExact) {
    TT_FATAL(input_height > 0 && input_width > 0, "2D LWT input dimensions must be positive");
    const SignalBuffer y_input{
        .length = input_height,
        .stick_width = kStickWidth,
        .element_size_bytes = sizeof(float),
    };
    const SignalBuffer x_input{
        .length = input_width,
        .stick_width = kStickWidth,
        .element_size_bytes = sizeof(float),
    };
    return make_lwt_2d_execution_plan(
        make_forward_lifting_plan<Scheme>(y_input, boundary_mode),
        make_forward_lifting_plan<Scheme>(x_input, boundary_mode),
        core_limit,
        l1_budget_bytes,
        fuse_terminal_scale,
        latency_oriented_planner,
        route_domain);
}

namespace plan_2d_detail {

inline void write_protocol_rectangle(
    std::vector<uint32_t>& words, const size_t offset, const IndexRectangle rectangle) {
    words[offset + device_protocol::kLwt2DRectYBegin] = checked_u32(rectangle.y.begin, "2D rectangle y begin");
    words[offset + device_protocol::kLwt2DRectYLength] = checked_u32(rectangle.y.length(), "2D rectangle height");
    words[offset + device_protocol::kLwt2DRectXBegin] = checked_u32(rectangle.x.begin, "2D rectangle x begin");
    words[offset + device_protocol::kLwt2DRectXLength] = checked_u32(rectangle.x.length(), "2D rectangle width");
}

}  // namespace plan_2d_detail

[[nodiscard]] inline std::vector<uint32_t> build_lwt_2d_chunk_config_words(const Lwt2DExecutionPlan& plan) {
    TT_FATAL(!plan.chunks.empty(), "2D LWT chunk protocol requires at least one chunk");
    std::vector<uint32_t> words(plan.chunks.size() * device_protocol::kLwt2DChunkConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const Lwt2DChunkPlan& chunk = plan.chunks[chunk_index];
        const size_t offset = chunk_index * device_protocol::kLwt2DChunkConfigWordCount;
        words[offset + device_protocol::kLwt2DFinalYBegin] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.y.begin, "2D final y begin");
        words[offset + device_protocol::kLwt2DFinalYLength] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.height(), "2D final height");
        words[offset + device_protocol::kLwt2DFinalXBegin] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.x.begin, "2D final x begin");
        words[offset + device_protocol::kLwt2DFinalXLength] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.width(), "2D final width");
        words[offset + device_protocol::kLwt2DExecutionTileYBegin] =
            plan_2d_detail::checked_u32(chunk.execution_band_rect.y.begin / kTileHeight2D, "2D execution tile y");
        words[offset + device_protocol::kLwt2DExecutionTileYCount] =
            plan_2d_detail::checked_u32(chunk.execution_band_rect.height() / kTileHeight2D, "2D execution tile rows");
        words[offset + device_protocol::kLwt2DExecutionTileXBegin] =
            plan_2d_detail::checked_u32(chunk.execution_band_rect.x.begin / kTileWidth2D, "2D execution tile x");
        words[offset + device_protocol::kLwt2DExecutionTileXCount] =
            plan_2d_detail::checked_u32(chunk.execution_band_rect.width() / kTileWidth2D, "2D execution tile columns");
        plan_2d_detail::write_protocol_rectangle(words, offset + device_protocol::kLwt2DInitialEe, chunk.initial.ee);
        plan_2d_detail::write_protocol_rectangle(words, offset + device_protocol::kLwt2DInitialEo, chunk.initial.eo);
        plan_2d_detail::write_protocol_rectangle(words, offset + device_protocol::kLwt2DInitialOe, chunk.initial.oe);
        plan_2d_detail::write_protocol_rectangle(words, offset + device_protocol::kLwt2DInitialOo, chunk.initial.oo);
    }
    return words;
}

[[nodiscard]] inline std::vector<uint32_t> build_lwt_2d_route_config_words(const Lwt2DExecutionPlan& plan) {
    TT_FATAL(!plan.chunks.empty(), "2D LWT route protocol requires at least one chunk");
    const size_t route_count = plan.chunks.front().routes.size();
    TT_FATAL(
        route_count == 2 * plan.y_plan.routes.size() + 2 * plan.x_plan.routes.size(),
        "2D LWT route protocol has an unexpected route count");
    std::vector<uint32_t> words(plan.chunks.size() * route_count * device_protocol::kLwt2DRouteConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const Lwt2DChunkPlan& chunk = plan.chunks[chunk_index];
        TT_FATAL(chunk.routes.size() == route_count, "2D LWT chunks have inconsistent route counts");
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
            if (route.inline_terminal_scale) {
                flags |= device_protocol::kLwt2DRouteFlagInlineTerminalScale;
            }
            words[offset + device_protocol::kLwt2DRouteFlags] = flags;
            words[offset + device_protocol::kLwt2DRouteAxisStepIndex] =
                plan_2d_detail::checked_u32(route.axis_route_index, "2D axis route index");
        }
    }
    return words;
}

[[nodiscard]] inline std::vector<uint32_t> build_lwt_2d_band_config_words(const Lwt2DExecutionPlan& plan) {
    TT_FATAL(!plan.chunks.empty(), "2D LWT band protocol requires at least one chunk");
    std::vector<uint32_t> words(plan.chunks.size() * device_protocol::kLwt2DBandConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const Lwt2DChunkPlan& chunk = plan.chunks[chunk_index];
        const size_t offset = chunk_index * device_protocol::kLwt2DBandConfigWordCount;
        words[offset + device_protocol::kLwt2DBandFinalYBegin] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.y.begin, "2D final band y begin");
        words[offset + device_protocol::kLwt2DBandFinalYLength] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.height(), "2D final band height");
        words[offset + device_protocol::kLwt2DBandFinalXBegin] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.x.begin, "2D final band x begin");
        words[offset + device_protocol::kLwt2DBandFinalXLength] =
            plan_2d_detail::checked_u32(chunk.final_band_rect.width(), "2D final band width");
        const auto write_band = [&](const size_t band_offset, const Lwt2DPlaneSlot slot, const IndexRectangle source) {
            words[offset + band_offset + device_protocol::kLwt2DBandSourceSlot] = static_cast<uint32_t>(slot);
            plan_2d_detail::write_protocol_rectangle(
                words, offset + band_offset + device_protocol::kLwt2DBandSourceRect, source);
        };
        write_band(device_protocol::kLwt2DBandLl, chunk.final_bands.ll, chunk.final_band_sources.ll);
        write_band(device_protocol::kLwt2DBandLh, chunk.final_bands.lh, chunk.final_band_sources.lh);
        write_band(device_protocol::kLwt2DBandHl, chunk.final_bands.hl, chunk.final_band_sources.hl);
        write_band(device_protocol::kLwt2DBandHh, chunk.final_bands.hh, chunk.final_band_sources.hh);
    }
    return words;
}

}  // namespace ttnn::operations::wavelet
