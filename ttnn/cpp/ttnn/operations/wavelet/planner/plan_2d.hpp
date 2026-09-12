// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <tt_stl/assert.hpp>
#include <utility>
#include <vector>

#include "tt-metalium/math.hpp"
#include "ttnn/operations/wavelet/common/signal.hpp"
#include "ttnn/operations/wavelet/common/storage_contract.hpp"
#include "ttnn/operations/wavelet/common/tiling_2d.hpp"
#include "ttnn/operations/wavelet/device/protocol/lwt_2d_config.hpp"
#include "ttnn/operations/wavelet/planner/cost_model.hpp"
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
    const size_t begin = tt::round_down(interval.begin, tile_extent);
    const size_t end = tt::round_up(interval.end, tile_extent);
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
    const PolyphaseDependencyRectangles& initial, const std::vector<Lwt2DRoutePlan>& routes) {
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
    for (size_t slot = 0; slot < plane_heights.size(); ++slot) {
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
    const TerminalScaleInline* terminal_scale,
    AxisPairSlots& slots,
    std::vector<Lwt2DRoutePlan>& routes) {
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
        const Lwt2DPlaneSlot output_slot = predict || update ? slots.free : source_slot;
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
    const TerminalScaleInline* y_terminal_scale,
    const TerminalScaleInline* x_terminal_scale) {
    std::vector<Lwt2DRoutePlan> routes;
    routes.reserve(2 * y_cone.routes.size() + 2 * x_cone.routes.size());

    AxisPairSlots x_even_pair{
        .even = Lwt2DPlaneSlot::kP0,
        .odd = Lwt2DPlaneSlot::kP2,
        .free = Lwt2DPlaneSlot::kScratch,
    };
    append_axis_routes(y_cone, Lwt2DAxis::kVertical, x_cone.initial_even, y_terminal_scale, x_even_pair, routes);

    AxisPairSlots x_odd_pair{
        .even = Lwt2DPlaneSlot::kP1,
        .odd = Lwt2DPlaneSlot::kP3,
        .free = x_even_pair.free,
    };
    append_axis_routes(y_cone, Lwt2DAxis::kVertical, x_cone.initial_odd, y_terminal_scale, x_odd_pair, routes);

    AxisPairSlots vertical_low_pair{
        .even = x_even_pair.even,
        .odd = x_odd_pair.even,
        .free = x_odd_pair.free,
    };
    append_axis_routes(x_cone, Lwt2DAxis::kHorizontal, y_cone.final_even, x_terminal_scale, vertical_low_pair, routes);

    AxisPairSlots vertical_high_pair{
        .even = x_even_pair.odd,
        .odd = x_odd_pair.odd,
        .free = vertical_low_pair.free,
    };
    append_axis_routes(x_cone, Lwt2DAxis::kHorizontal, y_cone.final_odd, x_terminal_scale, vertical_high_pair, routes);

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

    const TerminalScaleInline y_terminal_scale = execution_detail::terminal_scale_inline(y_plan);
    const TerminalScaleInline x_terminal_scale = execution_detail::terminal_scale_inline(x_plan);
    auto [routes, final_bands] = build_route_schedule(
        y_cone,
        x_cone,
        fuse_terminal_scale ? &y_terminal_scale : nullptr,
        fuse_terminal_scale ? &x_terminal_scale : nullptr);
    const Lwt2DResourceModel resources = make_resource_model(initial, routes);
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
                        .begin = tt::round_down(final_band_rect.y.begin, kTileHeight),
                        .end = tt::round_up(final_band_rect.y.end, static_cast<size_t>(kTileHeight)),
                    },
                .x =
                    IndexInterval{
                        .begin = tt::round_down(final_band_rect.x.begin, kTileWidth),
                        .end = tt::round_up(final_band_rect.x.end, static_cast<size_t>(kTileWidth)),
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
    const size_t chunk_rows = tt::div_up(y_plan.output_length, chunk_height);
    const size_t chunk_cols = tt::div_up(x_plan.output_length, chunk_width);
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
    uint64_t estimated_cost{0};
    std::vector<Lwt2DChunkPlan> chunks;
};

using AxisConeSignature = std::vector<int64_t>;

// Update the signature encoding below whenever either dependency structure changes.
static_assert(sizeof(AxisRouteRequirement) == 120);
static_assert(sizeof(AxisConePlan) == 88);

constexpr size_t kIntervalSignatureWords = 3;
constexpr size_t kConeHeaderSignatureWords = 4 * kIntervalSignatureWords + 1;
constexpr size_t kRouteSignatureWords = 1 + 7 * kIntervalSignatureWords;
constexpr size_t kOutputSignatureWords = 2;

inline void update_signature_anchor(const IndexInterval interval, size_t& anchor) noexcept {
    if (!interval.empty()) {
        anchor = std::min(anchor, interval.begin);
    }
}

inline void update_signature_anchor(const AxisConePlan& cone, size_t& anchor) noexcept {
    update_signature_anchor(cone.final_even, anchor);
    update_signature_anchor(cone.final_odd, anchor);
    update_signature_anchor(cone.initial_even, anchor);
    update_signature_anchor(cone.initial_odd, anchor);
    for (const AxisRouteRequirement& route : cone.routes) {
        update_signature_anchor(route.before.even, anchor);
        update_signature_anchor(route.before.odd, anchor);
        update_signature_anchor(route.after.even, anchor);
        update_signature_anchor(route.after.odd, anchor);
        update_signature_anchor(route.source, anchor);
        update_signature_anchor(route.base, anchor);
        update_signature_anchor(route.output, anchor);
    }
}

inline void append_interval_signature(
    AxisConeSignature& signature, const IndexInterval interval, const size_t anchor, const size_t tile_extent) {
    if (interval.empty()) {
        signature.push_back(-1);
        signature.push_back(0);
        signature.push_back(0);
        return;
    }
    TT_FATAL(
        interval.begin >= anchor && interval.end <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "2D planner axis signature exceeds int64_t geometry");
    signature.push_back(static_cast<int64_t>(interval.begin - anchor));
    signature.push_back(static_cast<int64_t>(interval.length()));
    signature.push_back(static_cast<int64_t>(interval.begin % tile_extent));
}

inline void append_cone_signature(
    AxisConeSignature& signature, const AxisConePlan& cone, const size_t anchor, const size_t tile_extent) {
    append_interval_signature(signature, cone.final_even, anchor, tile_extent);
    append_interval_signature(signature, cone.final_odd, anchor, tile_extent);
    append_interval_signature(signature, cone.initial_even, anchor, tile_extent);
    append_interval_signature(signature, cone.initial_odd, anchor, tile_extent);
    signature.push_back(static_cast<int64_t>(cone.routes.size()));
    for (const AxisRouteRequirement& route : cone.routes) {
        signature.push_back(static_cast<int64_t>(route.type));
        append_interval_signature(signature, route.before.even, anchor, tile_extent);
        append_interval_signature(signature, route.before.odd, anchor, tile_extent);
        append_interval_signature(signature, route.after.even, anchor, tile_extent);
        append_interval_signature(signature, route.after.odd, anchor, tile_extent);
        append_interval_signature(signature, route.source, anchor, tile_extent);
        append_interval_signature(signature, route.base, anchor, tile_extent);
        append_interval_signature(signature, route.output, anchor, tile_extent);
    }
}

[[nodiscard]] inline AxisConeSignature make_axis_cone_signature(
    const IndexInterval output,
    const AxisConePlan& exact_cone,
    const AxisConePlan* routed_cone,
    const size_t tile_extent) {
    size_t anchor = std::numeric_limits<size_t>::max();
    update_signature_anchor(exact_cone, anchor);
    if (routed_cone != nullptr) {
        update_signature_anchor(*routed_cone, anchor);
    }
    TT_FATAL(anchor != std::numeric_limits<size_t>::max(), "2D planner axis cone is empty");
    anchor = tt::round_down(anchor, tile_extent);

    AxisConeSignature signature;
    signature.reserve(
        kOutputSignatureWords + kConeHeaderSignatureWords + kRouteSignatureWords * exact_cone.routes.size() +
        (routed_cone != nullptr ? kConeHeaderSignatureWords + kRouteSignatureWords * routed_cone->routes.size() : 0));
    signature.push_back(static_cast<int64_t>(output.begin % tile_extent));
    signature.push_back(static_cast<int64_t>(output.length()));
    append_cone_signature(signature, exact_cone, anchor, tile_extent);
    if (routed_cone != nullptr) {
        append_cone_signature(signature, *routed_cone, anchor, tile_extent);
    }
    return signature;
}

struct AxisChunkClasses {
    std::vector<IndexInterval> representatives;
    std::vector<uint32_t> class_ids;
};

template <typename SignatureBuilder>
[[nodiscard]] AxisChunkClasses make_axis_chunk_classes(
    const size_t length, const size_t chunk_extent, const SignatureBuilder& build_signature) {
    TT_FATAL(chunk_extent > 0, "2D planner chunk extent must be positive");
    std::vector<AxisConeSignature> signatures;
    AxisChunkClasses classes;
    classes.class_ids.reserve(tt::div_up(length, chunk_extent));
    for (size_t begin = 0; begin < length;) {
        const size_t end = begin + std::min(chunk_extent, length - begin);
        const IndexInterval interval{.begin = begin, .end = end};
        AxisConeSignature signature = build_signature(interval);
        const auto existing = std::find(signatures.begin(), signatures.end(), signature);
        if (existing == signatures.end()) {
            signatures.push_back(std::move(signature));
            classes.representatives.push_back(interval);
            classes.class_ids.push_back(static_cast<uint32_t>(signatures.size() - 1));
        } else {
            classes.class_ids.push_back(static_cast<uint32_t>(std::distance(signatures.begin(), existing)));
        }
        begin = end;
    }
    return classes;
}

template <typename SignatureBuilder>
[[nodiscard]] std::vector<AxisChunkClasses> make_axis_candidate_classes(
    const size_t length,
    const uint32_t maximum_chunk_tiles,
    const size_t tile_extent,
    const SignatureBuilder& build_signature) {
    // Chunk-tile counts are one-based, so index zero is intentionally unused.
    std::vector<AxisChunkClasses> classes(static_cast<size_t>(maximum_chunk_tiles) + 1);
    for (uint32_t chunk_tiles = 1; chunk_tiles <= maximum_chunk_tiles; ++chunk_tiles) {
        classes[chunk_tiles] =
            make_axis_chunk_classes(length, static_cast<size_t>(chunk_tiles) * tile_extent, build_signature);
    }
    return classes;
}

struct RowMajorScheduleEstimate {
    uint32_t active_core_count{0};
    uint64_t cost{0};
};

[[nodiscard]] inline RowMajorScheduleEstimate estimate_row_major_schedule(
    const AxisChunkClasses& y_classes,
    const AxisChunkClasses& x_classes,
    const std::vector<uint64_t>& representative_costs,
    const uint32_t core_limit,
    const uint64_t penalty_per_core) {
    const size_t class_columns = x_classes.representatives.size();
    const size_t chunk_count = checked_area(y_classes.class_ids.size(), x_classes.class_ids.size(), "2D chunk grid");
    const uint32_t active_core_count = static_cast<uint32_t>(std::min(chunk_count, static_cast<size_t>(core_limit)));
    const size_t base = chunk_count / active_core_count;
    const size_t extra = chunk_count % active_core_count;

    std::vector<std::vector<uint64_t>> row_prefixes(
        y_classes.representatives.size(), std::vector<uint64_t>(x_classes.class_ids.size() + 1, 0));
    for (size_t y_class = 0; y_class < row_prefixes.size(); ++y_class) {
        auto& prefix = row_prefixes[y_class];
        for (size_t column = 0; column < x_classes.class_ids.size(); ++column) {
            const size_t cost_index = y_class * class_columns + x_classes.class_ids[column];
            prefix[column + 1] = prefix[column] + representative_costs[cost_index];
        }
    }
    std::vector<uint64_t> row_cost_prefix(y_classes.class_ids.size() + 1, 0);
    for (size_t row = 0; row < y_classes.class_ids.size(); ++row) {
        row_cost_prefix[row + 1] = row_cost_prefix[row] + row_prefixes[y_classes.class_ids[row]].back();
    }
    const auto prefix_cost = [&](const size_t linear_index) {
        const size_t row = linear_index / x_classes.class_ids.size();
        const size_t column = linear_index % x_classes.class_ids.size();
        return row_cost_prefix[row] +
               (row < y_classes.class_ids.size() ? row_prefixes[y_classes.class_ids[row]][column] : 0U);
    };

    size_t linear_index = 0;
    uint64_t estimated_cost = 0;
    for (uint32_t core = 0; core < active_core_count; ++core) {
        const size_t count = base + (core < extra ? 1U : 0U);
        const uint64_t core_cost =
            planner_cost_model::kCoreStartup + prefix_cost(linear_index + count) - prefix_cost(linear_index);
        estimated_cost = std::max(estimated_cost, core_cost);
        linear_index += count;
    }
    if (penalty_per_core > 0 && active_core_count > 64) {
        estimated_cost += static_cast<uint64_t>(active_core_count - 64) * penalty_per_core;
    }
    return RowMajorScheduleEstimate{
        .active_core_count = active_core_count,
        .cost = estimated_cost,
    };
}

template <typename ChunkBuilder, typename CostEstimator>
[[nodiscard]] std::optional<Candidate> evaluate_candidate(
    const AxisChunkClasses& y_classes,
    const AxisChunkClasses& x_classes,
    const uint32_t chunk_tiles_y,
    const uint32_t chunk_tiles_x,
    const uint32_t core_limit,
    const uint64_t l1_budget_bytes,
    ChunkBuilder&& build_chunk,
    CostEstimator&& estimate_cost,
    const uint64_t penalty_per_core) {
    const size_t class_columns = x_classes.representatives.size();
    std::vector<uint64_t> costs;
    costs.reserve(checked_area(y_classes.representatives.size(), class_columns, "2D planner chunk classes"));
    double max_dependency_overhead = 0.0;
    for (const IndexInterval y : y_classes.representatives) {
        for (const IndexInterval x : x_classes.representatives) {
            const Lwt2DChunkPlan chunk = build_chunk(IndexRectangle{.y = y, .x = x});
            if (chunk.resources.total_l1_bytes > l1_budget_bytes) {
                return std::nullopt;
            }
            max_dependency_overhead = std::max(max_dependency_overhead, chunk.dependency_overhead);
            costs.push_back(estimate_cost(chunk));
        }
    }

    const RowMajorScheduleEstimate schedule =
        estimate_row_major_schedule(y_classes, x_classes, costs, core_limit, penalty_per_core);
    return Candidate{
        .chunk_tiles_y = chunk_tiles_y,
        .chunk_tiles_x = chunk_tiles_x,
        .active_core_count = schedule.active_core_count,
        .max_dependency_overhead = max_dependency_overhead,
        .estimated_cost = schedule.cost,
        .chunks = {},
    };
}

[[nodiscard]] constexpr uint64_t maximum_candidate_tile_area(const uint64_t l1_budget_bytes) noexcept {
    const uint64_t fixed_bytes = kCircularBufferBytes + kMetadataBytes + kSynchronizationBytes;
    return l1_budget_bytes < fixed_bytes ? 0 : (l1_budget_bytes - fixed_bytes) / kFullFp32TileBytes;
}

[[nodiscard]] inline AxisConeSignature make_forward_axis_signature(
    const LiftingForwardPlan& plan,
    const IndexInterval output,
    const size_t tile_extent,
    const Lwt2DRouteDomainPolicy route_domain,
    const char* even_label,
    const char* odd_label) {
    const size_t tap_size = static_cast<size_t>(plan.preprocess_layout.pad_config.left) + 1;
    const IndexInterval final_even =
        canonical_to_stream_interval(output, plan.final_even_shift, plan.final_even_length, tap_size / 2, even_label);
    const IndexInterval final_odd =
        canonical_to_stream_interval(output, plan.final_odd_shift, plan.final_odd_length, tap_size / 2, odd_label);
    const AxisConePlan exact_cone = build_axis_cone(plan, final_even, final_odd);
    if (route_domain == Lwt2DRouteDomainPolicy::kExact) {
        return make_axis_cone_signature(output, exact_cone, nullptr, tile_extent);
    }
    const AxisConePlan routed_cone = build_axis_cone(plan, final_even, final_odd, tile_extent);
    return make_axis_cone_signature(output, exact_cone, &routed_cone, tile_extent);
}

enum class AlignmentCostClass : uint8_t {
    kExact,
    kOneAxisShifted,
    kGeneric,
};

[[nodiscard]] inline int64_t signed_tile_modulo(const int64_t value, const uint32_t tile_extent) noexcept {
    const int64_t remainder = value % static_cast<int64_t>(tile_extent);
    return remainder < 0 ? remainder + static_cast<int64_t>(tile_extent) : remainder;
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
    const bool y_aligned = signed_tile_modulo(requested_y, kTileHeight) == 0;
    const bool x_aligned = signed_tile_modulo(requested_x, kTileWidth) == 0;
    if (y_aligned && x_aligned) {
        return AlignmentCostClass::kExact;
    }
    if (y_aligned != x_aligned) {
        return AlignmentCostClass::kOneAxisShifted;
    }
    return AlignmentCostClass::kGeneric;
}

[[nodiscard]] constexpr uint64_t staging_class_cost(const AlignmentCostClass tile_class) noexcept {
    switch (tile_class) {
        case AlignmentCostClass::kExact: return planner_cost_model::kExactStaging;
        case AlignmentCostClass::kOneAxisShifted: return planner_cost_model::kOneAxisShiftedStaging;
        case AlignmentCostClass::kGeneric: return planner_cost_model::kGenericStaging;
    }
    return planner_cost_model::kUnknownStaging;
}

[[nodiscard]] inline uint64_t estimate_chunk_cost(
    const Lwt2DChunkPlan& chunk,
    const LiftingForwardPlan& y_plan,
    const LiftingForwardPlan& x_plan,
    const bool inverse = false) {
    uint64_t cost = chunk.initial.total_area() * planner_cost_model::kInitialElement;
    std::array<IndexRectangle, 5> stored = {
        chunk.initial.ee,
        chunk.initial.eo,
        chunk.initial.oe,
        chunk.initial.oo,
        IndexRectangle{},
    };
    for (const Lwt2DRoutePlan& route : chunk.routes) {
        cost += planner_cost_model::kRouteConfigAndSync;
        if (route.output.empty()) {
            continue;
        }
        const LiftingStepRoute& axis_route = route.axis == Lwt2DAxis::kVertical ? y_plan.routes[route.axis_route_index]
                                                                                : x_plan.routes[route.axis_route_index];
        const uint32_t k = is_predict_update_step(route.type) ? execution_detail::coefficient_count(axis_route) : 1U;
        const int64_t output_y_origin = static_cast<int64_t>(tt::round_down(route.output.y.begin, kTileHeight));
        const int64_t output_x_origin = static_cast<int64_t>(tt::round_down(route.output.x.begin, kTileWidth));
        const size_t tile_rows = tt::round_up(route.output.y.end, static_cast<size_t>(kTileHeight)) / kTileHeight -
                                 route.output.y.begin / kTileHeight;
        const size_t tile_columns = tt::round_up(route.output.x.end, static_cast<size_t>(kTileWidth)) / kTileWidth -
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
                    cost +=
                        staging_class_cost(alignment_cost_class(stored[slot_index(route.base_slot)], base_y, base_x));
                    const auto [source_y, source_x] = requested_origin(route.source);
                    for (uint32_t source_tile = 0; source_tile < 2; ++source_tile) {
                        const int64_t requested_y =
                            source_y +
                            (route.axis == Lwt2DAxis::kVertical ? static_cast<int64_t>(source_tile * kTileHeight) : 0);
                        const int64_t requested_x = source_x + (route.axis == Lwt2DAxis::kHorizontal
                                                                    ? static_cast<int64_t>(source_tile * kTileWidth) -
                                                                          static_cast<int64_t>(17 - k)
                                                                    : 0);
                        cost += staging_class_cost(
                            alignment_cost_class(stored[slot_index(route.source_slot)], requested_y, requested_x));
                    }
                    cost += route.axis == Lwt2DAxis::kVertical
                                ? planner_cost_model::kVerticalStencilBase + planner_cost_model::kVerticalStencilTap * k
                                : planner_cost_model::kHorizontalStencilBase +
                                      planner_cost_model::kHorizontalStencilTap * k;
                } else {
                    const auto [source_y, source_x] = requested_origin(route.source);
                    cost += staging_class_cost(
                        alignment_cost_class(stored[slot_index(route.source_slot)], source_y, source_x));
                    cost += planner_cost_model::kNonStencilRoute;
                }
                cost += planner_cost_model::kFullTilePersistence;
            }
        }
        stored[slot_index(route.output_slot)] = route.output;
    }

    const bool full_terminal_tiles =
        chunk.final_band_rect.y.begin % kTileHeight == 0 && chunk.final_band_rect.x.begin % kTileWidth == 0 &&
        chunk.final_band_rect.height() % kTileHeight == 0 && chunk.final_band_rect.width() % kTileWidth == 0;
    const uint64_t terminal_tiles =
        static_cast<uint64_t>(tt::div_up(chunk.final_band_rect.height(), static_cast<size_t>(kTileHeight))) *
        tt::div_up(chunk.final_band_rect.width(), static_cast<size_t>(kTileWidth));
    if (inverse) {
        cost += terminal_tiles * planner_cost_model::kInterleavedTerminalTile;
    } else {
        cost += device_protocol::kLwt2DBandCount * terminal_tiles *
                (full_terminal_tiles ? planner_cost_model::kTiledTerminalTile
                                     : planner_cost_model::kFragmentedTerminalTile);
    }
    return cost;
}

[[nodiscard]] inline bool is_better_candidate(
    const Candidate& candidate, const Candidate& best, const bool latency_oriented) noexcept {
    if (latency_oriented && candidate.estimated_cost != best.estimated_cost) {
        if (candidate.active_core_count < best.active_core_count) {
            return static_cast<long double>(candidate.estimated_cost) <
                   planner_cost_model::kFewerCoresCostRatio * static_cast<long double>(best.estimated_cost);
        }
        if (candidate.active_core_count > best.active_core_count) {
            return static_cast<long double>(candidate.estimated_cost) <=
                   planner_cost_model::kMoreCoresCostRatio * static_cast<long double>(best.estimated_cost);
        }
        return candidate.estimated_cost < best.estimated_cost;
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
        y_plan.preprocess_layout.input.length <= kMax2DLogicalExtent &&
            x_plan.preprocess_layout.input.length <= kMax2DLogicalExtent,
        "2D LWT input dimensions {}x{} exceed the signed device-coordinate limit {}",
        y_plan.preprocess_layout.input.length,
        x_plan.preprocess_layout.input.length,
        kMax2DLogicalExtent);
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

    const size_t band_tiles_y_size = tt::div_up(y_plan.output_length, static_cast<size_t>(plan_2d_detail::kTileHeight));
    const size_t band_tiles_x_size = tt::div_up(x_plan.output_length, static_cast<size_t>(plan_2d_detail::kTileWidth));
    TT_FATAL(
        band_tiles_y_size <= std::numeric_limits<uint32_t>::max() &&
            band_tiles_x_size <= std::numeric_limits<uint32_t>::max(),
        "2D LWT band tile grid {}x{} exceeds uint32_t planner geometry",
        band_tiles_y_size,
        band_tiles_x_size);
    const uint32_t band_tiles_y = static_cast<uint32_t>(band_tiles_y_size);
    const uint32_t band_tiles_x = static_cast<uint32_t>(band_tiles_x_size);

    const uint64_t maximum_tile_area = plan_2d_detail::maximum_candidate_tile_area(l1_budget_bytes);
    const uint32_t maximum_tiles_y = static_cast<uint32_t>(std::min<uint64_t>(band_tiles_y, maximum_tile_area));
    const uint32_t maximum_tiles_x = static_cast<uint32_t>(std::min<uint64_t>(band_tiles_x, maximum_tile_area));
    const auto y_candidate_classes = plan_2d_detail::make_axis_candidate_classes(
        y_plan.output_length, maximum_tiles_y, plan_2d_detail::kTileHeight, [&](const IndexInterval output) {
            return plan_2d_detail::make_forward_axis_signature(
                y_plan, output, plan_2d_detail::kTileHeight, route_domain, "vertical even", "vertical odd");
        });
    const auto x_candidate_classes = plan_2d_detail::make_axis_candidate_classes(
        x_plan.output_length, maximum_tiles_x, plan_2d_detail::kTileWidth, [&](const IndexInterval output) {
            return plan_2d_detail::make_forward_axis_signature(
                x_plan, output, plan_2d_detail::kTileWidth, route_domain, "horizontal even", "horizontal odd");
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
                [&](const IndexRectangle output) {
                    return plan_2d_detail::build_chunk(y_plan, x_plan, output, fuse_terminal_scale, route_domain);
                },
                [&](const Lwt2DChunkPlan& chunk) { return plan_2d_detail::estimate_chunk_cost(chunk, y_plan, x_plan); },
                0);
            if (!candidate.has_value()) {
                continue;
            }
            if (!found || plan_2d_detail::is_better_candidate(*candidate, best, latency_oriented_planner)) {
                best = std::move(*candidate);
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
    best.chunks = plan_2d_detail::build_chunks(
        y_plan, x_plan, best.chunk_tiles_y, best.chunk_tiles_x, fuse_terminal_scale, route_domain);
    const size_t input_height = y_plan.preprocess_layout.input.length;
    const size_t input_width = x_plan.preprocess_layout.input.length;
    const size_t band_height = y_plan.output_length;
    const size_t band_width = x_plan.output_length;
    const Lwt2DTilingContract tiling{
        .input = TiledShape2D::from_logical(Shape2D{.height = input_height, .width = input_width}),
        .band = TiledShape2D::from_logical(Shape2D{.height = band_height, .width = band_width}),
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
    TT_FATAL(
        input_height <= kMax2DLogicalExtent && input_width <= kMax2DLogicalExtent,
        "2D LWT input dimensions {}x{} exceed the signed device-coordinate limit {}",
        input_height,
        input_width,
        kMax2DLogicalExtent);
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
