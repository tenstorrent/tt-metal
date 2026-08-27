// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "neighborhood_sdpa_nanobind.hpp"

#include <array>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn/operations/transformer/sdpa/device/kernels/neighborhood_kernel_args.hpp"
#include "ttnn/operations/transformer/sdpa/device/neighborhood_sdpa_device_operation.hpp"

namespace ttnn::operations::transformer {

namespace {

namespace neighborhood = ttnn::transformer::neighborhood;
namespace kernel_args = ttnn::transformer::neighborhood::kernel_args;

using AxisTuple = std::array<uint32_t, 3>;
// Signed, because a halo at the low edge of the volume puts a shard at a negative origin.
using SignedAxisTuple = std::array<int32_t, 3>;

neighborhood::Extent3 to_extent(const AxisTuple& value) {
    return neighborhood::Extent3{{value[0], value[1], value[2]}};
}

neighborhood::NeighborhoodConfig to_config(
    const AxisTuple& volume,
    const AxisTuple& context_window,
    const AxisTuple& stride,
    const AxisTuple& brick,
    const std::optional<AxisTuple>& query_chunk_bricks = std::nullopt,
    const std::optional<AxisTuple>& shard_extent = std::nullopt,
    const std::optional<SignedAxisTuple>& shard_origin = std::nullopt,
    const std::optional<AxisTuple>& query_extent = std::nullopt,
    const std::optional<AxisTuple>& query_origin = std::nullopt) {
    neighborhood::NeighborhoodConfig config{
        to_extent(volume), to_extent(context_window), to_extent(stride), to_extent(brick)};
    if (query_chunk_bricks.has_value()) {
        config.query_chunk_bricks = to_extent(*query_chunk_bricks);
    }
    if (shard_extent.has_value()) {
        config.shard_extent = to_extent(*shard_extent);
    }
    if (shard_origin.has_value()) {
        const SignedAxisTuple& origin = *shard_origin;
        config.shard_origin = neighborhood::SiteOffset::at(origin[0], origin[1], origin[2]);
    }
    if (query_extent.has_value()) {
        config.query_extent = to_extent(*query_extent);
    }
    if (query_origin.has_value()) {
        const AxisTuple& origin = *query_origin;
        config.query_origin = neighborhood::Site::at(origin[0], origin[1], origin[2]);
    }
    return config;
}

}  // namespace

void bind_neighborhood_sdpa(nb::module_& mod) {
    mod.def(
        "neighborhood_choose_brick",
        [](const AxisTuple& context_window) {
            const neighborhood::Extent3 brick = neighborhood::choose_brick(to_extent(context_window));
            return AxisTuple{brick.time(), brick.height(), brick.width()};
        },
        nb::arg("context_window"),
        R"doc(
        The 32-site brick shape that minimises what one tile row must gather for this context
        window. A function, not a constant: a window flat in time wants a brick flat in time.
        )doc");

    mod.def(
        "neighborhood_plan",
        [](const AxisTuple& volume,
           const AxisTuple& context_window,
           const AxisTuple& stride,
           const AxisTuple& brick,
           const std::optional<AxisTuple>& query_chunk_bricks,
           const std::optional<AxisTuple>& shard_extent,
           const std::optional<SignedAxisTuple>& shard_origin,
           const std::optional<AxisTuple>& query_extent,
           const std::optional<AxisTuple>& query_origin) {
            const neighborhood::NeighborhoodPlan plan = neighborhood::build_plan(to_config(
                volume,
                context_window,
                stride,
                brick,
                query_chunk_bricks,
                shard_extent,
                shard_origin,
                query_extent,
                query_origin));

            // Flattened gather origin table, GATHER_ORIGIN_COLUMNS wide so each chunk's entry is
            // one DRAM-aligned page. Columns carry this chunk's gather origin (local sites) and
            // this device's shard origin (global sites) -- see kernel_args::gather_origin_column.
            std::vector<uint32_t> gather_origin_table(
                static_cast<size_t>(plan.chunk_count) * kernel_args::GATHER_ORIGIN_COLUMNS, 0u);
            for (uint32_t chunk_index = 0; chunk_index < plan.chunk_count; ++chunk_index) {
                const neighborhood::Site& origin = plan.gather_origin_by_chunk[chunk_index];
                const size_t row = static_cast<size_t>(chunk_index) * kernel_args::GATHER_ORIGIN_COLUMNS;
                namespace column = kernel_args::gather_origin_column;
                gather_origin_table[row + column::gather_time] = origin.time();
                gather_origin_table[row + column::gather_height] = origin.height();
                gather_origin_table[row + column::gather_width] = origin.width();
                // Signed, reinterpreted bit-for-bit: the table is uint32 but a low-edge halo
                // makes this negative, and the kernel reads it back as int32.
                gather_origin_table[row + column::shard_origin_time] =
                    static_cast<uint32_t>(plan.config.shard_origin.time());
                gather_origin_table[row + column::shard_origin_height] =
                    static_cast<uint32_t>(plan.config.shard_origin.height());
                gather_origin_table[row + column::shard_origin_width] =
                    static_cast<uint32_t>(plan.config.shard_origin.width());
            }

            nb::dict result;
            result["brick_count"] = plan.brick_count;
            // The QUERY grid. Equal to brick_count / volume_bricks unless a query sub-region was
            // asked for; the host sizes Q and the output from these, K and V from the pair above.
            result["query_brick_count"] = plan.query_brick_count;
            result["query_bricks"] =
                AxisTuple{plan.query_bricks.time(), plan.query_bricks.height(), plan.query_bricks.width()};
            result["chunk_count"] = plan.chunk_count;
            result["volume_chunks"] =
                AxisTuple{plan.volume_chunks.time(), plan.volume_chunks.height(), plan.volume_chunks.width()};
            result["bricks_per_query_chunk"] = plan.config.bricks_per_query_chunk();
            result["volume_bricks"] =
                AxisTuple{plan.volume_bricks.time(), plan.volume_bricks.height(), plan.volume_bricks.width()};
            result["gather_extent"] =
                AxisTuple{plan.gather_extent.time(), plan.gather_extent.height(), plan.gather_extent.width()};
            result["gather_sites"] = plan.gather_sites;
            result["gather_tiles"] = plan.gather_tiles;
            result["gather_bricks"] =
                AxisTuple{plan.gather_bricks.time(), plan.gather_bricks.height(), plan.gather_bricks.width()};
            result["gather_brick_count"] = plan.gather_brick_count;
            result["gather_origin_table"] = gather_origin_table;
            result["gather_origin_columns"] = kernel_args::GATHER_ORIGIN_COLUMNS;
            return result;
        },
        nb::arg("volume"),
        nb::arg("context_window"),
        nb::arg("stride"),
        nb::arg("brick"),
        nb::arg("query_chunk_bricks") = nb::none(),
        nb::arg("shard_extent") = nb::none(),
        nb::arg("shard_origin") = nb::none(),
        nb::arg("query_extent") = nb::none(),
        nb::arg("query_origin") = nb::none(),
        R"doc(
        Build the neighborhood plan for one geometry. `volume` is always the GLOBAL grid;
        `shard_extent`/`shard_origin` say what this device holds and where it sits, so windows
        clamp at the true volume edge rather than at a shard seam. Depends on no weights, so build it once
        per (volume, context_window, stride, brick) and keep the uploaded table resident --
        rebuilding it per block would dominate.

        `query_chunk_bricks` sets how many bricks share one gather. Keys-gathered-per-query is
        `gather_sites / chunk_queries`, so a one-brick chunk re-gathers nearly the same keys for
        every tile row: an 11^3 window costs 54 keys per query at one brick and 4.8 at 5x2x2. The
        chunk cannot exceed the context window on any axis, since stride <= window.

        Returns the gather origin table flattened row-major, `gather_origin_columns` wide, one row
        per CHUNK.
        )doc");

    mod.def(
        "neighborhood_scaled_dot_product_attention",
        [](const ttnn::Tensor& query_tensor,
           const ttnn::Tensor& key_tensor,
           const ttnn::Tensor& value_tensor,
           const ttnn::Tensor& gather_origin_table,
           const std::optional<ttnn::Tensor>& interior_mask,
           const AxisTuple& volume,
           const AxisTuple& context_window,
           const AxisTuple& stride,
           const AxisTuple& brick,
           const std::optional<AxisTuple>& query_chunk_bricks,
           const std::optional<AxisTuple>& shard_extent,
           const std::optional<SignedAxisTuple>& shard_origin,
           const std::optional<AxisTuple>& query_extent,
           const std::optional<AxisTuple>& query_origin,
           uint32_t head_count,
           float scale,
           uint32_t tiles_per_kv_chunk,
           const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
           std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
            return ttnn::prim::neighborhood_sdpa(
                query_tensor,
                key_tensor,
                value_tensor,
                gather_origin_table,
                interior_mask,
                to_config(
                    volume,
                    context_window,
                    stride,
                    brick,
                    query_chunk_bricks,
                    shard_extent,
                    shard_origin,
                    query_extent,
                    query_origin),
                head_count,
                scale,
                tiles_per_kv_chunk,
                memory_config.value_or(query_tensor.memory_config()),
                compute_kernel_config.value_or(DeviceComputeKernelConfig{}));
        },
        nb::arg("query_tensor"),
        nb::arg("key_tensor"),
        nb::arg("value_tensor"),
        nb::arg("gather_origin_table"),
        nb::kw_only(),
        nb::arg("interior_mask") = nb::none(),
        nb::arg("volume"),
        nb::arg("context_window"),
        nb::arg("stride") = AxisTuple{1, 1, 1},
        nb::arg("brick"),
        nb::arg("query_chunk_bricks") = nb::none(),
        nb::arg("shard_extent") = nb::none(),
        nb::arg("shard_origin") = nb::none(),
        nb::arg("query_extent") = nb::none(),
        nb::arg("query_origin") = nb::none(),
        nb::arg("head_count"),
        nb::arg("scale") = 1.0f,
        nb::arg("tiles_per_kv_chunk") = 8,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        R"doc(
        3D neighborhood attention. Query, key and value are in BRICKED site order --
        `[batch, 1, brick_count * 32, head_count * head_dim]`, TILE layout -- so one tile row is
        one compact 3D box of the volume rather than a pencil along width. Heads are the COLUMN
        axis, not a leading dimension, so nothing has to transpose heads against sites. Use
        `models.tt_dit.layers.neighborhood_permute` to get in and out of that order.
        )doc");
}

}  // namespace ttnn::operations::transformer
