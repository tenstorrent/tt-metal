// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tracy/Tracy.hpp>
#include "moe_tail.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/experimental/moe_block/common/host_floats.hpp"
#include "ttnn/operations/experimental/moe_block/common/l1_budget.hpp"
#include "ttnn/operations/experimental/moe_block/common/moe_block_resources.hpp"

#include "device/moe_tail_device_operation.hpp"

namespace ttnn::operations::experimental::moe_block::moe_tail {

namespace moe_tail_entry_detail {

using EnginePlanPtr = std::shared_ptr<const TailEnginePlan>;
using AllToAllPlanPtr = std::shared_ptr<const TailAllToAllPlan>;

// W_up's tile bytes: tt-metal's tile_size of the dtype (bfloat8_b 1088 B: 1024 B of mantissas + 64 B of shared
// exponents; bfloat16 2048 B), the formats the engine's matmul takes.
uint32_t w_up_tile_bytes(DataType dtype) {
    TT_FATAL(
        dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT16,
        "moe_tail: W_up as bfloat8_b or bfloat16, got {}",
        dtype);
    return tt::tt_metal::tile_size(dtype);
}

// A row block's tokens of a mesh row: a call's (one split's) / SP / the head's row blocks.
uint32_t rows_of(const MoEBlockResources& resources, uint32_t row_blocks) {
    TT_FATAL(resources.config().chunk_tokens != 0, "moe_tail: MoEBlockConfig.chunk_tokens is not set");
    TT_FATAL(row_blocks >= 1, "moe_tail: row_blocks {} (at least 1)", row_blocks);
    const uint32_t rows = resources.config().split_tokens() / resources.topology().R;
    TT_FATAL(rows % row_blocks == 0, "moe_tail: a mesh row's {} rows over {} row blocks", rows, row_blocks);
    return rows / row_blocks;
}

const EnginePlanPtr& engine_plan(MoEBlockResources& resources, uint32_t w_tile_bytes, uint32_t row_blocks) {
    const auto& dims = resources.dims();
    const uint32_t tp_size = resources.topology().C;
    TT_FATAL(dims.latent_dim.has_value(), "moe_tail: the engine is a latent model's tail");
    TT_FATAL(dims.latent_norm, "moe_tail: the engine always rms-normalizes the latent (latent_norm off)");
    TT_FATAL(
        dims.emb_dim % (tt::constants::TILE_WIDTH * tp_size) == 0,
        "moe_tail: H {} over TP {}: whole tiles",
        dims.emb_dim,
        tp_size);
    const auto& plan = resources.plan<EnginePlanPtr>("moe_tail/engine_plan/b" + std::to_string(row_blocks), [&] {
        return make_tail_engine_plan(
            resources,
            rows_of(resources, row_blocks),
            *dims.latent_dim,
            dims.emb_dim / tp_size,
            w_tile_bytes,
            free_cb_l1_bytes(resources.mesh_device(), kL1KeptLater, kMaxArenaBytes));
    });
    TT_FATAL(
        plan->geometry.w_tile_bytes == w_tile_bytes,
        "moe_tail: the engine was planned for {} B W' tiles, got {}",
        plan->geometry.w_tile_bytes,
        w_tile_bytes);
    return plan;
}

}  // namespace moe_tail_entry_detail
using namespace moe_tail_entry_detail;

Tensor moe_tail(
    const Tensor& y_own,
    const std::optional<Tensor>& shared_expert,
    MoEBlockResources& resources,
    uint32_t row_blocks,
    const std::optional<Tensor>& w_up,
    tt::tt_metal::MathFidelity math_fidelity,
    const std::optional<Tensor>& output,
    const std::optional<Tensor>& actual_start,
    const std::optional<Tensor>& actual_end,
    const std::optional<Tensor>& metadata,
    uint32_t split) {
    ZoneScopedN("moe_tail: entry");
    const auto& dims = resources.dims();
    TT_FATAL(row_blocks >= 1, "moe_tail: row_blocks {} (the head's MoEHeadOutputs.row_blocks)", row_blocks);
    TT_FATAL(y_own.logical_shape()[-2] % row_blocks == 0, "moe_tail: y_own's rows over {} row blocks", row_blocks);
    MoETailParams params{
        .plan_key = resources.plan_key(),
        .math_fidelity = math_fidelity,
        .row_blocks = row_blocks,
        .output_memory_config = output.has_value() ? output->memory_config() : ttnn::DRAM_MEMORY_CONFIG,
    };
    MoETailInputs inputs{.y_own = y_own, .shared_expert = shared_expert, .w_up = w_up, .output = output};
    // (a padded forward - moe_head's span: the all-to-all skips its padding row tiles; the engine computes every row)
    TT_FATAL(
        !metadata.has_value() || (!actual_start.has_value() && !actual_end.has_value()),
        "moe_tail: metadata or actual_start / actual_end");
    TT_FATAL(actual_start.has_value() == actual_end.has_value(), "moe_tail: actual_start and actual_end together");
    const auto& config = resources.config();
    TT_FATAL(split < config.splits(), "moe_tail: split {} of {}", split, config.splits());
    if (!dims.latent_dim.has_value() && (actual_start.has_value() || metadata.has_value())) {
        TT_FATAL(
            std::max<uint32_t>(1, config.forward_chunks) == 1 || metadata.has_value(),
            "moe_tail: a forward of {} chunks takes metadata",
            config.forward_chunks);
        inputs.actual_start = actual_start;
        inputs.actual_end = actual_end;
        inputs.metadata = metadata;
        params.first_block = split * row_blocks;
        params.forward_chunks = std::max<uint32_t>(1, config.forward_chunks);
        params.row_tokens = config.chunk_tokens / resources.topology().R;
    }
    if (dims.latent_dim.has_value()) {
        TT_FATAL(w_up.has_value(), "moe_tail: a latent model's tail needs w_up (prepare_moe_tail_weights)");
        TT_FATAL(math_fidelity != tt::tt_metal::MathFidelity::LoFi, "moe_tail: the up projection is never LoFi");
        const auto& plan = engine_plan(resources, w_up_tile_bytes(w_up->dtype()), row_blocks);
        const auto& geometry = plan->geometry;
        params.kind = TailKind::Engine;
        params.ring_slots = geometry.ring_slots;
        params.engine_plan = plan;
        // (the ports' ready: the fence's one persistent word - the engine's per-step words are a per-call CB, zeroed
        // each call behind the engine's zero gate, so no L1_SMALL grows with the rows)
        inputs.ready = resources.persistent_words("moe_tail/engine_ready", plan->port_cores, 4);
    } else {
        TT_FATAL(!w_up.has_value(), "moe_tail: a model without a latent takes no w_up");
        const auto& plan = resources.plan<AllToAllPlanPtr>(
            "moe_tail/all_to_all_plan/b" + std::to_string(row_blocks),
            [&] { return make_tail_all_to_all_plan(resources, rows_of(resources, row_blocks), dims.emb_dim); });
        const auto& geometry = plan->geometry;
        params.kind = TailKind::AllToAll;
        params.all_to_all_plan = plan;
        // (the fence's arrival and ready words; the z-ready words are a per-call CB on the ports)
        inputs.arrival = resources.persistent_words("moe_tail/all_to_all_arrival", plan->port_union, 4);
        inputs.ready = resources.persistent_words("moe_tail/all_to_all_ready", plan->port_union, 4);
        inputs.staging = resources.persistent_tensor(
            "moe_tail/all_to_all_staging",
            tt::tt_metal::TensorSpec(
                ttnn::Shape({1, 1, geometry.owned_rows, geometry.hidden_dim}),
                tt::tt_metal::TensorLayout(
                    DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG)));
    }
    return ttnn::prim::moe_block_tail(params, std::move(inputs));
}

void plan_moe_tail(MoEBlockResources& resources, uint32_t row_blocks, DataType dtype) {
    const auto& dims = resources.dims();
    const uint32_t rows = rows_of(resources, row_blocks);
    if (dims.latent_dim.has_value()) {
        (void)make_tail_engine_plan(
            resources,
            rows,
            *dims.latent_dim,
            dims.emb_dim / resources.topology().C,
            w_up_tile_bytes(dtype),
            free_cb_l1_bytes(resources.mesh_device(), kL1KeptLater, kMaxArenaBytes));
    } else {
        (void)make_tail_all_to_all_plan(resources, rows, dims.emb_dim);
    }
}

Tensor prepare_moe_tail_weights(
    const Tensor& latent_up,
    const Tensor& latent_norm,
    MoEBlockResources& resources,
    uint32_t row_blocks,
    DataType dtype) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    auto& mesh = resources.mesh_device();
    const auto& dims = resources.dims();
    const auto& topology = resources.topology();
    const auto& plan = *engine_plan(resources, w_up_tile_bytes(dtype), row_blocks);
    const auto& geometry = plan.geometry;
    const uint32_t hidden_dim = dims.emb_dim, latent_dim = geometry.latent_dim, output_dim = geometry.output_dim;
    TT_FATAL(
        latent_up.logical_volume() == static_cast<uint64_t>(hidden_dim) * latent_dim &&
            latent_up.logical_shape()[-1] == latent_dim,
        "moe_tail: latent_up must be [H {}, latent_dim {}], got {}",
        hidden_dim,
        latent_dim,
        latent_up.logical_shape());
    TT_FATAL(latent_norm.logical_volume() == latent_dim, "moe_tail: latent_norm must be [latent_dim {}]", latent_dim);
    const auto latent_up_values = to_floats(latent_up, "moe_tail");
    const auto norm_weight = to_floats(latent_norm, "moe_tail");

    // Region (K half, engine column, W' K range) = [K range][the column's output tiles] tiles of the half's K rows of
    // the column's output columns, padded to region_tiles; regions dealt over the banks (region i in bank i % banks,
    // slot i / banks): one burst each.
    const uint32_t banks = plan.num_dram_banks, region_tiles = geometry.region_tiles();
    const uint32_t num_regions = 2 * geometry.engine_columns * geometry.num_pairs;
    const uint32_t regions_per_bank = (num_regions + banks - 1) / banks;
    const uint32_t host_rows = regions_per_bank * region_tiles * TILE_HEIGHT, host_columns = banks * TILE_WIDTH;
    const tt::tt_metal::MemoryConfig memory_config{
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::DRAM,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange({0, 0}, {banks - 1, 0})),
            {host_rows, TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR)};
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({1, host_rows, host_columns}),
        tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(Layout::TILE), memory_config));
    std::vector<Tensor> shards;
    for (uint32_t mesh_row = 0; mesh_row < topology.R; ++mesh_row) {
        for (uint32_t mesh_column = 0; mesh_column < topology.C; ++mesh_column) {
            std::vector<float> host(static_cast<size_t>(host_rows) * host_columns, 0.0f);
            uint32_t region = 0;
            for (const auto& [half_first, half_end] : geometry.k_halves) {
                for (uint32_t column = 0; column < geometry.engine_columns; ++column) {
                    for (const auto& [k_first, k_end] : geometry.w_k_ranges) {
                        const uint32_t bank = region % banks, slot = region / banks;
                        uint32_t tile = 0;  // (tile in the region)
                        for (uint32_t k_tile = k_first; k_tile < k_end; ++k_tile) {
                            for (uint32_t column_tile = 0; column_tile < geometry.column_tiles[column];
                                 ++column_tile, ++tile) {
                                for (uint32_t tile_row = 0; tile_row < TILE_HEIGHT; ++tile_row) {
                                    const uint32_t latent_row = (half_first + k_tile) * TILE_HEIGHT + tile_row;
                                    const size_t row =
                                        static_cast<size_t>(slot * region_tiles + tile) * TILE_HEIGHT + tile_row;
                                    for (uint32_t tile_column = 0; tile_column < TILE_WIDTH; ++tile_column) {
                                        const uint32_t output_column =
                                            mesh_column * output_dim +
                                            (geometry.column_first_tile[column] + column_tile) * TILE_WIDTH +
                                            tile_column;
                                        host[row * host_columns + bank * TILE_WIDTH + tile_column] =
                                            latent_up_values
                                                [static_cast<size_t>(output_column) * latent_dim + latent_row] *
                                            norm_weight[latent_row];
                                    }
                                }
                            }
                        }
                        ++region;
                    }
                }
            }
            shards.push_back(Tensor::from_vector(std::move(host), spec));
        }
    }
    return ttnn::distributed::from_host_shards(to_mesh_order(std::move(shards), topology), mesh.shape())
        .to_device(&mesh, memory_config);
}
}  // namespace ttnn::operations::experimental::moe_block::moe_tail
