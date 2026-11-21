// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The head's weights in their device layouts (prepare_head_weights): the shared expert's per-bank W stream, the
// router's weight, bias and expert index tensors.
#include "head_chip_program.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

HeadDeviceWeights prepare_head_weights(
    const HeadModel& model,
    const std::vector<float>& gate,
    const std::vector<float>& up,
    const std::vector<float>& down,
    const std::vector<float>& latent_down,
    const std::vector<float>& router_w,
    const std::vector<float>& router_b) {
    using tt::tt_metal::BufferType;
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;
    using tt::tt_metal::TensorMemoryLayout;
    const auto& sizing = *model.sizing;
    const auto& shape = model.config.shape;
    const uint32_t hidden_dim = shape.hidden_dim, intermediate = shape.shared_expert_dim * model.tp_size,
                   intermediate_per_chip = shape.shared_expert_dim, tp_size = model.tp_size;
    TT_FATAL(
        gate.size() == size_t(intermediate) * hidden_dim && up.size() == gate.size(),
        "moe_head: gate / up must be [Is {}, H {}]",
        intermediate,
        hidden_dim);
    TT_FATAL(
        down.size() == size_t(hidden_dim) * intermediate,
        "moe_head: down must be [H {}, Is {}]",
        hidden_dim,
        intermediate);
    const uint32_t latent = shape.latent.value_or(0);
    TT_FATAL(
        latent_down.size() == size_t(latent) * hidden_dim,
        "moe_head: latent_down must be [Hr {}, H {}]",
        latent,
        hidden_dim);
    const uint32_t router_candidates = shape.router_candidates;
    TT_FATAL(
        router_w.size() == size_t(router_candidates) * hidden_dim && router_b.size() == router_candidates,
        "moe_head: the router must be [E {}, H {}]",
        router_candidates,
        hidden_dim);
    auto& mesh = *model.mesh;
    const uint32_t banks = u32(mesh.dram_grid_size().x);
    const uint32_t num_regions = sizing.engine_columns * sizing.engine_rows,
                   regions_per_bank = (num_regions + banks - 1) / banks;
    const uint32_t host_rows = regions_per_bank * u32(sizing.w_own_tiles) * 32, host_columns = banks * 32;
    const tt::tt_metal::MemoryConfig memory_config{
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::DRAM,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange({0, 0}, {banks - 1, 0})),
            {host_rows, 32},
            tt::tt_metal::ShardOrientation::ROW_MAJOR)};
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({1, host_rows, host_columns}),
        tt::tt_metal::TensorLayout(model.config.w_dtype, tt::tt_metal::PageConfig(Layout::TILE), memory_config));
    std::vector<Tensor> shards;
    for (uint32_t chip_index = 0; chip_index < model.num_chips; ++chip_index) {
        const uint32_t mesh_row = chip_index / model.mesh_columns, mesh_column = chip_index % model.mesh_columns;
        const uint32_t ring_position = model.chips.at({mesh_row, mesh_column}).gather_ring.position;
        // (the chip's gate / up as glu_weight's [H, 2 Ic]: tile column 2 i gate i, 2 i + 1 up i; down [Ic, H]; latent
        // [H, latent / TP] - its position's rows of the latent down, transposed)
        auto gate_up_value = [&](uint32_t k_row, uint32_t w_column) {
            const uint32_t pair_index = w_column / 64, odd = (w_column / 32) % 2, lane = w_column % 32;
            const uint32_t row = mesh_column * intermediate_per_chip + pair_index * 32 + lane;
            return (odd ? up : gate)[size_t(row) * hidden_dim + k_row];
        };
        auto down_value = [&](uint32_t k_row, uint32_t w_column) {
            return down[size_t(w_column) * intermediate + mesh_column * intermediate_per_chip + k_row];
        };
        auto latent_value = [&](uint32_t k_row, uint32_t w_column) {
            return latent_down[size_t(ring_position * latent / tp_size + w_column) * hidden_dim + k_row];
        };
        std::vector<float> host(size_t(host_rows) * host_columns, 0.0f);
        for (uint32_t core_x = 0; core_x < sizing.engine_columns; ++core_x) {
            // the column's blocks in stream order: a block = [K tiles][width tiles] of 32 x 32
            std::vector<std::function<float(uint32_t, uint32_t)>> sources;
            std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>
                blocks;  // (source, first K row tile, K tiles, first column) + width
            std::vector<uint32_t> widths;
            const uint32_t gate_up_first_column = 2 * sizing.column_first_pair[core_x];
            for (uint32_t gate_up_pass = 0; gate_up_pass < sizing.gate_up_pass_widths[core_x].size(); ++gate_up_pass) {
                const uint32_t pass_width = sizing.gate_up_pass_widths[core_x][gate_up_pass];
                uint32_t first_column = gate_up_first_column;
                for (uint32_t i = 0; i < gate_up_pass; ++i) {
                    first_column += sizing.gate_up_pass_widths[core_x][i];
                }
                uint32_t source = 0;
                if (gate_up_pass < sizing.gate_up_passes_without_latent &&
                    first_column + pass_width >
                        2 * (sizing.column_first_pair[core_x] +
                             sizing.column_pairs[core_x])) {  // (a dummy pass: the first pair again)
                    first_column = gate_up_first_column;
                }
                if (gate_up_pass >= sizing.gate_up_passes_without_latent) {  // (latent passes: this column's slice of
                                                                             // the chip's latent columns)
                    source = 2;
                    first_column = sizing.latent_column_first_tile[core_x];
                    for (uint32_t i = sizing.gate_up_passes_without_latent; i < gate_up_pass; ++i) {
                        first_column += sizing.gate_up_pass_widths[core_x][i];
                    }
                }
                for (uint32_t j = 0; j < sizing.num_k_steps; ++j) {
                    blocks.emplace_back(source, j * sizing.k_step_tiles, sizing.k_step_tiles, first_column);
                    widths.push_back(pass_width);
                }
            }
            for (uint32_t down_pass = 0; down_pass < sizing.down_pass_widths[core_x].size(); ++down_pass) {
                const uint32_t pass_width = sizing.down_pass_widths[core_x][down_pass];
                const uint32_t first_column =
                    model.reduce_scatter->column_passes(ring_position, core_x)[down_pass].first_column;
                for (uint32_t i = 0; i < sizing.down_k_steps; ++i) {
                    blocks.emplace_back(1, i * sizing.down_k_step_tiles, sizing.down_k_step_tiles, first_column);
                    widths.push_back(pass_width);
                }
            }
            for (uint32_t core_y = 0; core_y < sizing.engine_rows; ++core_y) {
                const uint32_t region = core_x * sizing.engine_rows + core_y, bank = region % banks,
                               slot = region / banks;
                uint32_t tile = 0;  // tile in the region
                for (size_t block_index = core_y; block_index < blocks.size(); block_index += sizing.engine_rows) {
                    const auto [block_source, first_k_tile, k_tiles, block_first_column] = blocks[block_index];
                    for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                        for (uint32_t w_tile = 0; w_tile < widths[block_index]; ++w_tile, ++tile) {
                            for (uint32_t tile_row = 0; tile_row < 32; ++tile_row) {
                                const size_t row = size_t(slot * sizing.w_own_tiles + tile) * 32 + tile_row;
                                for (uint32_t tile_column = 0; tile_column < 32; ++tile_column) {
                                    const uint32_t k_row = (first_k_tile + k_tile) * 32 + tile_row,
                                                   w_column = (block_first_column + w_tile) * 32 + tile_column;
                                    host[row * host_columns + bank * 32 + tile_column] =
                                        block_source == 0 ? gate_up_value(k_row, w_column)
                                                          : (block_source == 1 ? down_value(k_row, w_column)
                                                                               : latent_value(k_row, w_column));
                                }
                            }
                        }
                    }
                }
            }
        }
        shards.push_back(Tensor::from_vector(std::move(host), spec));
    }
    HeadDeviceWeights out;
    out.w = ttnn::distributed::from_host_shards(to_mesh_order(std::move(shards), model.topology()), mesh.shape())
                .to_device(&mesh, memory_config);
    // the router: W_r [E, H / TP] (the chip's H slice) bf16, the bias and the expert index as [E, 32] tiles
    const uint32_t hidden_per_chip = hidden_dim / tp_size;
    const tt::tt_metal::TensorSpec router_w_spec(
        ttnn::Shape({router_candidates, hidden_per_chip}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    std::vector<Tensor> router_shards;
    for (uint32_t chip_index = 0; chip_index < model.num_chips; ++chip_index) {
        const uint32_t mesh_column = chip_index % model.mesh_columns;
        std::vector<float> router_values(size_t(router_candidates) * hidden_per_chip);
        for (uint32_t expert = 0; expert < router_candidates; ++expert) {
            std::copy_n(
                router_w.begin() + size_t(expert) * hidden_dim + mesh_column * hidden_per_chip,
                hidden_per_chip,
                router_values.begin() + size_t(expert) * hidden_per_chip);
        }
        router_shards.push_back(Tensor::from_vector(std::move(router_values), router_w_spec));
    }
    out.router_w =
        ttnn::distributed::from_host_shards(to_mesh_order(std::move(router_shards), model.topology()), mesh.shape())
            .to_device(&mesh, ttnn::DRAM_MEMORY_CONFIG);
    std::vector<float> bias(size_t(router_candidates) * 32);
    std::vector<uint16_t> expert_index(size_t(router_candidates) * 32);
    for (uint32_t expert = 0; expert < router_candidates; ++expert) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
            bias[size_t(expert) * 32 + lane] = router_b[expert];
            expert_index[size_t(expert) * 32 + lane] = static_cast<uint16_t>(expert);
        }
    }
    const tt::tt_metal::TensorSpec bias_spec(
        ttnn::Shape({router_candidates, 32}),
        tt::tt_metal::TensorLayout(
            DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    const tt::tt_metal::TensorSpec index_spec(
        ttnn::Shape({router_candidates, 32}),
        tt::tt_metal::TensorLayout(DataType::UINT16, tt::tt_metal::PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    std::vector<Tensor> bias_shards(model.num_chips, Tensor::from_vector(bias, bias_spec)),
        index_shards(model.num_chips, Tensor::from_vector(expert_index, index_spec));
    out.router_bias =
        ttnn::distributed::from_host_shards(bias_shards, mesh.shape()).to_device(&mesh, ttnn::DRAM_MEMORY_CONFIG);
    out.router_index =
        ttnn::distributed::from_host_shards(index_shards, mesh.shape()).to_device(&mesh, ttnn::DRAM_MEMORY_CONFIG);
    return out;
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head
