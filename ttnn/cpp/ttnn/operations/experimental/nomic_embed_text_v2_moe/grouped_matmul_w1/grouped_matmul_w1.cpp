// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_w1.hpp"

#include <cstddef>
#include <cstring>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tile.hpp>
#include "device/grouped_matmul_w1_device_operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

Tensor grouped_matmul_w1(
    const Tensor& input_tensor,
    const Tensor& weight,
    const std::optional<Tensor>& expert_counts,
    tt::tt_metal::DataType dtype,
    bool transpose_weight,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    tt::tt_metal::Layout output_layout,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    bool per_expert_lanes,
    const std::optional<Tensor>& bias,
    uint32_t weight_chunk_tiles,
    uint32_t weight_n,
    bool weight_has_bias) {
    // Without a config: HiFi4 and fp32 accumulation, the most precise setting.
    const auto kernel_config = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);
    return ttnn::prim::grouped_matmul_w1(
        input_tensor,
        weight,
        expert_counts,
        dtype,
        transpose_weight,
        activation,
        output_layout,
        // A sharded input's layout does not fit the output's N; its buffer type does.
        memory_config.value_or(tt::tt_metal::MemoryConfig(
            tt::tt_metal::TensorMemoryLayout::INTERLEAVED, input_tensor.memory_config().buffer_type())),
        kernel_config,
        per_expert_lanes,
        bias,
        weight_chunk_tiles,
        weight_n,
        weight_has_bias);
}

Tensor prepare_grouped_matmul_w1_weight(
    const Tensor& weight,
    uint32_t num_experts,
    bool transpose_weight,
    const std::optional<Tensor>& bias,
    uint32_t chunk_tiles,
    ttnn::MeshDevice* device) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;
    constexpr uint32_t kTile = tt::constants::TILE_HEIGHT;

    TT_FATAL(device != nullptr, "prepare_grouped_matmul_w1_weight: device must be given");
    TT_FATAL(chunk_tiles > 0, "prepare_grouped_matmul_w1_weight: chunk_tiles must be positive");
    TT_FATAL(num_experts > 0, "prepare_grouped_matmul_w1_weight: num_experts must be positive");
    TT_FATAL(
        weight.layout() == Layout::TILE &&
            (weight.dtype() == DataType::BFLOAT16 || weight.dtype() == DataType::BFLOAT8_B),
        "prepare_grouped_matmul_w1_weight: weight must be TILE, BFLOAT16 or BFLOAT8_B");
    const auto& shape = weight.logical_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[0] == 1 && shape[1] == 1 && shape[2] % kTile == 0 &&
            shape[3] % (num_experts * kTile) == 0,
        "prepare_grouped_matmul_w1_weight: weight must be 1 x 1 x rows x (num_experts x columns) in whole tiles, got "
        "{}",
        shape);
    const uint32_t row_tiles = shape[3] / kTile;
    const uint32_t expert_column_tiles = row_tiles / num_experts;
    const uint32_t k_tiles = transpose_weight ? expert_column_tiles : shape[2] / kTile;
    const uint32_t n_tiles = transpose_weight ? shape[2] / kTile : expert_column_tiles;
    const bool has_bias = bias.has_value();
    if (has_bias) {
        TT_FATAL(!transpose_weight, "prepare_grouped_matmul_w1_weight: bias needs the weight untransposed");
        TT_FATAL(
            bias->layout() == Layout::TILE && bias->dtype() == weight.dtype() &&
                bias->logical_shape()[-1] == shape[3] && bias->logical_shape()[-2] == 1,
            "prepare_grouped_matmul_w1_weight: bias must be 1 x 1 x 1 x {} TILE of the weight's dtype",
            shape[3]);
    }

    const auto grid = device->compute_with_storage_grid_size();
    const auto chunks = grouped_weight_chunks(k_tiles, n_tiles, num_experts, has_bias, chunk_tiles, grid.x);
    const tt::tt_metal::Tile tile({kTile, kTile});
    const uint32_t tile_bytes = tile.get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(weight.dtype()));

    // A host TILE tensor holds whole tiles in row-major tile order, each laid out as on the device.
    const Tensor host_weight = weight.cpu();
    const auto weight_buffer = tt::tt_metal::host_buffer::get_host_buffer(host_weight);
    const auto weight_bytes = weight_buffer.view_bytes();
    std::optional<tt::tt_metal::HostBuffer> bias_buffer;
    if (has_bias) {
        bias_buffer = tt::tt_metal::host_buffer::get_host_buffer(bias->cpu());
    }

    const size_t total_bytes = static_cast<size_t>(kGroupedWeightBanks) * chunks.shard_tiles * tile_bytes;
    std::vector<std::byte> chunked(total_bytes);
    const uint32_t group_tiles = (k_tiles + (has_bias ? 1 : 0)) * chunks.subblock_w;
    for (uint32_t share = 0; share < chunks.num_shares; ++share) {
        const uint32_t first_n = share * chunks.share_n_tiles;
        const uint32_t width = share + 1 == chunks.num_shares ? chunks.last_share_n_tiles : chunks.share_n_tiles;
        const uint32_t slice_tiles = (k_tiles + (has_bias ? 1 : 0)) * width;
        for (uint32_t expert = 0; expert < num_experts; ++expert) {
            for (uint32_t position = 0; position < slice_tiles; ++position) {
                // Slot order: subblock position / group_tiles, then its row (k, or the bias row), then its column.
                const uint32_t row = position % group_tiles / chunks.subblock_w;
                const uint32_t n = first_n + position / group_tiles * chunks.subblock_w + position % chunks.subblock_w;
                const std::byte* source = nullptr;
                if (row < k_tiles) {
                    const uint32_t tile_row = transpose_weight ? n : row;
                    const uint32_t tile_column = transpose_weight ? expert * k_tiles + row : expert * n_tiles + n;
                    source =
                        weight_bytes.data() + (static_cast<size_t>(tile_row) * row_tiles + tile_column) * tile_bytes;
                } else {
                    source = bias_buffer->view_bytes().data() + static_cast<size_t>(expert * n_tiles + n) * tile_bytes;
                }
                const uint32_t chunk = position / chunk_tiles;
                const uint32_t bank = (share + chunk + expert) % kGroupedWeightBanks;
                const size_t target =
                    static_cast<size_t>(bank) * chunks.shard_tiles +
                    static_cast<size_t>(
                        (expert * chunks.max_chunks + chunk) * chunks.bank_sharers + share / kGroupedWeightBanks) *
                        chunk_tiles +
                    position % chunk_tiles;
                std::memcpy(chunked.data() + target * tile_bytes, source, tile_bytes);
            }
        }
    }

    const ttnn::Shape chunked_shape({1, 1, kTile, kGroupedWeightBanks * chunks.shard_tiles * kTile});
    const tt::tt_metal::TensorSpec spec(
        chunked_shape,
        tt::tt_metal::TensorLayout(
            weight.dtype(), tt::tt_metal::PageConfig(Layout::TILE), tt::tt_metal::MemoryConfig{}));
    // The host buffer's element type is the dtype's: bfloat16 values, or bfloat8_b's packed words.
    tt::tt_metal::HostBuffer buffer = [&] {
        if (weight.dtype() == DataType::BFLOAT16) {
            std::vector<bfloat16> values(total_bytes / sizeof(bfloat16));
            std::memcpy(values.data(), chunked.data(), total_bytes);
            return tt::tt_metal::HostBuffer(std::move(values));
        }
        std::vector<uint32_t> words(total_bytes / sizeof(uint32_t));
        std::memcpy(words.data(), chunked.data(), total_bytes);
        return tt::tt_metal::HostBuffer(std::move(words));
    }();

    const auto dram = device->dram_grid_size();
    TT_FATAL(
        dram.x * dram.y == kGroupedWeightBanks,
        "prepare_grouped_matmul_w1_weight: the chunk layout assumes {} DRAM banks, the device has {}",
        kGroupedWeightBanks,
        dram.x * dram.y);
    const tt::tt_metal::ShardSpec shard_spec(
        CoreRangeSet{CoreRange(CoreCoord(0, 0), CoreCoord(dram.x - 1, dram.y - 1))},
        {kTile, chunks.shard_tiles * kTile},
        tt::tt_metal::ShardOrientation::ROW_MAJOR);
    const tt::tt_metal::MemoryConfig memory_config(
        tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, tt::tt_metal::BufferType::DRAM, shard_spec);
    return Tensor(std::move(buffer), spec).to_device(device, memory_config);
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
