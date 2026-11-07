// SPDX-FileCopyrightText: © 2024-2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize.hpp"

#include "device/tilize_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {
using OwnedTilizeArgs = std::tuple<ttnn::Tensor>;
using BaseTilizeType = std::function<ttnn::Tensor(const ttnn::Tensor&)>;

using MassagedTilize = MassagedOperation<ttnn::Tensor, const ttnn::Tensor&>;
using MassagedTilizeParams = MassagedOperationParams<ttnn::Tensor, const ttnn::Tensor&>;

MassagedTilize build_ndiml_tilize(BaseTilizeType base_tilize, const std::optional<CoreRangeSet>& sub_core_grids) {
    auto original_shape = std::make_shared<Shape>();
    return MassagedTilize(MassagedTilizeParams{
        .predicate = [](const ttnn::Tensor& input_tensor) -> bool { return input_tensor.logical_shape().rank() > 4; },
        .pre_transform = [=](const ttnn::Tensor& input_tensor) -> OwnedTilizeArgs {
            *original_shape = input_tensor.logical_shape();
            ttnn::Tensor squeezed_tensor = squeeze_from_ND_to_4D(input_tensor, sub_core_grids);
            return std::make_tuple(squeezed_tensor);
        },
        .post_transform = [=](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto unsqueezed_tensor = ttnn::reshape(
                output, *original_shape, std::nullopt, std::nullopt, TileReshapeMapMode::CACHE, sub_core_grids);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_tilize)});
}

}  // namespace ttnn::operations::data_movement

namespace ttnn {

ttnn::Tensor tilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore,
    bool use_low_perf,
    tt::tt_metal::Tile tile,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // A zero-volume input has nothing to tilize. Legacy tolerated the resulting empty program, but the
    // Metal 2.0 spec validator rejects it (no work cores -> the output DFB has no producer), so
    // short-circuit to a correctly-shaped empty output the way the sibling tilize_with_val_padding does.
    if (input_tensor.physical_volume() == 0) {
        TensorSpec spec(
            input_tensor.logical_shape(),
            TensorLayout(
                output_dtype.value_or(input_tensor.dtype()),
                PageConfig(Layout::TILE, tile),
                memory_config.value_or(input_tensor.memory_config())));
        return create_device_tensor(spec, input_tensor.device());
    }

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tile.get_tile_size(input_cb_data_format);
    uint32_t output_single_tile_size =
        output_dtype.has_value()
            ? tile.get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(output_dtype.value()))
            : input_single_tile_size;
    uint32_t input_tile_width = tile.get_width();
    uint32_t input_tile_height = tile.get_height();

    uint32_t num_tiles_per_row = input_tensor.padded_shape()[-1] / input_tile_width;

    // Fold in the block factory's c_1 staging CB so routing does not pick "fits" when only c_0+c_16 fit.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const uint32_t staging_bytes_per_tile = input_single_tile_size / input_tile_height;
    const uint32_t fixed_staging_bytes = 2 * dram_alignment;

    // Reserve the output buffer's per-core L1 up front: it is allocated after this check
    // but before the CBs are placed, so leaving it out overestimates the CB budget and can
    // pick a factory whose static CBs then clash with it (issue #21358).
    const uint32_t pending_l1_output_bytes = ttnn::operations::data_movement::get_pending_l1_output_reservation(
        input_tensor,
        input_tensor.padded_shape(),
        memory_config.value_or(input_tensor.memory_config()),
        output_dtype.value_or(input_tensor.dtype()),
        Layout::TILE);

    bool enough_space_height = ttnn::operations::data_movement::is_enough_space(
        input_tensor,
        input_single_tile_size,
        output_single_tile_size,
        num_tiles_per_row,
        staging_bytes_per_tile,
        fixed_staging_bytes,
        pending_l1_output_bytes);

    auto block_split_cores = [&](uint32_t rows, uint32_t limit, size_t grid_area) -> uint32_t {
        if (limit == 0) {
            return 0;
        }
        const uint32_t height_tiles = rows / input_tile_height;
        const uint32_t nblocks = num_tiles_per_row * height_tiles;
        const auto split = ttnn::compute_ncores_wh(grid_area, nblocks, num_tiles_per_row, height_tiles);
        if (split.single_block_size <= limit) {
            return split.ncores;
        }
        const auto limited = ttnn::try_compute_ncores_wh_sb(grid_area, nblocks, num_tiles_per_row, height_tiles, limit);
        return limited.has_value() ? limited->ncores : 0;
    };

    auto base_tilize = [=](const ttnn::Tensor& input_tensor) {
        // Workaround for https://github.com/tenstorrent/tt-metal/issues/45331:
        // ttnn::prim::tilize routes wide width-sharded input to
        // TilizeMultiCoreDefaultProgramFactory, whose CBs are sized to a full
        // row of tiles (ntiles_per_block = ceil(logical_width / TILE_WIDTH))
        // and exceed L1. Reroute via interleaved DRAM so the prim selects
        // TilizeMultiCoreBlockProgramFactory, whose CBs are bounded by
        // max_l1 / (input_tile_size + output_tile_size) by construction.
        if (input_tensor.memory_config().is_sharded() && !enough_space_height) {
            log_debug(tt::LogOp, "ttnn::tilize: rerouting wide sharded input via DRAM interleaved (#45331)");
            const auto target_memory_config = memory_config.value_or(input_tensor.memory_config());
            auto interleaved_input = ttnn::to_memory_config(input_tensor, ttnn::DRAM_MEMORY_CONFIG);
            auto interleaved_tile = ttnn::prim::tilize(
                interleaved_input,
                ttnn::DRAM_MEMORY_CONFIG,
                output_dtype,
                use_multicore,
                /*enough_space_height=*/false,
                use_low_perf,
                tile,
                sub_core_grids);
            return ttnn::to_memory_config(interleaved_tile, target_memory_config);
        }
        const auto& shape = input_tensor.logical_shape();
        if (!enough_space_height && use_multicore && !use_low_perf && shape.rank() >= 3 &&
            input_tensor.device()->arch() == tt::ARCH::WORMHOLE_B0 && input_tensor.layout() == Layout::ROW_MAJOR &&
            input_tensor.dtype() == DataType::BFLOAT16 && input_tile_height == tt::constants::TILE_HEIGHT &&
            input_tile_width == tt::constants::TILE_WIDTH &&
            input_tensor.memory_config().buffer_type() == BufferType::L1 &&
            !memory_config.value_or(input_tensor.memory_config()).is_sharded() &&
            shape == input_tensor.padded_shape() && shape[-2] % input_tile_height == 0 &&
            shape[-1] % input_tile_width == 0 && shape.volume() / (static_cast<uint64_t>(shape[-2]) * shape[-1]) > 1) {
            const uint32_t max_l1 = ttnn::operations::data_movement::get_max_l1_space(input_tensor);
            const uint32_t budget = max_l1 > pending_l1_output_bytes + fixed_staging_bytes
                                        ? max_l1 - pending_l1_output_bytes - fixed_staging_bytes
                                        : 0;
            const uint32_t limit = budget / (input_single_tile_size + output_single_tile_size + staging_bytes_per_tile);
            const auto grid_size = input_tensor.device()->compute_with_storage_grid_size();
            const size_t grid_area =
                sub_core_grids.has_value() ? sub_core_grids->num_cores() : grid_size.x * grid_size.y;
            const uint32_t rows = shape.volume() / shape[-1];
            if (block_split_cores(rows, limit, grid_area) > block_split_cores(shape[-2], limit, grid_area)) {
                const ttnn::Shape folded_shape({1, 1, rows, shape[-1]});
                auto folded = ttnn::reshape(input_tensor, folded_shape, folded_shape);
                auto output = ttnn::prim::tilize(
                    folded,
                    memory_config,
                    output_dtype,
                    use_multicore,
                    /*enough_space_height=*/false,
                    use_low_perf,
                    tile,
                    sub_core_grids);
                return ttnn::reshape(output, shape, shape);
            }
        }
        return ttnn::prim::tilize(
            input_tensor,
            memory_config,
            output_dtype,
            use_multicore,
            enough_space_height,
            use_low_perf,
            tile,
            sub_core_grids);
    };

    return ttnn::operations::data_movement::build_ndiml_tilize(base_tilize, sub_core_grids)(input_tensor);
}

}  // namespace ttnn
