// SPDX-FileCopyrightText: © 2024-2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize.hpp"

#include "device/tilize_device_operation.hpp"
#include "codegen/tilize_codegen_device_operation.hpp"
#include "codegen/tilize_codegen_supported.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

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
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::string& implementation) {
    namespace tilize_codegen = ttnn::operations::data_movement::tilize_codegen;
    // Validate before any early return so an invalid value fails consistently.
    const auto sel = tilize_codegen::parse_implementation(implementation);

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

    bool enough_space_height = ttnn::operations::data_movement::is_enough_space(
        input_tensor,
        input_single_tile_size,
        output_single_tile_size,
        num_tiles_per_row,
        staging_bytes_per_tile,
        fixed_staging_bytes);

    auto base_tilize = [=](const ttnn::Tensor& input_tensor) {
        DataType resolved_output_dtype = output_dtype.value_or(input_tensor.dtype());
        MemoryConfig resolved_output_mem_config = memory_config.value_or(input_tensor.memory_config());

        // Every codegen builder places work over the full compute_with_storage_grid_size() and
        // implements neither a single-core mode nor a sub-core-grid restriction, so a caller that
        // set any of these controls to something codegen cannot honour must go to native.
        const bool controls_supported =
            tilize_codegen::supported_execution_controls(use_multicore, use_low_perf, sub_core_grids);

        if (sel != tilize_codegen::ImplementationSelector::Native) {
            const bool supported =
                controls_supported &&
                tilize_codegen::supported_by_codegen(input_tensor, resolved_output_mem_config, resolved_output_dtype);
            if (sel == tilize_codegen::ImplementationSelector::Codegen) {
                TT_FATAL(
                    supported,
                    "tilize: implementation=\"codegen\" requires a supported input/output configuration (row-major, "
                    "tile-aligned width, interleaved, in-scope dtype) and default execution controls "
                    "(use_multicore=true, use_low_perf=false, no sub_core_grids)");
                return ttnn::prim::tilize_codegen(
                    input_tensor, resolved_output_mem_config, resolved_output_dtype, tile);
            }
            // Auto: codegen iff supported and not perf-demoted; else fall through to native below.
            if (supported &&
                !tilize_codegen::is_demoted(input_tensor, resolved_output_mem_config, resolved_output_dtype)) {
                return ttnn::prim::tilize_codegen(
                    input_tensor, resolved_output_mem_config, resolved_output_dtype, tile);
            }
        }

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
