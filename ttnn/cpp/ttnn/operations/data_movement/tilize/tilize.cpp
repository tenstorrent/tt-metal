// SPDX-FileCopyrightText: © 2024-2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize.hpp"

#include "codegen/tilize_codegen_device_operation.hpp"
#include "codegen/tilize_codegen_supported.hpp"
#include "device/tilize_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

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

namespace {

// Populates the codegen prim's cache-key struct from the (already ND-squeezed) input tensor and
// the free function's resolved output placement/dtype. NC/Ht/Wt mirror ops/tilize/spec.py's
// _geometry(): NC folds every dim above the last two, Ht/Wt are ceil(H|W / TILE_H|W) (exact once
// supported_by_codegen has confirmed tile alignment).
ttnn::prim::TilizeCodegenParams make_codegen_params(
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& output_mem_config,
    DataType output_dtype,
    bool use_multicore,
    bool use_low_perf) {
    const auto& shape = input_tensor.logical_shape();
    const uint32_t rank = shape.rank();
    const uint32_t h = rank >= 2 ? shape[rank - 2] : 1;
    const uint32_t w = rank >= 1 ? shape[rank - 1] : 1;
    uint32_t nc = 1;
    for (uint32_t i = 0; i + 2 < rank; ++i) {
        nc *= shape[i];
    }

    ttnn::prim::TilizeCodegenParams params;
    params.NC = nc;
    params.Ht = (h + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
    params.Wt = (w + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
    params.input_dtype = input_tensor.dtype();
    params.output_dtype = output_dtype;
    params.input_mem_config = input_tensor.memory_config();
    params.output_mem_config = output_mem_config;
    params.use_multicore = use_multicore;
    params.use_low_perf = use_low_perf;
    params.preserve_logical_shape = false;
    return params;
}

}  // namespace

ttnn::Tensor tilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore,
    bool use_low_perf,
    tt::tt_metal::Tile tile,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::string& implementation) {
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tile.get_tile_size(input_cb_data_format);
    uint32_t output_single_tile_size =
        output_dtype.has_value()
            ? tile.get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(output_dtype.value()))
            : input_single_tile_size;
    uint32_t input_tile_width = tile.get_width();
    uint32_t input_tile_height = tile.get_height();

    uint32_t num_tiles_per_row = input_tensor.padded_shape()[-1] / input_tile_width;
    uint32_t num_tiles_per_col = input_tensor.padded_shape()[-2] / input_tile_height;

    // Fold in the block factory's c_1 staging CB so routing does not pick "fits" when only c_0+c_16 fit.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const uint32_t staging_bytes_per_tile = input_single_tile_size / input_tile_height;
    const uint32_t fixed_staging_bytes = 2 * dram_alignment;

    bool enough_space_width = ttnn::operations::data_movement::is_enough_space(
        input_tensor,
        input_single_tile_size,
        output_single_tile_size,
        num_tiles_per_col,
        staging_bytes_per_tile,
        fixed_staging_bytes);
    bool enough_space_height = ttnn::operations::data_movement::is_enough_space(
        input_tensor,
        input_single_tile_size,
        output_single_tile_size,
        num_tiles_per_row,
        staging_bytes_per_tile,
        fixed_staging_bytes);

    const auto selector = ttnn::prim::parse_implementation(implementation);

    auto base_tilize = [=](const ttnn::Tensor& input_tensor) {
        // Workaround for https://github.com/tenstorrent/tt-metal/issues/45331:
        // ttnn::prim::tilize routes wide width-sharded input to
        // TilizeMultiCoreDefaultProgramFactory, whose CBs are sized to a full
        // row of tiles (ntiles_per_block = ceil(logical_width / TILE_WIDTH))
        // and exceed L1. Reroute via interleaved DRAM so the prim selects
        // TilizeMultiCoreBlockProgramFactory, whose CBs are bounded by
        // max_l1 / (input_tile_size + output_tile_size) by construction.
        //
        // This reroute is sharded-input only, which the codegen prim never supports (see
        // tilize_codegen_supported.cpp), so under `auto`/`native` it stays on the native prim
        // unconditionally. An explicit `implementation="codegen"` request must instead reach the
        // selector dispatch below and hit supported_by_codegen()'s TT_FATAL rejection (R6) rather
        // than being silently downgraded to native here.
        if (selector != ttnn::prim::ImplementationSelector::Codegen && input_tensor.memory_config().is_sharded() &&
            !enough_space_height) {
            log_debug(tt::LogOp, "ttnn::tilize: rerouting wide sharded input via DRAM interleaved (#45331)");
            const auto target_memory_config = memory_config.value_or(input_tensor.memory_config());
            auto interleaved_input = ttnn::to_memory_config(input_tensor, ttnn::DRAM_MEMORY_CONFIG);
            auto interleaved_tile = ttnn::prim::tilize(
                interleaved_input,
                ttnn::DRAM_MEMORY_CONFIG,
                output_dtype,
                use_multicore,
                enough_space_width,
                /*enough_space_height=*/false,
                use_low_perf,
                tile,
                sub_core_grids);
            return ttnn::to_memory_config(interleaved_tile, target_memory_config);
        }

        auto call_native = [&]() {
            return ttnn::prim::tilize(
                input_tensor,
                memory_config,
                output_dtype,
                use_multicore,
                enough_space_width,
                enough_space_height,
                use_low_perf,
                tile,
                sub_core_grids);
        };

        if (selector == ttnn::prim::ImplementationSelector::Native) {
            return call_native();
        }

        // The codegen prim assumes the standard 32x32 tile and has no sub_core_grids parameter
        // (TilizeCodegenParams carries neither); either makes this call outside its scope
        // regardless of what supported_by_codegen() says about the tensor itself.
        const bool codegen_eligible_context = !sub_core_grids.has_value() &&
                                              tile.get_height() == tt::constants::TILE_HEIGHT &&
                                              tile.get_width() == tt::constants::TILE_WIDTH;

        const auto output_mem_config = memory_config.value_or(input_tensor.memory_config());
        const auto resolved_output_dtype = output_dtype.value_or(input_tensor.dtype());
        const auto codegen_params =
            make_codegen_params(input_tensor, output_mem_config, resolved_output_dtype, use_multicore, use_low_perf);
        const ttnn::prim::TilizeCodegenInputs codegen_inputs{input_tensor};

        if (selector == ttnn::prim::ImplementationSelector::Codegen) {
            TT_FATAL(
                codegen_eligible_context,
                "tilize: implementation=codegen does not support a custom tile shape or sub_core_grids");
            TT_FATAL(
                ttnn::prim::supported_by_codegen(codegen_params, codegen_inputs),
                "tilize: inputs not supported by the codegen implementation");
            return ttnn::prim::tilize_codegen(input_tensor, codegen_params);
        }

        // Auto: correctness gate && perf gate.
        if (codegen_eligible_context && ttnn::prim::supported_by_codegen(codegen_params, codegen_inputs) &&
            !ttnn::prim::is_demoted(codegen_params, codegen_inputs)) {
            return ttnn::prim::tilize_codegen(input_tensor, codegen_params);
        }
        return call_native();
    };

    return ttnn::operations::data_movement::build_ndiml_tilize(base_tilize, sub_core_grids)(input_tensor);
}

}  // namespace ttnn
