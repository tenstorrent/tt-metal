// SPDX-FileCopyrightText: © 2024-2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize.hpp"

#include "tilize_force.hpp"
#include "codegen/tilize_codegen_device_operation.hpp"
#include "codegen/tilize_codegen_supported.hpp"
#include "device/tilize_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

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

namespace {

// A zero-volume input has nothing to tilize. Legacy tolerated the resulting empty program, but the
// Metal 2.0 spec validator rejects it (no work cores -> the output DFB has no producer), so
// short-circuit to a correctly-shaped empty output the way the sibling tilize_with_val_padding does.
ttnn::Tensor tilize_empty_output(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    const tt::tt_metal::Tile& tile) {
    TensorSpec spec(
        input_tensor.logical_shape(),
        TensorLayout(
            output_dtype.value_or(input_tensor.dtype()),
            PageConfig(Layout::TILE, tile),
            memory_config.value_or(input_tensor.memory_config())));
    return create_device_tensor(spec, input_tensor.device());
}

// The existing native implementation, unconditionally. Calls the native prim directly rather than
// re-entering ttnn::tilize, so a call already routed to native cannot be routed a second time.
ttnn::Tensor tilize_native(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore,
    bool use_low_perf,
    tt::tt_metal::Tile tile,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (input_tensor.physical_volume() == 0) {
        return tilize_empty_output(input_tensor, memory_config, output_dtype, tile);
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

    return build_ndiml_tilize(base_tilize, sub_core_grids)(input_tensor);
}

}  // namespace

namespace detail {

ttnn::Tensor tilize_force_native(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore,
    bool use_low_perf,
    tt::tt_metal::Tile tile,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return tilize_native(input_tensor, memory_config, output_dtype, use_multicore, use_low_perf, tile, sub_core_grids);
}

ttnn::Tensor tilize_force_codegen(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype) {
    using tilize_codegen::plan_codegen;
    using tilize_codegen::supported_by_codegen;

    // Normalize exactly as the routed entry does, so a rank>4 input reaches the planner and the
    // prim as the squeezed 4D tensor rather than the raw one.
    auto dispatch = [=](const ttnn::Tensor& normalized_input) -> ttnn::Tensor {
        tilize_codegen::require_allocated_device_input(normalized_input);
        TT_FATAL(
            codegen_supported_on_arch(normalized_input),
            "tilize_force_codegen: the codegen implementation runs only on Wormhole and Blackhole. This entry never "
            "falls back to native.");
        const auto output_mem_config = memory_config.value_or(normalized_input.memory_config());
        const DataType dtype = output_dtype.value_or(normalized_input.dtype());
        TT_FATAL(
            supported_by_codegen(normalized_input, output_mem_config, dtype),
            "tilize_force_codegen invoked for a case the codegen implementation does not support. This entry never "
            "falls back to native. Use ttnn::tilize if you want the case routed.");
        const auto plan = plan_codegen(normalized_input, output_mem_config, dtype);
        TT_FATAL(
            plan.has_value(),
            "tilize_force_codegen: the codegen CB plan does not fit the L1 free now. This entry never falls back to "
            "native. Use ttnn::tilize if you want the case routed.");
        return ttnn::prim::tilize_codegen(normalized_input, output_mem_config, dtype, *plan);
    };

    return build_ndiml_tilize(dispatch, /*sub_core_grids=*/std::nullopt)(input_tensor);
}

}  // namespace detail

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
    using ttnn::operations::data_movement::tilize_native;
    using ttnn::operations::data_movement::tilize_codegen::is_demoted;
    using ttnn::operations::data_movement::tilize_codegen::plan_codegen;
    using ttnn::operations::data_movement::tilize_codegen::supported_by_codegen;
    using ttnn::operations::data_movement::tilize_codegen::supported_execution_controls;

    if (input_tensor.physical_volume() == 0) {
        return ttnn::operations::data_movement::tilize_empty_output(input_tensor, memory_config, output_dtype, tile);
    }

    const bool controls_ok = supported_execution_controls(use_multicore, use_low_perf, tile, sub_core_grids);

    // Route on the normalized (squeezed to 4D) tensor, the one native would see. The plan is sized against
    // the L1 free now, so nullopt (out of scope or no room for the CBs) falls back to native.
    // The two implementations convert BFLOAT16 to INT32 differently, so a dtype conversion stays on native to keep
    // ttnn::tilize results independent of the route.
    auto dispatch = [=](const ttnn::Tensor& normalized_input) -> ttnn::Tensor {
        const auto output_mem_config = memory_config.value_or(normalized_input.memory_config());
        const DataType dtype = output_dtype.value_or(normalized_input.dtype());
        if (dtype == normalized_input.dtype() &&
            ttnn::operations::data_movement::codegen_supported_on_arch(normalized_input) && controls_ok &&
            supported_by_codegen(normalized_input, output_mem_config, dtype) && !is_demoted(normalized_input, dtype)) {
            if (const auto plan = plan_codegen(normalized_input, output_mem_config, dtype); plan.has_value()) {
                return ttnn::prim::tilize_codegen(normalized_input, output_mem_config, dtype, *plan);
            }
        }
        return tilize_native(
            normalized_input, memory_config, output_dtype, use_multicore, use_low_perf, tile, sub_core_grids);
    };

    return ttnn::operations::data_movement::build_ndiml_tilize(dispatch, sub_core_grids)(input_tensor);
}

}  // namespace ttnn
