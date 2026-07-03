// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen.hpp"

#include <tt-metalium/constants.hpp>

#include "device/untilize_codegen_device_operation.hpp"

namespace ttnn {

ttnn::Tensor untilize_codegen(const ttnn::Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    const auto& shape = input_tensor.logical_shape();

    TT_FATAL(shape.rank() == 4, "untilize_codegen PoC requires a 4D tensor");
    TT_FATAL(input_tensor.layout() == ttnn::TILE_LAYOUT, "untilize_codegen PoC requires TILE layout");
    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "untilize_codegen PoC requires bfloat16");
    TT_FATAL(!input_tensor.is_sharded(), "untilize_codegen PoC requires interleaved input");
    TT_FATAL(
        shape[2] % tt::constants::TILE_HEIGHT == 0 && shape[3] % tt::constants::TILE_WIDTH == 0,
        "untilize_codegen PoC requires H and W tile-aligned");

    const uint32_t Wt = shape[3] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = shape[2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = shape[0] * shape[1];
    const uint32_t total_tile_rows = NC * Ht;

    // Column-parallel path (single tile-row, many columns) uses a different writer
    // kernel + program-factory variant — out of scope for this spike.
    TT_FATAL(
        !(total_tile_rows == 1 && Wt > 1),
        "untilize_codegen PoC does not support the column-parallel path (total_tile_rows==1 && Wt>1)");

    const MemoryConfig out_mc = memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::untilize_codegen(input_tensor, out_mc);
}

}  // namespace ttnn
