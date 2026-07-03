// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_codegen.hpp"

#include <numeric>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "device/repeat_codegen_device_operation.hpp"

namespace ttnn {

ttnn::Tensor repeat_codegen(
    const ttnn::Tensor& input_tensor,
    uint32_t dim,
    uint32_t repetitions,
    const std::optional<MemoryConfig>& memory_config) {
    const auto& shape = input_tensor.logical_shape();

    TT_FATAL(shape.rank() == 4, "repeat_codegen PoC requires a 4D tensor");
    TT_FATAL(input_tensor.layout() == ttnn::TILE_LAYOUT, "repeat_codegen PoC requires TILE layout");
    TT_FATAL(input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16, "repeat_codegen PoC requires bfloat16");
    TT_FATAL(dim < 3, "repeat_codegen PoC supports higher-dim repeat only (dim 0, 1, or 2)");
    TT_FATAL(
        shape[2] % tt::constants::TILE_HEIGHT == 0 && shape[3] % tt::constants::TILE_WIDTH == 0,
        "repeat_codegen PoC requires H and W tile-aligned");

    const uint32_t Ht = shape[2] / tt::constants::TILE_HEIGHT;
    const uint32_t Wt = shape[3] / tt::constants::TILE_WIDTH;
    const uint32_t dim_pages[4] = {shape[0], shape[1], Ht, Wt};

    // higher/rep_dim_pages/lower in tile-page space — matches the codegen sequencer.
    uint32_t higher = 1;
    for (uint32_t d = 0; d < dim; ++d) {
        higher *= dim_pages[d];
    }
    uint32_t lower = 1;
    for (uint32_t d = dim + 1; d < 4; ++d) {
        lower *= dim_pages[d];
    }
    const uint32_t rep_dim_pages = dim_pages[dim];

    const tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t tile_page_size = tt::tile_size(df);

    const MemoryConfig out_mc = memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::repeat_codegen(
        input_tensor, repetitions, static_cast<int32_t>(dim), out_mc, higher, rep_dim_pages, lower, tile_page_size);
}

}  // namespace ttnn
