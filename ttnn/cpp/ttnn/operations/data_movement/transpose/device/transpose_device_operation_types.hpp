// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Effective compute grid for a transpose program factory.
//
// Factories normally size their grid from device->compute_with_storage_grid_size(), which
// claims every Tensix core. Under an active SubDevice that fails with "Kernel group cores
// do not match sub device cores", because the kernel group must be a subset of the
// sub-device's cores. When sub_core_grids is supplied, return its bounding box as a grid
// size instead, so every downstream use (split_work_to_cores, the total_cores rectangle
// used for CB and kernel creation) is confined without further changes.
//
// The set must be a single rectangle anchored at (0,0): split_work_to_cores' CoreCoord
// overload and the total_cores rectangle both assume an origin-anchored grid, so a
// non-anchored set would silently address the wrong cores rather than error.
inline tt::tt_metal::CoreCoord transpose_effective_grid(
    const tt::tt_metal::CoreCoord& device_grid, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (!sub_core_grids.has_value()) {
        return device_grid;
    }
    const auto bbox = sub_core_grids->bounding_box();
    TT_FATAL(
        bbox.start_coord.x == 0 && bbox.start_coord.y == 0,
        "transpose sub_core_grids must be anchored at (0,0), got start ({},{})",
        bbox.start_coord.x,
        bbox.start_coord.y);
    TT_FATAL(
        bbox.end_coord.x < device_grid.x && bbox.end_coord.y < device_grid.y,
        "transpose sub_core_grids ({},{}) exceeds the device grid ({},{})",
        bbox.end_coord.x,
        bbox.end_coord.y,
        device_grid.x,
        device_grid.y);
    return tt::tt_metal::CoreCoord(bbox.end_coord.x + 1, bbox.end_coord.y + 1);
}

enum class TransposeOpDim { WH, HC, CN, NH, NW, CW };

enum class TransposeOpParallelizationStrategy { MULTI_CORE_WH, MULTI_CORE_HC, MULTI_CORE_CN };

struct TransposeParams {
    TransposeOpDim dim{};
    tt::tt_metal::MemoryConfig output_mem_config;
    float pad_value = 0.0f;
    // Optional restriction of the op to a subset of Tensix cores. Program factories
    // otherwise derive their grid from device->compute_with_storage_grid_size(), which
    // claims the whole grid and so fails under an active SubDevice with
    // "Kernel group cores do not match sub device cores". Mirrors the existing
    // sub_core_grids parameter on concat.
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
};

struct TransposeInputs {
    Tensor input;
};

}  // namespace ttnn::prim
