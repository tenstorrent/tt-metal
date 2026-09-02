// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <optional>

namespace ttnn::prim {

// Which path serves an argmax call. Internal: ttnn::argmax picks it from the
// input spec (select_argmax_path in argmax.cpp). Stored in ArgmaxParams so the
// program cache keys on it -- calls differing only in path must not share one.
enum class ArgMaxPath : uint8_t {
    // The pre-existing scalar reader kernels; the only path that serves
    // non-Blackhole devices, ROW_MAJOR input, non-last-dim reductions,
    // dim=None, integer and FLOAT32 dtypes, and reduction widths that are not a
    // multiple of the tile width.
    ScalarReader,
    // Blackhole TILE last-dim scan on the pack RISC's RVV (Zve32f) unit;
    // bit-identical to the scalar readers on every input at any core count.
    // Chosen below kSfpuMinRows rows.
    Rvv,
    // Blackhole TILE last-dim reduction on the SFPU, all 32 rows of a tile-row
    // per pass. Chosen at or above kSfpuMinRows rows, and never under
    // exact_special_values: the compare is IEEE-on-fp32 behind a bf16 gasket,
    // so NaN, denormals, -0 and tiny max values diverge from the scalar readers
    // (measured; see kernels/argmax_sfpu_tile_compute.cpp).
    Sfpu,
};

struct ArgmaxParams {
    tt::tt_metal::DataType output_dtype{};
    std::optional<int> dim;
    bool keepdim{};
    std::optional<CoreRangeSet> sub_core_grids;
    tt::tt_metal::MemoryConfig output_mem_config;
    // Path chosen by ttnn::argmax; see ArgMaxPath.
    ArgMaxPath path{ArgMaxPath::ScalarReader};
};

struct ArgmaxInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
    // Optional preallocated BFLOAT16 ROW_MAJOR tensor (same logical shape as the
    // index output), filled with the winning max VALUES by Rvv and Sfpu. The
    // scalar readers cannot, so supplying it there is a hard error.
    std::optional<Tensor> optional_maxval_tensor;
};

}  // namespace ttnn::prim
