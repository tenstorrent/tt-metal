// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <optional>

namespace ttnn::prim {

// Which path serves an argmax call.
//
// This is an INTERNAL decision: ttnn::argmax picks the path itself (see
// select_argmax_path in argmax.cpp) and there is no public argument that names
// one. It lives in ArgmaxParams so that the program cache keys on it -- two
// calls that agree on everything else but land on different paths must not
// share a cached program.
//
// Measurements and the threshold rationale live next to kSfpuMinRows in
// argmax.cpp.
enum class ArgMaxPath : uint8_t {
    // The pre-existing scalar reader kernels (single- or multi-core, depending
    // on layout and dim). The only path that can serve non-Blackhole devices,
    // ROW_MAJOR input, non-last-dim reductions, dim=None, integer and FLOAT32
    // dtypes, and widths that are not a multiple of the tile width.
    ScalarReader,
    // Blackhole TILE-layout last-dim scan on the pack RISC's RVV (Zve32f)
    // vector unit, multicore. Reads TILE directly (no untilize hop) and can
    // fill the optional max-value output. Bit-identical to the scalar readers
    // on every input, special values included, at any core count -- the
    // cross-core merge runs the same bit-pattern order the scan does. Cost is
    // linear in the reduction width AND in H: the scan visits each tile once
    // PER VALID ROW.
    Rvv,
    // Blackhole TILE-layout last-dim reduction on the SFPU (Tensix vector
    // FPU), multicore. Phase 1 reduces all 32 rows of a tile-row in one
    // lane-parallel pass, so cost is linear in the reduction width and
    // essentially FLAT in H. Also fills the optional max-value output.
    // DIVERGES from the scalar readers on special values: the compare is
    // IEEE-on-fp32 behind a bf16 gasket, so NaN behaves as same-signed
    // infinity, denormals and -0 flush to +0, and max values below ~2^-118
    // carry a +2^-127 pack bias (all silicon-measured; see
    // kernels/argmax_sfpu_tile_compute.cpp). Selection therefore never routes
    // here when the caller asked for exact_special_values.
    Sfpu,
};

struct ArgmaxParams {
    tt::tt_metal::DataType output_dtype{};
    std::optional<int> dim;
    bool keepdim{};
    std::optional<CoreRangeSet> sub_core_grids;
    tt::tt_metal::MemoryConfig output_mem_config;
    // Path chosen by ttnn::argmax; see ArgMaxPath. The two accelerated paths
    // also enable the optional max-value output (see
    // ArgmaxInputs::optional_maxval_tensor).
    ArgMaxPath path{ArgMaxPath::ScalarReader};
};

struct ArgmaxInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
    // Optional preallocated BFLOAT16 ROW_MAJOR tensor (same logical shape as
    // the index output). On the accelerated paths (Rvv / Sfpu) the winning max
    // VALUES are written alongside the indices, so the caller does not need a
    // second pass with ttnn.max to recover them. The scalar readers cannot
    // produce it, so supplying it for a call that lands there is a hard error
    // rather than a silently untouched buffer.
    std::optional<Tensor> optional_maxval_tensor;
};

}  // namespace ttnn::prim
