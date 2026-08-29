// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <optional>

namespace ttnn::prim {

// Which engine serves an argmax call.
//
// This is an INTERNAL decision: ttnn::argmax picks the engine itself (see
// select_argmax_engine in argmax.cpp) and there is no public argument that
// names one. It lives in ArgmaxParams so that the program cache keys on it --
// two calls that agree on everything else but land on different engines must
// not share a cached program.
enum class ArgMaxEngine : uint8_t {
    // The pre-existing scalar reader kernels (single- or multi-core, depending
    // on layout and dim). The only engine that can serve non-Blackhole
    // devices, ROW_MAJOR input, non-last-dim reductions, dim=None, integer and
    // FLOAT32 dtypes, and widths that are not a multiple of the tile width.
    Incumbent,
    // Blackhole TILE-layout last-dim scan on the pack RISC's RVV (Zve32f)
    // vector unit, multicore. Reads TILE directly (no untilize hop) and can
    // fill the optional max-value output. Bit-identical to Incumbent on every
    // input, special values included, at any core count -- the cross-core
    // merge runs the same bit-pattern order the scan does.
    //
    // COST MODEL: linear in the reduction width AND in H, because the scan
    // visits each tile once PER VALID ROW. Measured on one core over an
    // 8192-tile row (V = 262144), trace-replay device time on a p150: 350 us
    // at H = 1 and 5191 us at H = 32, i.e. ~0.043 us per tile for the first
    // row plus ~0.019 us per tile for each further row. That is what makes it
    // the engine for tile-rows holding few valid rows.
    Rvv,
    // Blackhole TILE-layout last-dim reduction on the SFPU (Tensix vector
    // FPU), multicore. Phase 1 reduces all 32 rows of a tile-row in one
    // lane-parallel pass; phase 2 and the cross-core merge are per-row scalar
    // compares on the dataflow RISC. Also fills the optional max-value output.
    //
    // COST MODEL: linear in the reduction width and essentially FLAT in H,
    // because the lane-parallel pass runs whether 1 or 32 of a tile-row's rows
    // are real. Same measurement as above (one core, 8192-tile row): 4875 us
    // at H = 1, 4881 us at H = 8, 4911 us at H = 32 -- ~0.60 us per tile, a
    // 0.7% spread over a 32x change in real work. So it is the engine for
    // batch shapes, and the loser by an order of magnitude at H = 1.
    //
    // Both cost models are throughput-mode trace-replay device time, not
    // single-op latency; regenerate with
    // tests/ttnn/unit_tests/operations/reduce/_argmax_engine_crossover_bench.py.
    // The routing boundary those numbers imply lives in
    // select_argmax_engine (argmax.cpp), which carries the full table.
    //
    // DIVERGES from Incumbent on special values -- the compare is IEEE-on-fp32
    // behind a bf16 gasket, so NaN behaves as same-signed infinity, denormals
    // and -0 flush to +0, and max values below ~2^-118 carry a +2^-127 pack
    // bias (all silicon-measured; see kernels/argmax_sfpu_tile_compute.cpp).
    // Selection therefore never routes here when the caller asked for
    // exact_special_values.
    Sfpu,
};

struct ArgmaxParams {
    tt::tt_metal::DataType output_dtype{};
    std::optional<int> dim;
    bool keepdim{};
    std::optional<CoreRangeSet> sub_core_grids;
    tt::tt_metal::MemoryConfig output_mem_config;
    // Engine chosen by ttnn::argmax; see ArgMaxEngine. The two accelerated
    // engines also enable the optional max-value output (see
    // ArgmaxInputs::optional_maxval_tensor).
    ArgMaxEngine engine{ArgMaxEngine::Incumbent};
};

struct ArgmaxInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
    // Optional preallocated BFLOAT16 ROW_MAJOR tensor (same logical shape as
    // the index output). On the accelerated engines (Rvv / Sfpu) the winning
    // max VALUES are written alongside the indices, so the caller does not
    // need a second pass with ttnn.max to recover them. The Incumbent
    // engine cannot produce it, so supplying it for a call that lands there is
    // a hard error rather than a silently untouched buffer.
    std::optional<Tensor> optional_maxval_tensor;
};

}  // namespace ttnn::prim
