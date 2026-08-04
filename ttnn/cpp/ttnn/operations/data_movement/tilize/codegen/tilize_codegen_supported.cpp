// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_supported.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt_stl/assert.hpp>

#include "tilize_codegen_program_factory.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

bool is_supported_dtype(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::UINT16: return true;
        default: return false;
    }
}

}  // namespace

const char* unsupported_execution_control(const Tile& tile, const std::optional<CoreRangeSet>& sub_core_grids) {
    if (sub_core_grids.has_value()) {
        return "sub_core_grids";
    }
    if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
        return "tile";
    }
    return nullptr;
}

bool supported_by_codegen(const TilizeCodegenParams& operation_attributes, const TilizeCodegenInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        return false;
    }
    // codegen_sharded / codegen_rank are excluded from this manifest's sweep_suite: only the RM
    // interleaved, same-dtype nightly/codegen_dtype/broaden_suite surface is transcribed here.
    if (operation_attributes.input_mem_config.is_sharded() || operation_attributes.output_mem_config.is_sharded()) {
        return false;
    }
    // No sweep vector in codegen_tilize.py's run()/_run_broaden() ever passes a `dtype=` kwarg
    // to TilizeCodegen.tilize(), so a dtype-cast tilize call was never exercised for this port.
    if (operation_attributes.input_dtype != operation_attributes.output_dtype) {
        return false;
    }
    if (!is_supported_dtype(operation_attributes.input_dtype)) {
        return false;
    }

    const auto& shape = input_tensor.logical_shape();
    const uint32_t rank = shape.rank();
    // ttnn::tilize() squeezes rank>4 to 4D before this predicate ever runs (build_ndiml_tilize in
    // tilize.cpp); a rank<2 tensor has no H/W plane to tilize.
    if (rank < 2 || rank > 4) {
        return false;
    }
    for (uint32_t i = 0; i < rank; ++i) {
        if (shape[i] == 0) {
            return false;
        }
    }
    const uint32_t h = shape[rank - 2];
    const uint32_t w = shape[rank - 1];
    // Sub-tile inputs need PadCodegen's pad-then-tilize detour (ops/tilize/tilize.py's
    // `TilizeCodegen.tilize`); TilizeCodegenParams carries no logical/padded shape, so that path
    // is not transcribed here — sub-tile falls back to native.
    if (h % tt::constants::TILE_HEIGHT != 0 || w % tt::constants::TILE_WIDTH != 0) {
        return false;
    }

    // Device-resource feasibility. Every builder's CB footprint scales with the per-core tile
    // count (page = one tile, depth = the core's column chunk / block width), and the reference
    // raises rather than shrinking below the compute-chunk / batched-writer contract
    // (_validate_tilize_pipeline). `auto` therefore has to ask the factory's own sizing whether a
    // plan exists before committing, or a wide row would abort inside program creation on a
    // configuration this gate advertised, where native handles any width.
    //
    // Skipped for a host tensor: the plan needs the device's per-core L1 and grid. A host input is
    // never routed here (the free function only reaches this after the op has a device tensor) and
    // is rejected by the prim's own structural TT_FATALs, not by this scope answer.
    if (is_device_tensor(input_tensor) &&
        !tilize_codegen_cb_plan_fits(input_tensor.device(), operation_attributes, input_tensor)) {
        return false;
    }
    return true;
}

namespace {

// One measured-slow configuration, keyed on the normalized cache-key attributes. NC/Ht/Wt rather
// than the logical shape because that is what every builder's split and CB plan reads: shapes with
// the same tile geometry ([1, 4, 96, 32] and [4, 96, 32]) produce the same program.
struct DemotedCase {
    uint32_t nc;
    uint32_t ht;
    uint32_t wt;
    DataType dtype;
    BufferType output_buffer_type;
};

// Ungeneralized: no mechanism explains these, so each row is an exact match rather than a
// condition. A predicate is not offered because the measurements separate identical tile
// geometries by output placement alone — (NC=40, Ht=2, Wt=2) and (NC=24, Ht=3, Wt=2) are slow in
// L1 and above parity in DRAM, (NC=6, Ht=7, Wt=5) the other way round — and both the row and the
// column split carry rows here as well as above-parity geometries, so any condition over geometry
// or over the dispatched path would have to contradict a measured case. Comments name the ledger
// shape each row came from.
//
// The Wt >= 2 rows are the class the upstream `uses_2d_column_path` change moved onto the 2D
// column split by dropping its `Wt <= 2` bail. Being on that split is not on its own a demotion:
// the geometries whose column blocks stay wide enough to overlap measure at or above parity and
// are deliberately absent — (NC=10, Ht=2, Wt=2) [1, 10, 64, 64], (NC=28, Ht=1, Wt=2)
// [4, 7, 32, 64], (NC=40, Ht=2, Wt=2) in DRAM [5, 8, 64, 64], (NC=24, Ht=3, Wt=2) in DRAM
// [6, 4, 96, 64] and (NC=6, Ht=7, Wt=5) in L1 [6, 224, 160] all cleared the gate on the ported
// kernel. That is why this stays an enumeration rather than a `Wt == 2 && max_tpc == 1`
// predicate: such a predicate would demote every one of them too.
constexpr DemotedCase kDemotedCases[] = {
    {1, 1, 2, DataType::UINT16, BufferType::DRAM},     // [1, 32, 64] / [32, 64]
    {4, 3, 1, DataType::BFLOAT16, BufferType::DRAM},   // [1, 4, 96, 32]
    {4, 3, 1, DataType::BFLOAT16, BufferType::L1},     // [1, 4, 96, 32]
    {2, 3, 1, DataType::INT32, BufferType::DRAM},      // [2, 1, 96, 32]
    {2, 3, 1, DataType::UINT16, BufferType::DRAM},     // [2, 1, 96, 32]
    {2, 3, 1, DataType::UINT32, BufferType::DRAM},     // [2, 1, 96, 32]
    {2, 3, 1, DataType::INT32, BufferType::L1},        // [2, 1, 96, 32]
    {2, 3, 1, DataType::UINT16, BufferType::L1},       // [2, 1, 96, 32]
    {2, 3, 1, DataType::UINT32, BufferType::L1},       // [2, 1, 96, 32]
    {24, 2, 3, DataType::BFLOAT16, BufferType::DRAM},  // [2, 12, 64, 96]
    {24, 2, 3, DataType::BFLOAT16, BufferType::L1},    // [2, 12, 64, 96]
    {2, 1, 1, DataType::INT32, BufferType::DRAM},      // [2, 32, 32]
    {2, 1, 1, DataType::UINT16, BufferType::DRAM},     // [2, 32, 32]
    {2, 1, 1, DataType::UINT32, BufferType::DRAM},     // [2, 32, 32]
    {2, 1, 1, DataType::INT32, BufferType::L1},        // [2, 32, 32]
    {2, 1, 1, DataType::UINT16, BufferType::L1},       // [2, 32, 32]
    {2, 1, 1, DataType::UINT32, BufferType::L1},       // [2, 32, 32]
    {2, 3, 1, DataType::FLOAT32, BufferType::DRAM},    // [2, 96, 32]
    {2, 3, 1, DataType::FLOAT32, BufferType::L1},      // [2, 96, 32]
    {1, 7, 1, DataType::BFLOAT16, BufferType::DRAM},   // [224, 32]
    {1, 7, 1, DataType::BFLOAT16, BufferType::L1},     // [224, 32]
    {6, 4, 1, DataType::FLOAT32, BufferType::DRAM},    // [3, 2, 128, 32]
    {6, 4, 1, DataType::FLOAT32, BufferType::L1},      // [3, 2, 128, 32]
    {6, 2, 1, DataType::FLOAT32, BufferType::DRAM},    // [3, 2, 64, 32]
    {6, 2, 1, DataType::FLOAT32, BufferType::L1},      // [3, 2, 64, 32]
    {6, 3, 1, DataType::FLOAT32, BufferType::DRAM},    // [3, 2, 96, 32]
    {6, 3, 1, DataType::FLOAT32, BufferType::L1},      // [3, 2, 96, 32]
    {21, 2, 3, DataType::BFLOAT16, BufferType::DRAM},  // [3, 7, 64, 96]
    {21, 2, 3, DataType::BFLOAT16, BufferType::L1},    // [3, 7, 64, 96]
    {24, 3, 1, DataType::BFLOAT16, BufferType::DRAM},  // [3, 8, 96, 32] / [4, 6, 96, 32]
    {24, 3, 1, DataType::BFLOAT16, BufferType::L1},    // [3, 8, 96, 32] / [4, 6, 96, 32]
    {3, 3, 1, DataType::FLOAT32, BufferType::DRAM},    // [3, 96, 32]
    {3, 3, 1, DataType::FLOAT32, BufferType::L1},      // [3, 96, 32]
    {1, 1, 1, DataType::INT32, BufferType::DRAM},      // [32, 32]
    {1, 1, 1, DataType::UINT16, BufferType::DRAM},     // [32, 32]
    {1, 1, 1, DataType::UINT32, BufferType::DRAM},     // [32, 32]
    {1, 1, 1, DataType::INT32, BufferType::L1},        // [32, 32]
    {1, 1, 1, DataType::UINT16, BufferType::L1},       // [32, 32]
    {1, 1, 1, DataType::UINT32, BufferType::L1},       // [32, 32]
    {48, 3, 3, DataType::BFLOAT16, BufferType::DRAM},  // [4, 12, 96, 96]
    {48, 3, 3, DataType::BFLOAT16, BufferType::L1},    // [4, 12, 96, 96]
    {36, 2, 1, DataType::BFLOAT16, BufferType::DRAM},  // [4, 9, 64, 32]
    {36, 2, 1, DataType::BFLOAT16, BufferType::L1},    // [4, 9, 64, 32]
    {5, 4, 1, DataType::BFLOAT16, BufferType::DRAM},   // [5, 128, 32]
    {5, 4, 1, DataType::BFLOAT16, BufferType::L1},     // [5, 128, 32]
    {15, 2, 1, DataType::BFLOAT16, BufferType::DRAM},  // [5, 3, 64, 32]
    {15, 2, 1, DataType::BFLOAT16, BufferType::L1},    // [5, 3, 64, 32]
    {40, 2, 2, DataType::BFLOAT16, BufferType::L1},    // [5, 8, 64, 64] (DRAM twin is above parity)
    {60, 1, 2, DataType::BFLOAT16, BufferType::DRAM},  // [6, 10, 32, 64]
    {60, 1, 2, DataType::BFLOAT16, BufferType::L1},    // [6, 10, 32, 64]
    {6, 7, 5, DataType::BFLOAT16, BufferType::DRAM},   // [6, 224, 160] (L1 twin is above parity)
    {24, 3, 2, DataType::BFLOAT16, BufferType::L1},    // [6, 4, 96, 64] (DRAM twin is above parity)
    {1, 2, 1, DataType::BFLOAT16, BufferType::DRAM},   // [64, 32]
    {1, 2, 1, DataType::INT32, BufferType::DRAM},      // [64, 32]
    {1, 2, 1, DataType::UINT16, BufferType::DRAM},     // [64, 32]
    {1, 2, 1, DataType::UINT32, BufferType::DRAM},     // [64, 32]
    {1, 2, 1, DataType::BFLOAT16, BufferType::L1},     // [64, 32]
    {1, 2, 1, DataType::INT32, BufferType::L1},        // [64, 32]
    {1, 2, 1, DataType::UINT16, BufferType::L1},       // [64, 32]
    {1, 2, 1, DataType::UINT32, BufferType::L1},       // [64, 32]
    {9, 4, 1, DataType::BFLOAT16, BufferType::DRAM},   // [9, 128, 32]
    {9, 4, 1, DataType::BFLOAT16, BufferType::L1},     // [9, 128, 32]
    {9, 5, 3, DataType::BFLOAT16, BufferType::DRAM},   // [9, 160, 96]
    {9, 5, 3, DataType::BFLOAT16, BufferType::L1},     // [9, 160, 96]
};

}  // namespace

bool is_demoted(const TilizeCodegenParams& operation_attributes, const TilizeCodegenInputs&) {
    // A caller-forced single-core route. tilize_codegen_dispatch's RowSingleCore condition, asked
    // directly so this needs no device: use_multicore=false / use_low_perf leave one worker, where
    // codegen's pipelining has nothing to overlap, and native has a route built for that request.
    // Outside the swept space (no sweep vector varies either flag), so no ledger entry arbitrates
    // it — the demotion rests on the absence of parallelism, not on a measurement.
    if (!operation_attributes.use_multicore || operation_attributes.use_low_perf) {
        return true;
    }

    // supported_by_codegen has already rejected a dtype-cast call, so input_dtype is the case's
    // dtype; output placement is the axis the ledger varies (vector_map's memory_config kwarg).
    for (const auto& c : kDemotedCases) {
        if (c.nc == operation_attributes.NC && c.ht == operation_attributes.Ht && c.wt == operation_attributes.Wt &&
            c.dtype == operation_attributes.input_dtype &&
            c.output_buffer_type == operation_attributes.output_mem_config.buffer_type()) {
            return true;
        }
    }
    return false;
}

ImplementationSelector parse_implementation(std::string_view implementation) {
    if (implementation == "auto") {
        return ImplementationSelector::Auto;
    }
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_THROW("tilize: unknown implementation selector '{}'", implementation);
}

}  // namespace ttnn::prim
