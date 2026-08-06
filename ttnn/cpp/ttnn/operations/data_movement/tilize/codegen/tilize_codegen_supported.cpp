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

// UNGENERALIZED exact matches: no mechanism was identified for these, so each row is one measured
// configuration rather than a condition. This is the floor when analysis finds no predicate, not a
// preferred form.
//
// Comments name the ledger shape each row came from; shapes that normalize to the same NC/Ht/Wt
// are one row, since they produce the same program.
// Rows whose Wt is 1 are here only because their total_Ht exceeds the core count, which puts them
// outside the general Wt == 1 predicate below (that one is deliberately bounded to one tile-row per
// core); every Wt == 1 case at or under the bound is covered there and is absent here.
constexpr DemotedCase kUngeneralizedDemotedCases[] = {
    // [1, 32, 64] and [32, 64] — NC 1, Ht 1, Wt 2. Same tile geometry, one row.
    // DRAM only: the L1 pair cleared the gate on the ported kernel.
    {1, 1, 2, DataType::UINT16, BufferType::DRAM},
    // [1, 1, 64, 64] — NC 1, Ht 2, Wt 2. L1 only; DRAM cleared the gate on the ported kernel.
    {1, 2, 2, DataType::UINT16, BufferType::L1},
    // [1, 10, 64, 64] — NC 10, Ht 2, Wt 2.
    {10, 2, 2, DataType::BFLOAT16, BufferType::DRAM},
    // [2, 12, 64, 96] — NC 24, Ht 2, Wt 3.
    {24, 2, 3, DataType::BFLOAT16, BufferType::DRAM},
    // [3, 7, 64, 96] — NC 21, Ht 2, Wt 3.
    {21, 2, 3, DataType::BFLOAT16, BufferType::DRAM},
    {21, 2, 3, DataType::BFLOAT16, BufferType::L1},
    // [3, 8, 96, 32] and [4, 6, 96, 32] — NC 24, Ht 3, Wt 1, total_Ht 72.
    {24, 3, 1, DataType::BFLOAT16, BufferType::DRAM},
    {24, 3, 1, DataType::BFLOAT16, BufferType::L1},
    // [4, 12, 96, 96] — NC 48, Ht 3, Wt 3.
    {48, 3, 3, DataType::BFLOAT16, BufferType::DRAM},
    {48, 3, 3, DataType::BFLOAT16, BufferType::L1},
    // [4, 224, 64] — NC 4, Ht 7, Wt 2. DRAM cleared the gate on the ported kernel; L1 did not.
    {4, 7, 2, DataType::BFLOAT16, BufferType::L1},
    // [4, 7, 32, 64] — NC 28, Ht 1, Wt 2. DRAM cleared the gate on the ported kernel.
    {28, 1, 2, DataType::BFLOAT16, BufferType::L1},
    // [4, 9, 64, 32] — NC 36, Ht 2, Wt 1, total_Ht 72.
    {36, 2, 1, DataType::BFLOAT16, BufferType::DRAM},
    {36, 2, 1, DataType::BFLOAT16, BufferType::L1},
    // [5, 160, 96] — NC 5, Ht 5, Wt 3.
    {5, 5, 3, DataType::BFLOAT16, BufferType::DRAM},
    // [5, 8, 64, 64] — NC 40, Ht 2, Wt 2.
    {40, 2, 2, DataType::BFLOAT16, BufferType::L1},
    // [6, 10, 32, 64] — NC 60, Ht 1, Wt 2.
    {60, 1, 2, DataType::BFLOAT16, BufferType::DRAM},
    {60, 1, 2, DataType::BFLOAT16, BufferType::L1},
    // [6, 224, 160] — NC 6, Ht 7, Wt 5. DRAM cleared the gate on the ported kernel; L1 did not.
    {6, 7, 5, DataType::BFLOAT16, BufferType::L1},
    // [6, 4, 96, 64] — NC 24, Ht 3, Wt 2.
    {24, 3, 2, DataType::BFLOAT16, BufferType::L1},
    // [9, 160, 96] — NC 9, Ht 5, Wt 3.
    {9, 5, 3, DataType::BFLOAT16, BufferType::DRAM},
    {9, 5, 3, DataType::BFLOAT16, BufferType::L1},
};

}  // namespace

bool is_demoted(const TilizeCodegenParams& operation_attributes, const TilizeCodegenInputs& tensor_args) {
    // A caller-forced single-core route. tilize_codegen_dispatch's RowSingleCore condition, asked
    // directly so this needs no device: use_multicore=false / use_low_perf leave one worker, where
    // codegen's pipelining has nothing to overlap, and native has a route built for that request.
    // Outside the swept space (no sweep vector varies either flag), so no ledger entry arbitrates
    // it — the demotion rests on the absence of parallelism, not on a measurement.
    if (!operation_attributes.use_multicore || operation_attributes.use_low_perf) {
        return true;
    }

    // tilize-rm-single-tile-column. Wt == 1: the padded last dimension is a single tile wide
    // (supported_by_codegen already requires W % TILE_WIDTH == 0, so exactly W == 32).
    // choose_tilize_2d_ncol returns 1 for wt < 2, so the case can never reach the 2D-column path and
    // always lands on the row path with chunk_wt = 1, num_col_chunks = 1 — which also trips
    // compute_row_shape's minimal_work clamp (cb_depth 1, cb_out depth 1, write_batch 1: no double
    // buffering and no write batching). The resulting reader/writer are then structural equivalents
    // of native's reader_unary_stick_layout_split_rows_multicore / writer_unary_interleaved_start_id
    // on the same core count (native's nblocks is also total_Ht), so there is no data-movement or
    // parallelism advantage left to offset the one remaining difference, which is a pure deficit:
    // native's compute kernel takes the per-block tile count as a template parameter and picks its
    // fp32 mode at compile time (ttnn/cpp/ttnn/kernel/compute/tilize.cpp), while compute_tilize.cpp
    // receives chunk_Wt as a runtime arg and calls the unspecialized tilize_init/tilize_block.
    // Structural, not payload-sized: holds for every dtype and both interleaved placements. Wt >= 2
    // is the winning side.
    //
    // Bounded by total_Ht <= core count: above it, each core owns several tile-rows instead of one,
    // which is a different same-kernel parity gap, not this one — do not widen this predicate to
    // cover that regime. The measured points from it sit in the table above instead.
    //
    // Needs the device for the core count. A host tensor is never routed here (the free function only
    // reaches this with a device tensor); skipping the check for one keeps this from dereferencing a
    // null device, and correctness never depends on a perf gate's answer.
    if (operation_attributes.Wt == 1 && is_device_tensor(tensor_args.input_tensor)) {
        const CoreCoord grid = tensor_args.input_tensor.device()->compute_with_storage_grid_size();
        const uint32_t num_cores = grid.x * grid.y;
        const uint32_t total_ht = operation_attributes.NC * operation_attributes.Ht;
        if (total_ht <= num_cores) {
            return true;
        }
    }

    // supported_by_codegen has already rejected a dtype-cast call, so input_dtype is the case's
    // dtype; output placement is the axis the ledger varies (vector_map's memory_config kwarg).
    for (const auto& c : kUngeneralizedDemotedCases) {
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
