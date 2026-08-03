// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_supported.hpp"

#include <algorithm>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt_stl/assert.hpp>

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

// Exact-match ledger entries that demotion analysis could NOT reduce to a mechanism. Each is a
// single measured configuration, not a condition — see is_demoted() for the generalized row-path
// predicates that cover the rest of the ledger. Fields are the sweep's shape, the buffer type of
// the OUTPUT memory config, and the dtype (same in and out — this port never routes a dtype-cast
// call to codegen, see supported_by_codegen).
//
// The ledger keys its `memory_config` field on the sweep's `output_memory_config` parameter
// (manifest vector_map), which the sweep varies independently of `input_a_memory_config`; a ledger
// entry marked L1 therefore constrains the output placement only, and matching on the input's
// buffer type does not identify these cases.
struct DemotedCase {
    std::vector<uint32_t> shape;
    BufferType buffer_type;
    DataType dtype;
};

const std::vector<DemotedCase>& ungeneralized_demoted_cases() {
    static const std::vector<DemotedCase> cases = {
        // UNGENERALIZED: row path at Wt == 5. Wt == 5 is prime, so choose_2d_ncol can never find a
        // divisor d >= 2 that fits and these fall to build_row no matter how much grid is free —
        // but Wt == 5 alone is not the condition (other Wt == 5 shapes clear/straddle parity), so
        // this stays enumerated rather than becoming a predicate.
        {{4, 224, 160}, BufferType::DRAM, DataType::BFLOAT16},
        {{4, 224, 160}, BufferType::L1, DataType::BFLOAT16},
        // The former Wt == 3 ungeneralized/L1 entries ({3,7,64,96}, {5,160,96}, and the
        // wt3_l1_source predicate) are removed: phase-7 root-caused the Wt == 3 row-path loss to a
        // program-factory defect (write_batch not a multiple of chunk_wt degenerating the
        // batched-writer pipeline to lockstep, fixed in build_row) rather than a genuine
        // performance ceiling, so none of that family is demoted anymore.
        //
        // round 5/8 resolution: [7,96,160] L1 bf16 (Wt == 5, total_Ht == 21) was reclassified by
        // analysis as a marginal, not-credibly-established loss (CI straddles parity, DRAM sibling
        // and nearest same-regime L1 siblings both win), but reroutes were exhausted for this round,
        // so it stays demoted rather than left on codegen unmeasured.
        {{7, 96, 160}, BufferType::L1, DataType::BFLOAT16},
    };
    return cases;
}

}  // namespace

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
    return true;
}

bool is_demoted(const TilizeCodegenParams& operation_attributes, const TilizeCodegenInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    // Every predicate below is a property of the 1-D row path (build_row in
    // tilize_codegen_program_factory.cpp), which is a transliteration of native's
    // TilizeMultiCoreDefaultProgramFactory: same split over NC*Ht tile-rows, same 32-stick reader
    // loop, same per-tile compute. There is no structural win to recover there, only codegen's
    // deeper CB / batched writer traded against a larger per-core setup, so the row path loses
    // where it cannot pipeline. The 2D-column and block paths are untouched by all of this.
    const uint32_t total_ht = operation_attributes.NC * operation_attributes.Ht;
    const uint32_t wt = operation_attributes.Wt;
    // Same grid query the factory's path dispatch uses (64 cores on wormhole_b0).
    const CoreCoord grid = input_tensor.device()->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    // The ledger's placement axis is the output memory config (manifest vector_map maps
    // `memory_config` -> the sweep's `output_memory_config`), so every placement-conditioned
    // demotion below keys on this side.
    const auto output_buffer_type = operation_attributes.output_mem_config.buffer_type();

    // tilize_rm_row_path_wt1: Wt == 1 forces uses_2d_column_path() -> ncol == 1 (Wt <= 2), so
    // build_row is selected for any NC*Ht. There the minimal_work clamp sets cb_depth = 1 and
    // write_batch = 1, leaving exactly native's serialized read -> tilize -> write with a larger
    // per-core setup: uniformly 12-17% behind native, device_vs_generic_op 0.98-1.02.
    if (wt == 1) {
        return true;
    }

    // tilize_rm_row_path_wt2_one_row_per_core: Wt == 2 also forces ncol == 1 (uses_2d_column_path
    // short-circuits at Wt <= 2), so build_row is selected for any NC*Ht. When NC*Ht <= grid_cores
    // every core owns exactly one tile-row (2 tiles): the read/barrier count matches native's
    // reader_unary_stick_layout_split_rows_multicore exactly, so codegen's heavier per-core setup
    // (unified-reader named-CT dispatch, TensorAccessor construction, deeper CB plan, a write batch
    // of 4 a 2-tile core never fills) is paid with no transport advantage to trade against it. Once
    // NC*Ht exceeds grid_cores a core owns >= 2 tile-rows and codegen's double-buffered CB_IN lets
    // row n+1 prefetch while compute drains row n - the only structural edge codegen has here - so
    // DRAM crosses back above parity there. Interleaved L1 keeps the underlying transfer cheap
    // enough that the fixed per-core cost stays dominant regardless of tile-rows per core, so the
    // L1 arm has no NC*Ht bound.
    if (wt == 2 && (total_ht <= grid_cores || output_buffer_type == BufferType::L1)) {
        return true;
    }

    // tilize_rm_row_path_wt3_no_column_split: choose_tilize_2d_ncol can only pick a divisor
    // d >= 2 of Wt with total_ht * d <= grid_cores; for Wt == 3 the only candidate is d == 3, so
    // once 3 * total_ht > grid_cores the 2D column split is unreachable and create_descriptor
    // falls through to build_row. build_row is then a structural clone of native's
    // TilizeMultiCoreDefaultProgramFactory (same 32-stick reader loop, same one-tile-row-per-core
    // split), so codegen's extra per-core setup (batched-writer priming, deeper CB bookkeeping)
    // has no transport advantage to offset. Wt == 3 with 3 * total_ht <= grid_cores keeps the
    // column split and wins, so the bound is exact, not a shape list.
    if (wt == 3 && 3 * total_ht > grid_cores) {
        return true;
    }

    const auto& shape = input_tensor.logical_shape();
    std::vector<uint32_t> shape_vec;
    shape_vec.reserve(shape.rank());
    for (uint32_t i = 0; i < shape.rank(); ++i) {
        shape_vec.push_back(shape[i]);
    }
    for (const auto& demoted : ungeneralized_demoted_cases()) {
        if (demoted.buffer_type == output_buffer_type && demoted.dtype == operation_attributes.input_dtype &&
            demoted.shape == shape_vec) {
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
