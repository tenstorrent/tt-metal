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
// predicates that cover the rest of the ledger. Fields are the sweep's shape, the memory config's
// buffer type (input and output share it: the sweep's vector_map feeds one `memory_config` and the
// input is created with it), and the dtype (same in and out — this port never routes a dtype-cast
// call to codegen, see supported_by_codegen).
struct DemotedCase {
    std::vector<uint32_t> shape;
    BufferType buffer_type;
    DataType dtype;
};

const std::vector<DemotedCase>& ungeneralized_demoted_cases() {
    static const std::vector<DemotedCase> cases = {
        // UNGENERALIZED: row path at Wt == 3, DRAM source. Its L1 twin is covered by the
        // wt3_l1_source predicate; no mechanism was found for the DRAM leg of these two shapes.
        {{3, 7, 64, 96}, BufferType::DRAM, DataType::BFLOAT16},
        {{5, 160, 96}, BufferType::DRAM, DataType::BFLOAT16},
        // UNGENERALIZED: row path at Wt == 2 with NC*Ht > grid_cores, i.e. outside the
        // wt2_one_row_per_core boundary (where the CB does pipeline across tile-rows and the
        // DRAM twins of both shapes measured above parity). No mechanism found.
        {{5, 8, 64, 64}, BufferType::L1, DataType::BFLOAT16},
        {{6, 4, 96, 64}, BufferType::L1, DataType::BFLOAT16},
        // UNGENERALIZED: row path at Wt == 5, DRAM source. Phase 7 reported this as a port "FIX"
        // at generic/ported = 0.998 — the port reproduces the codegen prototype to within 0.2%,
        // so the 1.3% native deficit is the prototype's, not a translation defect. No mechanism.
        {{6, 224, 160}, BufferType::DRAM, DataType::BFLOAT16},
    };
    return cases;
}

// Largest column-block count the 2D-column tilize split can use, mirroring
// _choose_tilize_2d_ncol / uses_2d_column_path in ops/tilize/spec.py (and the transliteration in
// tilize_codegen_program_factory.cpp): the largest divisor d >= 2 of Wt with total_Ht * d <= cores.
// 1 means "no 2D column split", i.e. the program factory falls through to build_row.
uint32_t choose_2d_ncol(uint32_t total_ht, uint32_t wt, uint32_t grid_cores) {
    if (total_ht >= grid_cores || wt < 2) {
        return 1;
    }
    const uint32_t max_ncol = std::min(grid_cores / total_ht, wt);
    uint32_t best = 1;
    for (uint32_t d = 2; d <= max_ncol; ++d) {
        if (wt % d == 0) {
            best = d;
        }
    }
    return best;
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
    const auto input_buffer_type = operation_attributes.input_mem_config.buffer_type();

    // tilize_rm_row_path_wt1: Wt == 1 forces uses_2d_column_path() -> ncol == 1 (Wt <= 2), so
    // build_row is selected for any NC*Ht. There the minimal_work clamp sets cb_depth = 1 and
    // write_batch = 1, leaving exactly native's serialized read -> tilize -> write with a larger
    // per-core setup: uniformly 12-17% behind native, device_vs_generic_op 0.98-1.02.
    if (wt == 1) {
        return true;
    }

    // tilize_rm_row_path_wt2_one_row_per_core: Wt == 2 also forces ncol == 1. The minimal_work
    // clamp does not fire (chunk_wt == 2), so cb_depth = 2 and write_batch = 4 are configured, but
    // with at most one tile-row per core there are only 2 output tiles per core - below the 4-deep
    // write batch's priming depth, and with no second tile-row for the double-buffered CB to
    // prefetch. Bounded at NC*Ht <= grid_cores: past that each core owns several tile-rows, the CB
    // pipelines across them, and the path measures above parity.
    if (wt == 2 && total_ht <= grid_cores) {
        return true;
    }

    // tilize_rm_row_path_wt3_l1_source: interleaved-L1 sourcing makes the reader's 32 stick reads
    // per tile-row core-to-core L1 transfers rather than DRAM reads, which speeds up the shared
    // part of both legs while codegen's fixed per-core setup is unchanged - enough to cross below
    // parity on the Wt == 3 row path. Restricted to Wt == 3 falling to the row path, i.e. no
    // divisor d >= 2 of Wt fits (for Wt == 3: NC*Ht > grid_cores/3), and to an L1 source: the same
    // shapes sourced from DRAM cleared the gate, and Wt == 3 with a 2D-column split wins outright.
    if (wt == 3 && input_buffer_type == BufferType::L1 && choose_2d_ncol(total_ht, wt, grid_cores) == 1) {
        return true;
    }

    const auto& shape = input_tensor.logical_shape();
    std::vector<uint32_t> shape_vec;
    shape_vec.reserve(shape.rank());
    for (uint32_t i = 0; i < shape.rank(); ++i) {
        shape_vec.push_back(shape[i]);
    }
    const auto output_buffer_type = operation_attributes.output_mem_config.buffer_type();
    for (const auto& demoted : ungeneralized_demoted_cases()) {
        if (demoted.buffer_type == input_buffer_type && demoted.buffer_type == output_buffer_type &&
            demoted.dtype == operation_attributes.input_dtype && demoted.shape == shape_vec) {
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
