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

bool is_demoted(const TilizeCodegenParams& operation_attributes, const TilizeCodegenInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    // The dispatch query needs the device's grid and per-core L1. A host tensor has no answer and
    // no perf question: leave it to the prim's structural TT_FATALs.
    if (!is_device_tensor(input_tensor)) {
        return false;
    }
    // The general arms below are conditions on WHICH builder create_descriptor selects and on the
    // work that builder hands each core, so they key on the shared dispatch query rather than on
    // shapes. Where measurement found no such condition, the ledger entry is an enumerated exact
    // match, flagged UNGENERALIZED at its branch.
    const uint32_t total_ht = operation_attributes.NC * operation_attributes.Ht;
    // Same grid query the factory's path dispatch uses (64 cores on wormhole_b0).
    const CoreCoord grid = input_tensor.device()->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const auto dispatch = tilize_codegen_dispatch(input_tensor.device(), operation_attributes, input_tensor);

    // UNGENERALIZED. One-tile-per-core column splits are NOT demoted as a class: thirteen such
    // configurations measured above parity (native/ported 1.02-2.18, generic/ported ~1.00) against
    // these three below it, and nothing separates them, so these are enumerated exact matches.
    // build_2d_column's `minimal_work` clamp (CB depth 1, write_batch 1) applies to all of them
    // alike, which is why it is not the predicate.
    if (dispatch.path == TilizeCodegenPath::Column && dispatch.max_tiles_per_column_block == 1) {
        // The ledger's DRAM/L1 discriminator is the OUTPUT placement (the sweep varies
        // output_memory_config while every input is interleaved DRAM), so these arms key on
        // output_mem_config; keying on the input would collapse each pair into one answer.
        const bool out_is_l1 = operation_attributes.output_mem_config.buffer_type() == BufferType::L1;
        const uint32_t nc = operation_attributes.NC;
        const uint32_t ht = operation_attributes.Ht;
        const uint32_t wt = operation_attributes.Wt;
        // [1, 32, 64] and [32, 64], uint16, DRAM out — a single tile-row split two ways.
        if (operation_attributes.input_dtype == DataType::UINT16 && !out_is_l1 && nc == 1 && ht == 1 && wt == 2) {
            return true;
        }
        // [12, 32, 160], bfloat16, L1 out — twelve tile-rows split five ways.
        if (operation_attributes.input_dtype == DataType::BFLOAT16 && out_is_l1 && nc == 12 && ht == 1 && wt == 5) {
            return true;
        }
    }

    // A column split whose blocks are not all the same width needs a second reader/compute kernel
    // instance for the one-tile-wider cliff group, doubling the program's kernel count over the
    // same core set. The two non-divisor configurations measured on wormhole_b0 (Wt=7 split five
    // ways, Wt=5 split three ways) beat native on device (~2.1x / ~1.5x, generic parity) but lost
    // wall-clock (0.68 / 0.64 of native): the extra group's host-side program cost outweighs the
    // device saving at these sizes. Every column configuration at or above wall-clock parity has a
    // uniform width, so `auto` keeps the cliff on native while forced codegen still exercises it.
    if (dispatch.path == TilizeCodegenPath::Column && (operation_attributes.Wt % dispatch.ncol) != 0) {
        return true;
    }

    // A caller-forced single-core route (use_multicore=false / use_low_perf) has no parallelism for
    // codegen's pipeline to exploit, and native has a factory built for exactly that request.
    if (dispatch.path == TilizeCodegenPath::RowSingleCore) {
        return true;
    }

    // The row path unless it fills the grid with exactly one tile-row per core. build_row is a
    // transliteration of native's TilizeMultiCoreDefaultProgramFactory — the same 1-D split over
    // NC*Ht tile-rows, the same TILE_H-stick reader loop, the same per-tile compute — so it has no
    // structural advantage to trade against its heavier per-core setup. Its two edges over native
    // are a double-buffered input CB and the batched writer, and both need one full tile-row per
    // core on every core to pay off: with total_Ht > grid_cores a core owns several tile-rows,
    // which forces write_batch back to 1 (writer_tilize_interleaved's batched branch mis-orders
    // rows), and with total_Ht < grid_cores the split leaves cores idle that the column split was
    // unable to recruit (grid_cores / total_Ht < 2, or Wt == 1).
    //
    // This arm carries the bulk of the demoted ledger (~65 entries). Its boundary position rests on
    // a single measured row-path configuration above parity (total_Ht == grid_cores) rather than on
    // a swept range, plus the two reversals below.
    if (dispatch.path == TilizeCodegenPath::Row && total_ht != grid_cores) {
        // UNGENERALIZED reversals. Two row-path configurations measured above parity on the ported
        // kernel (native/ported 1.09 and 1.04, generic/ported ~1.00) and are no longer demoted.
        // Both sit within ~9% of parity and both have an L1-output twin that measured below it, so
        // the separating term is the OUTPUT buffer type (what the sweep varies) and nothing more
        // general is claimed here.
        const bool out_is_dram = operation_attributes.output_mem_config.buffer_type() == BufferType::DRAM;
        const uint32_t nc = operation_attributes.NC;
        const uint32_t ht = operation_attributes.Ht;
        const uint32_t wt = operation_attributes.Wt;
        if (operation_attributes.input_dtype == DataType::BFLOAT16 && out_is_dram && wt == 2 &&
            // [5, 8, 64, 64] (total_Ht 80) and [6, 4, 96, 64] (total_Ht 72).
            ((nc == 40 && ht == 2) || (nc == 24 && ht == 3))) {
            return false;
        }
        return true;
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
