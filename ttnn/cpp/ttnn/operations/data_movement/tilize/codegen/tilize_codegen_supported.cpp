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

// Fully qualified: unity builds share a translation unit with code that pulls in tt::Tile
// (hostdevcommon/kernel_structs.h), which makes a bare `Tile` ambiguous with tt::tt_metal::Tile.
const char* unsupported_execution_control(
    const tt::tt_metal::Tile& tile, const std::optional<CoreRangeSet>& sub_core_grids) {
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

    // Widest swept Wt is 64 ([1, 1, 512, 2048]); past that the port is unmeasured and the block
    // builder's CB-fit plan is not sufficient evidence: a Wt == 162816 input ([160, 5210112])
    // passes the plan check and then wedges physical cores at run time, where native's block path
    // handles it. Only the measured envelope is claimed.
    constexpr uint32_t kMaxMeasuredWt = 64;
    if ((w + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH > kMaxMeasuredWt) {
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
    // A caller-forced single-core route. tilize_codegen_dispatch's RowSingleCore condition, asked
    // directly so this needs no device: use_multicore=false / use_low_perf leave one worker, where
    // codegen's pipelining has nothing to overlap, and native has a route built for that request.
    // Outside the swept space (no sweep vector varies either flag), so no ledger entry arbitrates
    // it — the demotion rests on the absence of parallelism, not on a measurement.
    if (!operation_attributes.use_multicore || operation_attributes.use_low_perf) {
        return true;
    }

    // GENERAL PREDICATE — a single tile per compute block leaves the codegen pipeline nothing to
    // overlap, so all of its extra per-core setup is unamortized.
    //
    // Wt == 1 forces the row path (uses_block_path needs Wt > 32, uses_2d_column_path needs Wt >= 2,
    // both in tilize_codegen_program_factory.cpp), and there chunk_wt == 1 with num_col_chunks == 1,
    // which drives write_batch to 1 through compute_row_shape's force_single — either via
    // `minimal_work` when total_Ht <= num_cores, or via the total_Ht > num_cores clause otherwise. So
    // on EVERY Wt == 1 program the batched writer — the one thing writer_tilize_interleaved.cpp does
    // that native's per-page writer_unary_interleaved_start_id.cpp does not — is switched off, and
    // each compute invocation tilizes one tile. What is left is codegen's per-core setup against
    // native's, which is spec.py's own reading of this shape class ("ttnn single-buffers both CBs
    // here; our double-buffer + BATCH>1 writer priming only adds fixed per-core setup with nothing to
    // overlap").
    //
    // The ledger agrees on every Wt == 1 configuration it covers — bfloat16, float32, uint32, int32
    // and uint16, DRAM and L1 outputs, total_Ht both under and over the core count — and holds no
    // Wt == 1 counterexample, so this is a condition, not an enumerated case.
    if (operation_attributes.Wt == 1) {
        return true;
    }

    // GENERAL PREDICATE, arch-calibrated — 2D-column stream concurrency against strip width.
    // The column reader issues, per core, 32 strided reads of strip_bytes at row-pitch stride;
    // once concurrent strided streams outgrow the strip width, native's contiguous full-row
    // streams win. The crossover is arch-specific because the two sides scale differently across
    // arches: on blackhole native's contiguous reads run ~2x faster than on wormhole while the
    // column reader's strided narrow streams do not (an identical column program — same ncol,
    // same core count — still measures 1.2-1.3x slower on BH), so one configuration can be a win
    // on WH and a loss on BH. Thresholds are measured on the cross-arch sweep suite, not derived:
    // on BH, cores * 2 >= strip_bytes catches every measured loser and sacrifices no win; WH
    // measures wins up to twice that concurrency, so the rule arms only where a regression was
    // measured. Scoring the dispatch's own ncol (rather than a re-derived one) keeps the rule
    // correct on harvested grids. Forced implementation=codegen never consults this gate.
    if (is_device_tensor(tensor_args.input_tensor)) {
        IDevice* device = tensor_args.input_tensor.device();
        uint32_t streams_per_64b = 0;  // 0: rule not armed on this arch
        switch (device->arch()) {
            case tt::ARCH::BLACKHOLE: streams_per_64b = 32; break;
            default: break;
        }
        if (streams_per_64b != 0) {
            const TilizeCodegenDispatch dispatch =
                tilize_codegen_dispatch(device, operation_attributes, tensor_args.input_tensor);
            if (dispatch.path == TilizeCodegenPath::Column) {
                uint32_t elt_bytes = 0;
                switch (operation_attributes.input_dtype) {
                    case DataType::BFLOAT16:
                    case DataType::UINT16: elt_bytes = 2; break;
                    case DataType::FLOAT32:
                    case DataType::UINT32:
                    case DataType::INT32: elt_bytes = 4; break;
                    default: break;
                }
                if (elt_bytes != 0) {
                    // Narrowest block of the (possibly ragged) ncol split sets the strip width.
                    const uint32_t strip_bytes =
                        (operation_attributes.Wt / dispatch.ncol) * tt::constants::TILE_WIDTH * elt_bytes;
                    const uint32_t cores = operation_attributes.NC * operation_attributes.Ht * dispatch.ncol;
                    if (cores * 64 >= streams_per_64b * strip_bytes) {
                        return true;
                    }
                }
            }
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
