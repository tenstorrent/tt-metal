// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"

#include <optional>

#include <tt-metalium/workload/run_args/program_run_args.hpp>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::prim {

// Whether ConcatTiledRowShiftProgramFactory can run this concat: a TILE concat of interleaved
// bfloat16 or float32 tensors along the rows (dim -2), with 32x32 tiles, into an interleaved output, on
// the full grid. It also takes inputs whose rows are all whole tiles, which it copies as whole tiles, so
// that one program serves a concat whether or not its split falls on a tile boundary at a given size.
bool can_use_tiled_row_shift_concat(
    const std::vector<Tensor>& input_tensors,
    uint32_t normalized_dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

// Concat along the rows of TILE tensors whose row counts need not be whole tiles. A reader builds each
// output tile from the source rows it holds, shifting the rows of an input that starts inside a tile, and
// zero-fills the rows past the last input's; the shared tile writer drains them. Every count that follows
// the rows is a runtime argument, and the cores are the whole grid at every count, so the program hash
// leaves the rows out and one program serves every row count.
struct ConcatTiledRowShiftProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value);

    // Rewrites the per-core tile ranges and the row counts, and rebinds the tensors.
    static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
        const ConcatParams& operation_attributes,
        const ConcatInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim
