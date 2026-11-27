// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

// Whether a SliceTileProgramFactory program can serve another row count: interleaved tiles, unit
// steps, bounds given as attributes and the full grid. Everything that follows the shapes and bounds
// is then a runtime argument, so the program hash leaves the input's rows and the row bounds out.
bool slice_tile_program_is_row_free(const SliceParams& args, const SliceInputs& tensor_args);

// Whether a row-free slice has at least one output tile per core. Its core set is then the whole grid
// at every row count, and the hash leaves the output's rows out too. With fewer tiles the core set
// follows the output's tile count, so the hash keeps the whole output spec.
bool slice_tile_program_fills_grid(const SliceParams& args, const SliceInputs& tensor_args);

struct SliceTileProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    struct shared_variables_t {
        // Keep the named scalar assignments, not pointers into enqueue/trace runtime storage.
        tt::tt_metal::ProgramRunArgs run_args;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program, const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim
