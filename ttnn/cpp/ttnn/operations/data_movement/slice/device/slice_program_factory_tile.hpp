// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

struct SliceTileProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

    // Cache-hit hook: the tensor bindings plus the per-core scalars, which are hash-excluded.
    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// Per-core scalars are hash-excluded; a divergent-partition cache hit leaves them stale -> all-zero output (#52651).
// Returns the reader and writer KernelRunArgs entries carrying those scalars, for the named kernels.
tt::tt_metal::experimental::Group<tt::tt_metal::experimental::KernelRunArgs> slice_tile_run_args(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    const Tensor& output,
    uint32_t start_offset,
    const tt::tt_metal::experimental::KernelSpecName& reader_kernel,
    const tt::tt_metal::experimental::KernelSpecName& writer_kernel);

}  // namespace ttnn::prim
