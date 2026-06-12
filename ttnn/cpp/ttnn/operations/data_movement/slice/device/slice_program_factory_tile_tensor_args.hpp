// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim {

// Metal 2.0 slice tile factory where the slice start/end come from device TENSORS.
// The immutable contract (create_program_spec) plus the enqueue-invariant per-core
// work geometry (create_invariant_run_args, a pure function of shape + slice params,
// all in the cache key) and the per-dispatch tensor bindings (create_per_enqueue_args:
// src, starts, ends, dst). The legacy descriptor for fusion/pybind/mesh_partition
// lives separately.
struct SliceTileTensorArgsProgramFactory {
    static tt::tt_metal::experimental::ProgramSpec create_program_spec(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
    static tt::tt_metal::experimental::ProgramRunArgs create_invariant_run_args(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
    static tt::tt_metal::experimental::ProgramRunArgs create_per_enqueue_args(
        const SliceParams& args,
        const SliceInputs& tensor_args,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::prim
