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

// Metal 2.0 sharded row-major slice factory. The immutable contract (create_program_spec)
// plus the enqueue-invariant per-core work geometry (create_invariant_run_args — a pure
// function of shape + shard spec + slice params, all in the cache key) and the per-dispatch
// tensor bindings (create_per_enqueue_args).
//
// Both circular buffers are SHARDED — in the descriptor era CBDescriptor::buffer was pinned to
// input.buffer()/output.buffer(). In Metal 2.0 those become BORROWED-MEMORY DataflowBuffers:
// DataflowBufferSpec::borrowed_from = TensorParamName, with the backing L1 address supplied per
// dispatch via the tensor argument. The single data-movement reader does shard-to-shard NOC
// reads (it is a DM kernel, so binding the sharded tensors to it is allowed) and writes directly
// into the borrowed output CB; there is no separate writer kernel (same as the legacy factory).
struct SliceRmShardedProgramFactory {
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
