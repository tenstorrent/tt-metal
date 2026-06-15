// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/sliding_window/halo/device/halo_device_operation_types.hpp"

namespace ttnn::prim {

// MetalV2 base-case factory for untilize-with-halo. The four sliding-window config tensors
// (pad_config0/1, gather_config0/1) are op-allocated internal tensors: create_owned_tensors builds them on
// host and the framework keeps them alive for the cached program's lifetime, binding each to a
// TensorParameter the reader streams from (an L1-borrowed CB, or a DRAM TensorAccessor). create_program_spec
// is the immutable blueprint; create_invariant_run_args carries the per-core work scalars; the input/output
// shard addresses ride create_per_enqueue_args.
struct UntilizeWithHaloProgramFactory {
    static tt::tt_metal::experimental::ProgramSpec create_program_spec(
        const HaloParams& attrs, const Tensor& input, Tensor& output);
    static tt::tt_metal::experimental::ProgramRunArgs create_invariant_run_args(
        const HaloParams& attrs, const Tensor& input, Tensor& output);
    static tt::tt_metal::experimental::ProgramRunArgs create_per_enqueue_args(
        const HaloParams& attrs,
        const Tensor& input,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    static tt::tt_metal::experimental::Table<tt::tt_metal::experimental::TensorParamName, Tensor>
    create_owned_tensors(const HaloParams& attrs, const Tensor& input, Tensor& output);
};

}  // namespace ttnn::prim
