// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "plusone_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::experimental::prim {

// MetalV2 base-case factory: the cache key is the default reflection hash (op type + attrs + tensor spec),
// which excludes the buffer address. create_program_spec is the immutable contract; create_invariant_run_args
// carries the per-core work geometry (W/H/stick_size); create_per_enqueue_args carries the input tensor,
// re-applied on every dispatch. plusone is in-place: the returned tensor is the input.
struct PlusOneProgramFactory {
    static tt::tt_metal::experimental::ProgramSpec create_program_spec(
        const PlusoneParams& attrs, const Tensor& input, Tensor& output);
    static tt::tt_metal::experimental::ProgramRunArgs create_invariant_run_args(
        const PlusoneParams& attrs, const Tensor& input, Tensor& output);
    static tt::tt_metal::experimental::ProgramRunArgs create_per_enqueue_args(
        const PlusoneParams& attrs,
        const Tensor& input,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
