// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "plusone_device_operation_types.hpp"
#include "ttnn/metalv2_artifacts.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::experimental::prim {

// MetalV2 base-case factory: the cache key is the default reflection hash (op type + attrs + tensor spec),
// which excludes the buffer address. create_program_spec carries the enqueue-invariant work geometry plus
// the input tensor binding in run_args; create_per_enqueue_args opts out (nullopt), so the only per-dispatch
// work is re-applying the tensor address via UpdateTensorArgs on a cache hit. plusone is in-place: the
// returned tensor is the input.
struct PlusOneProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const PlusoneParams& attrs, const Tensor& input, Tensor& output);
    static std::optional<tt::tt_metal::experimental::ProgramRunArgs> create_per_enqueue_args(
        const PlusoneParams& attrs,
        const Tensor& input,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
