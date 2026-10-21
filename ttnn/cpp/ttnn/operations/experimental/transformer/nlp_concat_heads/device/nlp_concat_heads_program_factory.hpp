// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "nlp_concat_heads_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct NLPConcatHeadsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const NlpConcatHeadsParams& operation_attributes, const Tensor& input, Tensor& output);

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const NlpConcatHeadsParams& operation_attributes,
        const Tensor& input,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
