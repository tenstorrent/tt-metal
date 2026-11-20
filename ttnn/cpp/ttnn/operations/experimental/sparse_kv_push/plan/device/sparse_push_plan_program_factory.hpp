// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/distributed/types.hpp"
#include "sparse_push_plan_device_operation_types.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

struct SparsePushPlanProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SparsePushPlanParams& attrs, const SparsePushPlanInputs& args, std::vector<Tensor>& outputs);

    // Re-applies everything the program hash leaves out: the buffer addresses and the chunk start.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const SparsePushPlanParams& attrs,
        const SparsePushPlanInputs& args,
        std::vector<Tensor>& outputs,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
