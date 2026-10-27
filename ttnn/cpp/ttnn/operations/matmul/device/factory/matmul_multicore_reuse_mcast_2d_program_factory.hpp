// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Runs on as much of the config's grid as the problem needs, from its allowed_worker_cores corner. Inside a
// worker-cores scope the prim sizes that grid from the worker cores and puts it at their corner
// (normalize_on_worker_cores), so the factory takes them without reading them.
struct MatmulMultiCoreReuseMcast2DProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const tt::tt_metal::CoreRangeSet& worker_cores);
};

}  // namespace ttnn::prim
