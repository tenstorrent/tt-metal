// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rand_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::operations::rand {

// Metal 2.0 (MetalV2FactoryConcept) factory for rand.
//
// seed/from/to are excluded from the program-cache key (see RandOperationAttributes::attribute_values)
// and are therefore per-enqueue: create_program_run_args re-derives them from the live attributes and
// the framework re-applies them via SetProgramRunArgs on every cache hit. That is the Metal 2.0-native
// replacement for the descriptor framework's get_dynamic_runtime_args — no positional (kernel, arg)
// indices, and SetProgramRunArgs validates that every named runtime arg in the ProgramSpec is set.
struct RandProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RandOperationAttributes& operation_attributes, const RandTensorArgs& tensor_args, Tensor& output);

    static tt::tt_metal::experimental::ProgramRunArgs create_program_run_args(
        const RandOperationAttributes& operation_attributes, const RandTensorArgs& tensor_args, Tensor& output);
};

}  // namespace ttnn::operations::rand
