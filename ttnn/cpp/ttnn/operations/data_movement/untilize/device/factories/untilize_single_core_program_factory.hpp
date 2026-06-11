// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation_types.hpp"
#include "ttnn/metal2_artifacts.hpp"
namespace ttnn::prim {

// Metal 2.0 untilize single-core factory — the degenerate ProgramSpecFactoryConcept. Every run-arg is a
// pure function of the input/output layout (no per-call dynamic scalar like a seed), so there is nothing
// to vary between two dispatches that share a cache entry. create_program_artifacts returns the spec PLUS
// all run-args; on a cache hit the framework just refreshes the tensor bindings (UpdateTensorArgs). No
// extract_immutable_info, no create_per_enqueue_args, no custom hash.
struct UntilizeSingleCoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const UntilizeOperationAttributes& operation_attributes,
        const UntilizeTensorArgs& tensor_args,
        UntilizeTensorReturnValue& tensor_return_value);
};
}  // namespace ttnn::prim
