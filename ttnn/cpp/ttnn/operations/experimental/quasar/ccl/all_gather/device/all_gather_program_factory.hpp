// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"
#include "all_gather_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

// Quasar-native (Metal 2.0 ProgramSpec) all-gather: one WorkUnit over every worker cluster, a
// multi-thread reader KernelSpec producing a packet DFB that a multi-thread writer KernelSpec drains.
struct AllGatherProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const AllGatherOperationAttributes& operation_attributes,
        const AllGatherTensorArgs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
