// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Ported to the Metal 2.0 host API (create_program_artifacts / ProgramArtifacts). The legacy
// ProgramDescriptor create_descriptor entry point and its pybind-only core_range_set parameter have
// been removed (see the device-op dispatch and matmul_nanobind.cpp). The shared compute kernel is
// served from a _metal2-suffixed fork (bmm_large_block_zm_fused_bias_activation_metal2.cpp) so the
// not-yet-ported sibling factories keep using the legacy original.
struct MatmulMultiCoreReuseOptimizedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ttnn::prim::qsr::MatmulParams& operation_attributes,
        const ttnn::prim::qsr::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim::qsr
