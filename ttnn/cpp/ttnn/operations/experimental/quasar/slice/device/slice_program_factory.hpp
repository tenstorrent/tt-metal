// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// The one Quasar slice program: a multi-threaded data-movement kernel per cluster that moves output pages
// (ROW_MAJOR rows, TILE tiles) through a per-thread scratchpad region. See slice_program_factory.cpp.
struct SliceProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::prim::qsr
