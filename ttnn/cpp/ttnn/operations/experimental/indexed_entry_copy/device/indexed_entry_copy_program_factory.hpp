// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "indexed_entry_copy_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct IndexedEntryCopyProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const IndexedEntryCopyParams& params, const IndexedEntryCopyInputs& inputs, Tensor& output);
};

}  // namespace ttnn::experimental::prim
