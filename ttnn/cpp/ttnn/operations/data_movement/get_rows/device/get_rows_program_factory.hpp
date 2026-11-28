// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"

#include "get_rows_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Interleaved row-major gather: each core reads its contiguous slice of the index vector and stages
// the selected input rows, in order, into the output. A sharded-output factory and a chunked path
// for rows wider than the staging budget are future work (see #15942).
struct GetRowsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const GetRowsParams& operation_attributes, const GetRowsInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
