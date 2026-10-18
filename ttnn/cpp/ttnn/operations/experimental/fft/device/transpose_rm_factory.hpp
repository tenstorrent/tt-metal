// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TransposeRmFactory — ProgramSpec factory for
// transpose_rm.  Tile-based (32×32) multi-core inner-axis transpose for
// ROW_MAJOR fp32/bf16 tensors.  See header of the kernels for the
// per-unit DMA pattern.

#pragma once

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "transpose_rm_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct TransposeRmFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const TransposeRmParams& operation_attributes,
        const TransposeRmTensorArgs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
