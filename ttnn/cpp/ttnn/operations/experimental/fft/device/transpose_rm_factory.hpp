// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TransposeRmFactory — ProgramDescriptor program factory for
// transpose_rm.  Tile-based (32×32) multi-core inner-axis transpose for
// ROW_MAJOR fp32/bf16 tensors.  See header of the kernels for the
// per-unit DMA pattern.

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "transpose_rm_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct TransposeRmFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TransposeRmParams& operation_attributes,
        const TransposeRmTensorArgs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
