// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"

#include "unary_backward_device_operation_types.hpp"

namespace ttnn::operations::unary_backward {

// One factory for every op in UnaryBackwardOpType: the reader, writer, circular buffers and
// work split are identical across unary gradients, and the only per-op input is the compute
// kernel named by get_kernel_spec().
struct UnaryBackwardProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UnaryBackwardParams& args, const UnaryBackwardInputs& tensor_args, Tensor& output);
};

}  // namespace ttnn::operations::unary_backward
