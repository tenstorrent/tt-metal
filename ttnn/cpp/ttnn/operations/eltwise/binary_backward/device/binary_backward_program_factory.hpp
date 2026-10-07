// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

#include "binary_backward_device_operation_types.hpp"

namespace ttnn::operations::binary_backward {

struct BinaryBackwardProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const BinaryBackwardParams& args, const BinaryBackwardInputs& tensor_args, std::vector<Tensor>& outputs);
};

}  // namespace ttnn::operations::binary_backward
