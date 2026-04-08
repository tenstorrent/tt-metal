// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>

namespace ttnn::operations::generic {

using operation_attributes_t = tt::tt_metal::experimental::MeshProgramDescriptor;
using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

// NOTE: output tensor is the last element in the vector io_tensors
struct tensor_args_t {
    const std::vector<Tensor>& io_tensors;
    const Tensor& output_tensor;
};

}  // namespace ttnn::operations::generic
