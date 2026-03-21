// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>

namespace ttnn::operations::experimental::generic {

// Same types as generic_op — patched_generic_op has the same interface.
using patched_operation_attributes_t = tt::tt_metal::experimental::MeshProgramDescriptor;
using patched_tensor_return_value_t = Tensor;
using patched_spec_return_value_t = TensorSpec;

struct patched_tensor_args_t {
    const std::vector<Tensor>& io_tensors;
    const Tensor& output_tensor;
};

}  // namespace ttnn::operations::experimental::generic
