// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>

namespace ttnn::operations::experimental::fusion {

using fusion_dispatch_operation_attributes_t = tt::tt_metal::experimental::MeshProgramDescriptor;
using fusion_dispatch_tensor_return_value_t = Tensor;
using fusion_dispatch_spec_return_value_t = TensorSpec;

struct fusion_dispatch_tensor_args_t {
    const std::vector<Tensor>& io_tensors;
    const Tensor& output_tensor;
};

}  // namespace ttnn::operations::experimental::fusion
