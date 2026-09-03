// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "graph_kernel_device_operation_types.hpp"
#include "graph_kernel_program_factory.hpp"

namespace ttnn::experimental::prim {

struct GraphKernelDeviceOperation {
    using operation_attributes_t = GraphKernelParams;
    using tensor_args_t = GraphKernelInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<GraphKernelProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor graph_kernel(const std::vector<Tensor>& inputs, const std::string& text);
}  // namespace ttnn::prim
