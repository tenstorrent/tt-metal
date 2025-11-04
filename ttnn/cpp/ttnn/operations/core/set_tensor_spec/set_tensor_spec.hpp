// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::core {

// SetTensorSpecOperation updates the tensor metadata (TensorSpec) while keeping the same underlying storage.
// This is useful for operations like view/reshape that don't change the actual data but only the tensor shape.
struct SetTensorSpecOperation {
    using tensor_return_value_t = Tensor;
    using spec_return_value_t = TensorSpec;

    SetTensorSpecOperation(const Tensor& input_tensor, const TensorSpec& new_tensor_spec) :
        new_tensor_spec(new_tensor_spec) {}

    static std::tuple<Tensors, OptionalTensors, OptionalTensors> get_tensor_inputs(
        const Tensor& input_tensor, const TensorSpec& new_tensor_spec) {
        return {{input_tensor}, {}, {}};
    }

    spec_return_value_t compute_output_specs(
        const Tensors& input_tensors,
        const OptionalTensors& optional_input_tensors,
        OptionalTensors& optional_output_tensors) const;

    void validate(
        const Tensors& input_tensors,
        const OptionalTensors& optional_input_tensors,
        OptionalTensors& optional_output_tensors) const;

    tensor_return_value_t invoke(
        const Tensors& input_tensors,
        const OptionalTensors& optional_input_tensors,
        OptionalTensors& optional_output_tensors) const;

    tt::stl::hash::hash_t to_hash() const { return tt::stl::hash::hash_objects_with_default_seed(new_tensor_spec); }

private:
    spec_return_value_t new_tensor_spec;
};

constexpr auto set_tensor_spec =
    ttnn::register_operation<"ttnn::core::set_tensor_spec", ttnn::operations::core::SetTensorSpecOperation>();

}  // namespace operations::core

}  // namespace ttnn
