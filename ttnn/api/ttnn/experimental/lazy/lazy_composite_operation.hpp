// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/lazy/lazy_operation.hpp"
#include "ttnn/experimental/lazy/lazy_utils.hpp"
#include "ttnn/device_operation.hpp"
#include <tt_stl/reflection.hpp>
#include <vector>
#include <memory>
#include <utility>
#include <boost/pfr.hpp>

namespace ttnn::experimental::lazy {

// Generic wrapper that adapts any device operation to LazyOperation
template <typename operation_t, typename... args_t>
    requires ttnn::composite_operation::LazifyableCompositeOperationConcept<operation_t, args_t...>
class LazyCompositeOperation : public LazyOperation {
public:
    using tensor_return_value_t = typename operation_t::tensor_return_value_t;

    // TODO: It's not greate to capture tensor_args_ in the op
    LazyCompositeOperation(const std::string& name, args_t&&... args) :
        name_(name), operation_(std::forward<args_t>(args)...) {}

    // TODO: Can we cache validation result?
    void validate(const Tensors& input_tensors) const {
        // TODO: this shouldn't ignore optional tensors
        OptionalTensors optional_input_tensors;
        OptionalTensors optional_output_tensors;
        return operation_.validate(input_tensors, optional_input_tensors, optional_output_tensors);
    }

    std::vector<ttnn::TensorSpec> compute_output_specs(const Tensors& input_tensors) const {
        // Return cached output specs that were computed at construction time
        // TODO: this shouldn't ignore optional tensors
        OptionalTensors optional_input_tensors;
        OptionalTensors optional_output_tensors;
        if constexpr (std::is_same_v<typename operation_t::spec_return_value_t, TensorSpec>) {
            return {operation_.compute_output_specs(input_tensors, optional_input_tensors, optional_output_tensors)};
        } else if constexpr (std::is_same_v<typename operation_t::spec_return_value_t, std::vector<TensorSpec>>) {
            return operation_.compute_output_specs(input_tensors, optional_input_tensors, optional_output_tensors);
        } else {
            static_assert(
                std::is_same_v<typename operation_t::spec_return_value_t, TensorSpec>,
                "Unsupported spec return value type");
        }
    }

    std::string_view name() const override { return std::string_view(name_.c_str()); }

    std::vector<tt::tt_metal::metal_tensor::Tensor> invoke(
        const std::vector<tt::tt_metal::metal_tensor::Tensor>& input_tensors) override {
        // Use the standard device operation eager execution path
        // TODO: Remove this conversion once we have a proper metal tensor
        std::vector<Tensor> tnn_input_tensors;
        std::transform(
            input_tensors.begin(), input_tensors.end(), std::back_inserter(tnn_input_tensors), [](const auto& tensor) {
                return Tensor(tensor);
            });
        // TODO: this shouldn't ignore optional tensors
        OptionalTensors optional_input_tensors;
        OptionalTensors optional_output_tensors;
        return convert_result_to_vector(
            operation_.invoke(tnn_input_tensors, optional_input_tensors, optional_output_tensors));
    }

    tt::stl::hash::hash_t operation_type_id() const override { return tt::stl::hash::type_hash<operation_t>; }

    // TODO: Verify to_hash
    // Required for hashing support
    tt::stl::hash::hash_t to_hash() const {
        // Hash based on the operation type and attributes
        if constexpr (requires { operation_.to_hash(); }) {
            return operation_.to_hash();
        } else {
            return tt::stl::hash::hash_objects_with_default_seed(operation_);
        }
    }

    operation_t& operation() { return operation_; }

private:
    std::string name_;
    operation_t operation_;

    // Convert result from device operation to vector of tensors
    std::vector<tt::tt_metal::metal_tensor::Tensor> convert_result_to_vector(
        const tensor_return_value_t& result) const {
        // TODO: device operations are supposed to return metal tensors, not tnn::Tensor
        if constexpr (std::same_as<tensor_return_value_t, Tensor>) {
            return {result.get_materialized_tensor()};
        } else if constexpr (std::same_as<tensor_return_value_t, std::vector<Tensor>>) {
            std::vector<tt::tt_metal::metal_tensor::Tensor> metal_tensors;
            metal_tensors.reserve(result.size());
            for (const auto& tensor : result) {
                metal_tensors.push_back(tensor.get_materialized_tensor());
            }
            return metal_tensors;
        } else {
            TT_THROW("Unsupported tensor_return_value_t type");
        }
    }
};

// Helper function to create a lazy operation wrapper
template <typename operation_t, typename... args_t>
std::shared_ptr<LazyCompositeOperation<operation_t, args_t...>> make_lazy_composite_operation(
    const std::string& name, args_t&&... args) {
    return std::make_shared<LazyCompositeOperation<operation_t, args_t...>>(name, std::forward<args_t>(args)...);
}

}  // namespace ttnn::experimental::lazy
