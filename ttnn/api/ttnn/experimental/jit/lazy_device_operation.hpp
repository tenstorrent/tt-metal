// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/jit/IDeviceOperation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt_stl/reflection.hpp>
#include <vector>
#include <memory>

namespace ttnn::experimental::jit {

template <typename object_t, typename T>
std::vector<T> object_to_vector(const object_t& object) {
    std::vector<T> vector;
    tt::stl::reflection::visit_object_of_type<T>([&](const auto& t) { vector.push_back(t); }, object);
    return vector;
}

// Generic wrapper that adapts any device operation to IDeviceOperation
template <typename operation_t>
    requires ttnn::device_operation::DeviceOperationConcept<operation_t>
class LazyDeviceOperation : public IDeviceOperation {
public:
    using operation_attributes_t = typename operation_t::operation_attributes_t;
    using tensor_args_t = typename operation_t::tensor_args_t;
    using tensor_return_value_t = typename operation_t::tensor_return_value_t;

    LazyDeviceOperation(operation_attributes_t attributes, tensor_args_t tensor_args) :
        attributes_(std::move(attributes)), tensor_args_(std::move(tensor_args)) {
        // Compute and cache output specs once at construction time
        // Since Tensor is shallow-copyable, we can store tensor_args directly
        if constexpr (requires { operation_t::compute_output_specs(attributes_, tensor_args_); }) {
            auto spec = operation_t::compute_output_specs(attributes_, tensor_args_);
            cached_output_specs_ = convert_spec_to_vector(spec);
        }
    }

    // TODO: Can we cache validation result?
    void validate(const std::vector<Tensor>& input_tensors) const override {
        // TODO: remove unnecessary copy of tensor_args
        // Create a temporary copy of tensor_args and update it with input tensors for validation
        auto temp_tensor_args = tensor_args_;
        update_tensor_args_helper(temp_tensor_args, input_tensors);

        // Call the operation's validation method if it exists
        // Use validate_on_program_cache_miss for comprehensive validation
        if constexpr (requires { operation_t::validate_on_program_cache_miss(attributes_, temp_tensor_args); }) {
            operation_t::validate_on_program_cache_miss(attributes_, temp_tensor_args);
        }
    }

    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const override {
        // Return cached output specs that were computed at construction time
        return cached_output_specs_;
    }

    std::vector<Tensor> invoke(std::vector<Tensor> input_tensors) override {
        // Update our stored tensor_args with the new materialized tensors
        update_tensor_args_with_input_tensors(input_tensors);

        // Use the standard device operation eager execution path
        auto result = ttnn::device_operation::detail::invoke<operation_t>(attributes_, tensor_args_);

        // Convert result to vector
        return convert_result_to_vector(result);
    }

    const operation_attributes_t& get_attributes() const { return attributes_; }

    // TODO: Verify vibecoded to_hash and attributes()
    // Required for hashing support
    tt::stl::hash::hash_t to_hash() const {
        // Hash based on the operation type and attributes
        if constexpr (requires { attributes_.to_hash(); }) {
            return tt::stl::hash::hash_objects_with_default_seed(
                tt::stl::hash::type_hash<operation_t>, attributes_.to_hash());
        } else {
            return tt::stl::hash::hash_objects_with_default_seed(tt::stl::hash::type_hash<operation_t>, attributes_);
        }
    }

    // Required for reflection support
    auto attributes() const {
        if constexpr (requires { attributes_.attributes(); }) {
            return attributes_.attributes();
        } else {
            // Return minimal attributes if the operation doesn't support it
            using tt::stl::reflection::Attribute;
            std::vector<std::tuple<std::string, Attribute>> attrs;
            attrs.emplace_back("operation_type", tt::stl::get_type_name<operation_t>());
            return attrs;
        }
    }

    // Not needed for lazy execution - we use invoke() instead which calls eager path
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const override {
        TT_THROW("create_program should not be called for LazyDeviceOperation - use invoke() instead");
    }

private:
    operation_attributes_t attributes_;
    tensor_args_t tensor_args_;
    std::vector<ttnn::TensorSpec> cached_output_specs_;

    // Helper to convert any spec type to vector
    template <typename SpecType>
    std::vector<ttnn::TensorSpec> convert_spec_to_vector(const SpecType& spec) const {
        using spec_type = std::decay_t<SpecType>;

        // Handle single TensorSpec
        if constexpr (std::same_as<spec_type, ttnn::TensorSpec>) {
            return {spec};
        }
        // Handle vector<TensorSpec>
        else if constexpr (std::same_as<spec_type, std::vector<ttnn::TensorSpec>>) {
            return spec;
        } else {
            TT_THROW("Unsupported spec type");
        }
    }

    // Helper to update tensor_args with input tensors (used by both validate and invoke)
    static void update_tensor_args_helper(tensor_args_t& tensor_args, const std::vector<Tensor>& input_tensors) {
        // Use reflection to update all Tensor fields in tensor_args
        size_t tensor_index = 0;
        tt::stl::reflection::visit_object_of_type<Tensor>(
            [&](const Tensor& tensor) {
                if (tensor_index < input_tensors.size()) {
                    // Cast away const since we're modifying our own mutable tensor_args
                    const_cast<Tensor&>(tensor) = input_tensors[tensor_index];
                    tensor_index++;
                }
            },
            tensor_args);
    }

    // Update stored tensor_args with new input tensors from lazy execution
    void update_tensor_args_with_input_tensors(const std::vector<Tensor>& input_tensors) {
        update_tensor_args_helper(tensor_args_, input_tensors);
    }

    // Convert result from device operation to vector of tensors
    std::vector<Tensor> convert_result_to_vector(const tensor_return_value_t& result) const {
        if constexpr (std::same_as<tensor_return_value_t, Tensor>) {
            return {result};
        } else if constexpr (std::same_as<tensor_return_value_t, std::vector<Tensor>>) {
            return result;
        } else {
            TT_THROW("Unsupported tensor_return_value_t type");
        }
    }
};

// Helper function to create a lazy operation wrapper
template <typename operation_t>
std::shared_ptr<IDeviceOperation> make_lazy_device_operation(
    const typename operation_t::operation_attributes_t& attributes,
    const typename operation_t::tensor_args_t& tensor_args) {
    return std::make_shared<LazyDeviceOperation<operation_t>>(attributes, tensor_args);
}

}  // namespace ttnn::experimental::jit
