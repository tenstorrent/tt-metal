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

// template <typename object_t, typename T>
// object_t vector_to_object(const std::vector<T>& vector) {
//     if ()
// }

// Generic wrapper that adapts any device operation to IDeviceOperation
template <typename operation_t>
    requires ttnn::device_operation::DeviceOperationConcept<operation_t>
class LazyDeviceOperation : public IDeviceOperation {
public:
    using operation_attributes_t = typename operation_t::operation_attributes_t;
    using tensor_args_t = typename operation_t::tensor_args_t;
    using tensor_return_value_t = typename operation_t::tensor_return_value_t;

    LazyDeviceOperation(operation_attributes_t attributes, const tensor_args_t& tensor_args) :
        attributes_(std::move(attributes)) {
        // Compute and cache output specs once at construction time
        // This avoids needing to reconstruct tensor_args later
        if constexpr (requires { operation_t::compute_output_specs(attributes_, tensor_args); }) {
            auto spec = operation_t::compute_output_specs(attributes_, tensor_args);
            cached_output_specs_ =
                ttnn::experimental::jit::object_to_vector<typename operation_t::spec_return_value_t, ttnn::TensorSpec>(
                    spec);
        }
    }

    void validate(const std::vector<Tensor>& input_tensors) const override {
        // For now, skip validation in lazy mode
        // Full validation will happen when the operation is actually executed
        // This avoids issues with reconstructing tensor_args from stored tensors
    }

    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const override {
        // Return cached output specs that were computed at construction time
        return cached_output_specs_;
    }

private:
    operation_attributes_t attributes_;
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

public:
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const override {
        auto tensor_args = reconstruct_tensor_args(input_tensors);

        // Update tensor_args with output tensor if present
        if constexpr (requires { tensor_args.output_tensor; }) {
            if (!output_tensors.empty()) {
                tensor_args.output_tensor = output_tensors[0];
            }
        }

        // Select program factory and create program
        auto program_factory = operation_t::select_program_factory(attributes_, tensor_args);

        // Create the output tensor return value
        tensor_return_value_t tensor_return_value;
        if constexpr (std::same_as<tensor_return_value_t, Tensor>) {
            tensor_return_value = output_tensors[0];
        } else if constexpr (std::same_as<tensor_return_value_t, std::vector<Tensor>>) {
            tensor_return_value = output_tensors;
        } else {
            TT_THROW("Unsupported tensor_return_value_t type");
        }

        // Visit the program factory variant and call create
        // We need to convert from CachedProgram (program + shared_variables) to ProgramWithCallbacks (program +
        // callback)
        return std::visit(
            [&](auto&& factory) -> tt::tt_metal::operation::ProgramWithCallbacks {
                using factory_t = std::decay_t<decltype(factory)>;
                using shared_variables_t = typename factory_t::shared_variables_t;

                auto cached_program = factory_t::create(attributes_, tensor_args, tensor_return_value);

                // Extract program and shared variables
                // Note: We need to move these into the callback since CachedProgram holds references
                tt::tt_metal::Program program = std::move(cached_program.program);
                shared_variables_t shared_vars = std::move(cached_program.shared_variables);

                // Create a callback that captures the shared variables and uses them for runtime argument updates
                // The callback signature matches OverrideRuntimeArgumentsCallback
                tt::tt_metal::operation::OverrideRuntimeArgumentsCallback<std::vector<Tensor>> callback =
                    [attributes = this->attributes_, shared_vars = std::move(shared_vars)](
                        const void* operation,
                        tt::tt_metal::Program& program,
                        const std::vector<Tensor>& input_tensors,
                        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                        const std::vector<Tensor>& output_tensors) mutable {
                        // Reconstruct tensor_args for override_runtime_arguments
                        tensor_args_t tensor_args =
                            build_tensor_args_for_callback<tensor_args_t>(input_tensors, optional_input_tensors);

                        // Create tensor_return_value from output_tensors
                        tensor_return_value_t tensor_return_value;
                        if constexpr (std::same_as<tensor_return_value_t, Tensor>) {
                            tensor_return_value = const_cast<Tensor&>(output_tensors[0]);
                        } else if constexpr (std::same_as<tensor_return_value_t, std::vector<Tensor>>) {
                            tensor_return_value = const_cast<std::vector<Tensor>&>(output_tensors);
                        }

                        // Create a CachedProgram proxy to pass to override_runtime_arguments
                        auto cached_program_proxy = factory_t::cached_program_t::proxy(program, shared_vars);

                        // Call the factory's override_runtime_arguments
                        factory_t::override_runtime_arguments(
                            cached_program_proxy, attributes, tensor_args, tensor_return_value);
                    };

                return {std::move(program), callback};
            },
            program_factory);
    }

    // Helper to build tensor_args for the callback
    template <typename T = tensor_args_t>
    static T build_tensor_args_for_callback(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) {
        TT_FATAL(!input_tensors.empty(), "Expected at least one input tensor");

        // Try common patterns
        // Pattern 1: Binary operation with optional: (const Tensor&, const Tensor&, std::optional<Tensor>)
        if constexpr (std::is_constructible_v<T, const Tensor&, const Tensor&, std::optional<Tensor>>) {
            if (input_tensors.size() >= 2) {
                return T{input_tensors[0], input_tensors[1], std::nullopt};
            }
            // Fallthrough to unary patterns if we don't have 2 tensors
        }
        // Pattern 2: Binary operation: (const Tensor&, const Tensor&)
        if constexpr (std::is_constructible_v<T, const Tensor&, const Tensor&>) {
            if (input_tensors.size() >= 2) {
                return T{input_tensors[0], input_tensors[1]};
            }
            // Fallthrough to unary patterns if we don't have 2 tensors
        }
        // Pattern 3: Unary operation with optional output: (const Tensor&, std::optional<Tensor>)
        if constexpr (std::is_constructible_v<T, const Tensor&, std::optional<Tensor>>) {
            return T{input_tensors[0], std::nullopt};
        }
        // Pattern 4: Simple unary: (const Tensor&)
        if constexpr (std::is_constructible_v<T, const Tensor&>) {
            return T{input_tensors[0]};
        }

        TT_THROW("Cannot reconstruct tensor_args_t for callback");
    }

    std::vector<Tensor> invoke(std::vector<Tensor> input_tensors) override {
        return tt::tt_metal::operation::run(*this, input_tensors);
    }

    const operation_attributes_t& get_attributes() const { return attributes_; }

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

private:
    template <typename T = tensor_args_t>
    T reconstruct_tensor_args(const std::vector<Tensor>& input_tensors) const {
        // We need to manually reconstruct the tensor_args based on the structure
        // This is a simplified version - in practice, you might need more sophisticated logic

        // For now, we'll use a placeholder that works for simple cases
        // You may need to specialize this for complex tensor_args_t structures

        if constexpr (requires { typename operation_t::tensor_args_t; }) {
            // Try to use reflection to build tensor_args
            return build_tensor_args_with_reflection<T>(input_tensors);
        } else {
            TT_THROW("Cannot reconstruct tensor_args_t");
        }
    }

    // Use reflection to build tensor_args from input tensors
    // Class concept guarantees this is only called for reconstructible types
    template <typename T = tensor_args_t>
    T build_tensor_args_with_reflection(const std::vector<Tensor>& input_tensors) const {
        // This is a simplified reconstruction that works for common patterns
        // For operations with complex tensor_args_t, we'll need custom handling

        TT_FATAL(!input_tensors.empty(), "Expected at least one input tensor");

        // Try common patterns
        // Pattern 1: Binary operation with two optionals
        if constexpr (
            std::is_constructible_v<tensor_args_t, const Tensor&, std::optional<Tensor>, std::optional<Tensor>>) {
            if (input_tensors.size() >= 2) {
                return tensor_args_t{input_tensors[0], input_tensors[1], std::nullopt};
            } else if (input_tensors.size() == 1) {
                return tensor_args_t{input_tensors[0], std::nullopt, std::nullopt};
            } else {
                TT_THROW("Binary operation requires at least 2 input tensors");
            }
        }
        // Pattern 2: Unary operation with optional output: (const Tensor&, std::optional<Tensor>)
        else if constexpr (std::is_constructible_v<tensor_args_t, const Tensor&, std::optional<Tensor>>) {
            return tensor_args_t{input_tensors[0], std::nullopt};
        }
        // For any other pattern, throw an error
        // This will cause the operation to fail gracefully and not be made lazy
        else {
            TT_THROW("Currently support reconstructing tensor_args_t for unary/binary ops");
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
