// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <reflect>
#include <tt-metalium/graph_tracking.hpp>
#include <tracy/Tracy.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/lazy_mode.hpp"
#include "ttnn/experimental/jit/lazy_device_operation.hpp"
#include "ttnn/experimental/jit/context.hpp"

namespace ttnn {
namespace decorators {

using Tensors = tt::tt_metal::operation::Tensors;
using OptionalTensors = tt::tt_metal::operation::OptionalTensors;
using OptionalConstTensors = tt::tt_metal::operation::OptionalConstTensors;

namespace detail {

// Get "add" from "ttnn::add"
static std::string base_name(const std::string& cpp_fully_qualified_name) {
    auto last_token = cpp_fully_qualified_name.substr(cpp_fully_qualified_name.rfind("::") + 2);
    return last_token;
}

// Convert "ttnn::add" to "add_t"
inline std::string class_name(const std::string& cpp_fully_qualified_name) {
    return base_name(cpp_fully_qualified_name) + "_t";
}

// Convert "ttnn::add" to "ttnn.add"
inline std::string python_fully_qualified_name(const std::string& cpp_fully_qualified_name) {
    auto replace = [](const std::string& input, const std::string& from, const std::string& to) {
        if (from.empty()) {
            return input;
        }
        auto output = input;
        size_t start = 0;
        while ((start = output.find(from, start)) != std::string::npos) {
            output.replace(start, from.length(), to);
            start += to.length();  // In case 'to' contains 'from', like replacing 'x' with 'yx'
        };
        return output;
    };
    return replace(cpp_fully_qualified_name, "::", ".");
}

}  // namespace detail

// Primitive operations map directly to device operations
template <typename operation_t>
concept PrimitiveOperationConcept = device_operation::DeviceOperationConcept<operation_t>;

// Composite operation allows any code to be executed
template <typename operation_t>
concept CompositeOperationConcept = !PrimitiveOperationConcept<operation_t>;

template <typename Op, typename... Args>
concept HasInvoke = requires {
    { Op::invoke(std::declval<Args>()...) };
};

namespace detail {

// Concept to check if an operation can be reconstructed from input tensors
// This checks the patterns that LazyDeviceOperation::build_tensor_args_with_reflection supports
template <typename operation_t>
concept CanReconstructTensorArgs =
    requires { typename operation_t::tensor_args_t; } &&
    (std::is_constructible_v<typename operation_t::tensor_args_t, const Tensor&, std::optional<Tensor>> ||
     std::is_constructible_v<typename operation_t::tensor_args_t, const Tensor&>);

// Concept to check if compute_output_specs is available
template <typename operation_t>
concept HasComputeOutputSpecs = requires(
    const typename operation_t::operation_attributes_t& attrs, const typename operation_t::tensor_args_t& tensor_args) {
    { operation_t::compute_output_specs(attrs, tensor_args) };
};

// Concept to check if return type is supported for lazy execution
template <typename operation_t>
concept HasSupportedLazyReturnType = std::same_as<typename operation_t::tensor_return_value_t, Tensor> ||
                                     std::same_as<typename operation_t::tensor_return_value_t, std::vector<Tensor>>;

// Helper to check if a program factory has the standard create() method
template <typename factory_t, typename operation_t>
concept HasStandardCreateMethod = requires(
    const typename operation_t::operation_attributes_t& attrs,
    const typename operation_t::tensor_args_t& tensor_args,
    typename operation_t::tensor_return_value_t& tensor_return_value) {
    { factory_t::create(attrs, tensor_args, tensor_return_value) };
};

// Concept to check if program factories support standard create() method
// This excludes mesh workload factories that use create_at() or create_mesh_workload()
template <typename operation_t>
concept HasStandardProgramFactories =
    requires { typename operation_t::program_factory_t; } && []<typename... Ts>(std::variant<Ts...>*) {
        return (HasStandardCreateMethod<Ts, operation_t> && ...);
    }(static_cast<typename operation_t::program_factory_t*>(nullptr));

// Main concept: can this operation be made lazy?
template <typename operation_t>
concept CanBeMadeLazy = PrimitiveOperationConcept<operation_t> && CanReconstructTensorArgs<operation_t> &&
                        HasComputeOutputSpecs<operation_t> && HasSupportedLazyReturnType<operation_t> &&
                        HasStandardProgramFactories<operation_t>;

}  // namespace detail

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
struct registered_operation_t {
    static constexpr auto is_primitive = PrimitiveOperationConcept<operation_t>;

    // Get "add" from "ttnn::add"
    std::string base_name() const { return detail::base_name(std::string{cpp_fully_qualified_name}); }

    // Convert "ttnn::add" to "add_t"
    std::string class_name() const { return detail::class_name(std::string{cpp_fully_qualified_name}); }

    // Convert "ttnn::add" to "ttnn.add"
    std::string python_fully_qualified_name() const {
        return detail::python_fully_qualified_name(std::string{cpp_fully_qualified_name});
    }

    template <typename... Args>
        requires(HasInvoke<operation_t, Args && ...>)
    auto operator()(Args&&... args) const {
        return traced_invoke(std::forward<Args>(args)...);
    }

private:
    template <typename... args_t>
    auto traced_invoke(args_t&&... args) const {
        log_debug(tt::LogOp, "Started C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});
        tt::tt_metal::GraphTracker::instance().track_function_start(cpp_fully_qualified_name, args...);

        auto output = invoke(std::forward<args_t>(args)...);

        tt::tt_metal::GraphTracker::instance().track_function_end(output);
        log_debug(tt::LogOp, "Finished invoking C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});
        return output;
    }

    template <typename... args_t>
        requires PrimitiveOperationConcept<operation_t>
    auto invoke(args_t&&... args) const {
        static_assert(
            requires { operation_t::invoke(std::forward<decltype(args)>(args)...); },
            "Primitive Operation must implement invoke() method to be invoked.");
        ZoneScopedN("Run primitive ttnn operation");
        ZoneName(static_cast<const char*>(cpp_fully_qualified_name.data), cpp_fully_qualified_name.size());
        auto [operation_attributes, tensors_args] = operation_t::invoke(std::forward<decltype(args)>(args)...);

        // Check if lazy mode is enabled
        if (ttnn::lazy_mode::is_lazy_enabled()) {
            return invoke_lazy(operation_attributes, tensors_args);
        }

        return ttnn::device_operation::detail::invoke<operation_t>(operation_attributes, tensors_args);
    }

    template <typename operation_attributes_t, typename tensor_args_t>
    auto invoke_lazy(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) const {
        using tensor_return_value_t = typename operation_t::tensor_return_value_t;

        if constexpr (detail::CanBeMadeLazy<operation_t>) {
            // Extract input tensors from tensor_args
            std::vector<Tensor> input_tensors =
                ttnn::experimental::jit::object_to_vector<tensor_args_t, Tensor>(tensor_args);

            // TODO: Do I need that?
            // Filter out output tensors if present
            // For most operations, output_tensor is std::optional and should be excluded
            if constexpr (requires { tensor_args.output_tensor; }) {
                if (tensor_args.output_tensor.has_value()) {
                    // Remove the output tensor from inputs (it's typically at the end)
                    if (!input_tensors.empty()) {
                        input_tensors.pop_back();
                    }
                }
            }

            // Create lazy operation wrapper
            auto lazy_op =
                ttnn::experimental::jit::make_lazy_device_operation<operation_t>(operation_attributes, tensor_args);

            // Get output specs to create placeholder tensors
            auto output_specs = lazy_op->compute_output_specs(input_tensors);

            // Add operation to context
            auto& context = ttnn::experimental::jit::Context::instance();
            auto node_id = context.create_node(
                input_tensors,
                std::string(cpp_fully_qualified_name.data, cpp_fully_qualified_name.size()),
                std::move(lazy_op));

            // Create placeholder output tensors
            auto first_input_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
            auto device = first_input_tensor.device();

            // TODO: Move this if constexprs to a separate function
            if constexpr (std::same_as<tensor_return_value_t, Tensor>) {
                // Single tensor output
                TT_FATAL(!output_specs.empty(), "Expected at least one output spec");
                auto output_tensor = tt::tt_metal::create_device_tensor(output_specs[0], device);
                output_tensor = tt::tt_metal::set_tensor_id(output_tensor);
                output_tensor.set_producer_node(node_id);
                return output_tensor;
            } else if constexpr (std::same_as<tensor_return_value_t, std::vector<Tensor>>) {
                // Multiple tensor outputs (vector)
                std::vector<Tensor> output_tensors;
                for (const auto& spec : output_specs) {
                    auto output_tensor = tt::tt_metal::create_device_tensor(spec, device);
                    output_tensor = tt::tt_metal::set_tensor_id(output_tensor);
                    output_tensor.set_producer_node(node_id);
                    output_tensors.push_back(output_tensor);
                }
                return output_tensors;
            }
        } else {
            // Fallback for other return types - execute eagerly
            log_warning(
                tt::LogOp,
                "Lazy mode not supported for operation {} with return type {}, falling back to eager execution",
                std::string{cpp_fully_qualified_name},
                tt::stl::get_type_name<tensor_return_value_t>());
            return ttnn::device_operation::detail::invoke<operation_t>(operation_attributes, tensor_args);
        }
    }

    template <typename... args_t>
        requires(CompositeOperationConcept<operation_t>)
    auto invoke(args_t&&... args) const {
        return invoke_composite(std::forward<args_t>(args)...);
    }

    template <typename... args_t>
    auto invoke_composite(args_t&&... args) const {
        ZoneScopedN("Run composite ttnn operation ");
        ZoneName(static_cast<const char*>(cpp_fully_qualified_name.data), cpp_fully_qualified_name.size());
        return operation_t::invoke(std::forward<decltype(args)>(args)...);
    }
};

template <reflect::fixed_string cpp_fully_qualified_name>
struct operation_name_key_t {
    friend consteval auto get(operation_name_key_t<cpp_fully_qualified_name>);
};

template <typename operation_t>
struct operation_key_t {
    friend consteval auto get(operation_key_t<operation_t>);
};

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t, auto operation>
struct set_operation_t : std::true_type {
    friend consteval auto get(operation_key_t<operation_t>) { return operation; }
    friend consteval auto get(operation_name_key_t<cpp_fully_qualified_name>) { return operation; }
};

constexpr reflect::fixed_string prim_namespace = "ttnn::prim";

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
consteval void assert_operation_in_correct_namespace() {
    if constexpr (PrimitiveOperationConcept<operation_t>) {
        if constexpr (cpp_fully_qualified_name.size() > prim_namespace.size()) {
            constexpr auto namespace_substring =
                tt::stl::reflection::fixed_string_substring<0, prim_namespace.size()>(cpp_fully_qualified_name);
            static_assert(
                tt::stl::reflection::fixed_string_equals(namespace_substring, prim_namespace),
                "Primitive operations must be in the `ttnn::prim` namespace.");
        } else {
#ifndef DISABLE_NAMESPACE_STATIC_ASSERT
            static_assert(false, "Primitive operations must be in the `ttnn::prim` namespace.");
#endif
        }
    } else {
        if constexpr (cpp_fully_qualified_name.size() > prim_namespace.size()) {
            constexpr auto namespace_substring =
                tt::stl::reflection::fixed_string_substring<0, prim_namespace.size()>(cpp_fully_qualified_name);
            static_assert(
                not tt::stl::reflection::fixed_string_equals(namespace_substring, prim_namespace),
                "Composite operations must not be in the `ttnn::prim` namespace. You may have forgotten to implement "
                "one of: validate_on_program_cache_hit, validate_on_program_cache_miss, create_output_tensors, or "
                "select_program_factory.");
        }
    }
}

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
constexpr auto register_operation_impl() {
    assert_operation_in_correct_namespace<cpp_fully_qualified_name, operation_t>();
    constexpr auto operation = registered_operation_t<cpp_fully_qualified_name, operation_t>{};
    static_assert(
        not requires(operation_name_key_t<cpp_fully_qualified_name> key) { get(key); },
        "Operation with this `cpp_fully_qualified_name` was already registered. Please use a different name.");
    static_assert(
        not requires(operation_key_t<operation_t> key) { get(key); },
        "Operation with this `operation_t` was already registered. Please use a different type.");
    static_assert(set_operation_t<cpp_fully_qualified_name, operation_t, operation>::value);
    return operation;
}

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
constexpr auto register_operation() {
    return register_operation_impl<cpp_fully_qualified_name, operation_t>();
}

}  // namespace decorators

using ttnn::decorators::register_operation;

}  // namespace ttnn
