// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <reflect>
#include <tt-metalium/graph_tracking.hpp>
#include <tracy/Tracy.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

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

template <typename T, typename... Args>
concept FirstArgIs =
    sizeof...(Args) > 0 && std::same_as<std::decay_t<std::tuple_element_t<0, std::tuple<Args&&...>>>, T>;

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

    // --- operator() Overloads ---

    // (1) Overload when the first argument is a QueueId.
    template <typename First, typename... Rest>
        requires std::same_as<std::decay_t<First>, QueueId>
    auto operator()(First&& first, Rest&&... rest) const {
        return traced_invoke(std::forward<First>(first), std::forward<Rest>(rest)...);
    }

    // (2a) Overload when no QueueId is provided AND the operation is invocable without a QueueId.
    template <typename... Args>
        requires(sizeof...(Args) == 0 || (!FirstArgIs<QueueId, Args...> && HasInvoke<operation_t, Args && ...>))
    auto operator()(Args&&... args) const {
        return traced_invoke(std::forward<Args>(args)...);
    }

    // (2b) Overload when no QueueId is provided but the operation is NOT invocable without a QueueId,
    // so we inject DefaultQueueId.
    template <typename... Args>
        requires(
            sizeof...(Args) == 0 || (!FirstArgIs<QueueId, Args...> && !HasInvoke<operation_t, Args && ...> &&
                                     HasInvoke<operation_t, QueueId, Args && ...>))
    auto operator()(Args&&... args) const {
        return traced_invoke(DefaultQueueId, std::forward<Args>(args)...);
    }

private:
    template <typename... args_t>
    auto traced_invoke(args_t&&... args) const {
        log_debug(tt::LogOp, "Started C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});
        tt::tt_metal::GraphTracker::instance().track_function_start(cpp_fully_qualified_name, args...);

        auto output = invoke(std::forward<args_t>(args)...);

        tt::tt_metal::GraphTracker::instance().track_function_end(output);
        log_debug(tt::LogOp, "Finished C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});
        return output;
    }

    template <typename... args_t>
        requires PrimitiveOperationConcept<operation_t>
    auto invoke(QueueId queue_id, args_t&&... args) const {
        static_assert(
            requires { operation_t::invoke(std::forward<decltype(args)>(args)...); },
            "Primitive Operation must implement invoke() method to be invoked.");
        ZoneScopedN("Run primitive ttnn operation");
        ZoneName(static_cast<const char*>(cpp_fully_qualified_name.data), cpp_fully_qualified_name.size());
        auto [operation_attributes, tensors_args] = operation_t::invoke(std::forward<decltype(args)>(args)...);
        return ttnn::device_operation::detail::invoke<operation_t>(queue_id, operation_attributes, tensors_args);
    }

    template <typename... args_t>
        requires(PrimitiveOperationConcept<operation_t>)
    auto invoke(args_t&&... args) const {
        return invoke(DefaultQueueId, std::forward<args_t>(args)...);
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
