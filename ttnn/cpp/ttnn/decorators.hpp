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

template <typename Tuple, typename T>
constexpr bool is_homogenous_tuple() {
    return []<std::size_t... Ns>(std::index_sequence<Ns...>) {
        return (std::is_same_v<T, std::tuple_element_t<Ns, Tuple>> && ...);
    }(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <typename Tuple, typename T>
constexpr Tuple make_tuple_from_vector(const std::vector<T>& vector) {
    return ([&vector]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        return std::forward_as_tuple(vector.at(Ns)...);
    }(std::make_index_sequence<std::tuple_size_v<Tuple>>{}));
}

template <typename Include, typename... args_t>
auto extract_args_to_vector(args_t&&... args) {
    std::vector<Include> result;
    auto process_arg = [&](auto&& arg) {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Include>) {
            result.push_back(arg);
        }
    };
    (process_arg(std::forward<args_t>(args)), ...);
    return result;
}

template <typename operation_t, typename execute_on_worker_thread_return_t, typename... args_t>
inline auto create_async_output_tensors(
    const Tensors& inputs, const OptionalConstTensors& optional_inputs, args_t&&... args) {
    constexpr bool custom_create_async_outputs =
        requires(const operation_t& t) { t.create_async_output_tensors(inputs, optional_inputs); };

    if constexpr (custom_create_async_outputs) {
        return operation_t::create_async_output_tensors(inputs, optional_inputs);
    } else if constexpr (std::is_same_v<std::decay_t<execute_on_worker_thread_return_t>, OptionalTensors>) {
        constexpr bool custom_create_async_optional_outputs = requires(const operation_t& t) {
            t.create_async_optional_output_tensors(std::forward<decltype(args)>(args)...);
        };
        static_assert(
            custom_create_async_optional_outputs,
            "If the operation returns a vector of optional Tensors, it must "
            "implement create_async_optional_output_tensors.");

        return operation_t::create_async_optional_output_tensors(std::forward<decltype(args)>(args)...);
    } else if constexpr (std::is_same_v<std::decay_t<execute_on_worker_thread_return_t>, Tensor>) {
        return std::vector{Tensor(tt::tt_metal::operation::get_workers_for_op_output(inputs, optional_inputs))};

    } else if constexpr (detail::is_homogenous_tuple<execute_on_worker_thread_return_t, Tensor>()) {
        Tensors output_tensors;
        output_tensors.reserve(std::tuple_size_v<execute_on_worker_thread_return_t>);
        for (auto index = 0; index < std::tuple_size_v<execute_on_worker_thread_return_t>; index++) {
            output_tensors.emplace_back(
                Tensor(tt::tt_metal::operation::get_workers_for_op_output(inputs, optional_inputs)));
        }
        return output_tensors;
    } else {
        static_assert(
            tt::stl::concepts::always_false_v<operation_t>,
            "Operation is expecting the operator() method to return either a single Tensor or a tuple "
            "of "
            "Tensor(s). If the operation returns a vector of Tensors, it must implement create_async_output_tensors.");
    }
}

template <typename... args_t>
auto map_launch_op_args_to_execute_on_worker_thread_args(
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    const args_t&... args) {
    auto input_tensor_index = 0;
    auto optional_input_tensor_index = 0;
    auto optional_output_tensor_index = 0;
    return std::tuple{[&input_tensor_index,
                       &input_tensors,
                       &optional_input_tensor_index,
                       &optional_input_tensors,
                       &optional_output_tensor_index,
                       &optional_output_tensors](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::vector<Tensor>>) {
            return input_tensors;
        }
        if constexpr (std::is_same_v<T, Tensor>) {
            return input_tensors.at(input_tensor_index++);
        } else if constexpr (std::is_same_v<T, std::optional<const Tensor>>) {
            return optional_input_tensors.at(optional_input_tensor_index++);
        } else if constexpr (std::is_same_v<T, std::optional<Tensor>>) {
            return optional_output_tensors.at(optional_output_tensor_index++);
        } else {
            return arg;
        }
    }(args)...};
}

template <typename operation_t, typename T>
auto map_execute_on_worker_thread_return_to_launch_op_return(const T&& value) {
    if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Tensors>) {
        return value;
    } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Tensor>) {
        return std::vector{value};
    } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>, OptionalTensors>) {
        return value;
    } else if constexpr (is_homogenous_tuple<T, Tensor>()) {
        Tensors output_tensors;
        output_tensors.reserve(std::tuple_size_v<T>);
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            using std::get;
            (output_tensors.emplace_back(std::forward<decltype(get<Is>(value))>(get<Is>(value))), ...);
        }(std::make_index_sequence<std::tuple_size_v<T>>{});
        return output_tensors;
    } else {
        static_assert(
            tt::stl::concepts::always_false_v<operation_t>,
            "Operation must return either a single Tensor or a vector of Tensors or a vector of optional Tensors "
            "implement map_execute_on_worker_thread_return_to_launch_op_return.");
    }
}

template <typename... args_t>
void log(const std::string& prefix, args_t&&... args) {
    auto args_tuple = std::tuple{[](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_pointer_v<T>) {
            return fmt::format("{}", tt::stl::get_type_name<T>());
        } else {
            return arg;
        }
    }(std::forward<args_t>(args))...};

    std::string fmt{prefix};
    fmt += "\n";
    for (int i = 0; i < sizeof...(args); i++) {
        fmt += fmt::format("\t{:2}: {}\n", i, "{}");
    }
    std::apply([&fmt](const auto&... args) { tt::log_debug(tt::LogOp, fmt.c_str(), args...); }, args_tuple);
}

// Get "add" from "ttnn::add"
static const std::string base_name(const std::string& cpp_fully_qualified_name) {
    auto last_token = cpp_fully_qualified_name.substr(cpp_fully_qualified_name.rfind("::") + 2);
    return last_token;
}

// Convert "ttnn::add" to "add_t"
static const std::string class_name(const std::string& cpp_fully_qualified_name) {
    return base_name(cpp_fully_qualified_name) + "_t";
}

// Convert "ttnn::add" to "ttnn.add"
static const std::string python_fully_qualified_name(const std::string& cpp_fully_qualified_name) {
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

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t, bool auto_launch_op>
struct registered_operation_t {
    static constexpr auto is_primitive = PrimitiveOperationConcept<operation_t>;

    // Get "add" from "ttnn::add"
    const std::string base_name() const { return detail::base_name(std::string{cpp_fully_qualified_name}); }

    // Convert "ttnn::add" to "add_t"
    const std::string class_name() const { return detail::class_name(std::string{cpp_fully_qualified_name}); }

    // Convert "ttnn::add" to "ttnn.add"
    const std::string python_fully_qualified_name() const {
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
        tt::log_debug(tt::LogOp, "Started C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});
        tt::tt_metal::GraphTracker::instance().track_function_start(cpp_fully_qualified_name, args...);

        auto output = invoke(std::forward<args_t>(args)...);

        tt::tt_metal::GraphTracker::instance().track_function_end(output);
        tt::log_debug(tt::LogOp, "Finished C++ ttnn operation: {}", std::string_view{cpp_fully_qualified_name});
        return output;
    }

    template <typename... args_t>
        requires PrimitiveOperationConcept<operation_t>
    auto invoke(QueueId queue_id, args_t&&... args) const {
        static_assert(
            requires { operation_t::invoke(std::forward<decltype(args)>(args)...); },
            "Primitive Operation must implement operator() method to be invoked.");
        ZoneScopedN("Run primitive ttnn operation");
        ZoneName(static_cast<const char*>(cpp_fully_qualified_name.data.data()), cpp_fully_qualified_name.size());
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
        requires(not auto_launch_op)
    auto invoke_composite(args_t&&... args) const {
        ZoneScopedN("Run composite ttnn operation ");
        ZoneName(static_cast<const char*>(cpp_fully_qualified_name.data.data()), cpp_fully_qualified_name.size());
        return operation_t::invoke(std::forward<decltype(args)>(args)...);
    }

    template <typename... args_t>
        requires(auto_launch_op)
    auto invoke_composite(args_t&&... args) const {
        ZoneScopedN("Run composite ttnn operation (using auto async)");
        ZoneName(static_cast<const char*>(cpp_fully_qualified_name.data.data()), cpp_fully_qualified_name.size());

        // #8479: Fix and re-enable logging in cpp operation decorator
        // detail::log("Arguments: ", std::forward<args_t>(args)...);

        using execute_on_worker_thread_return_t = decltype(operation_t::invoke(args...));

        Tensors single_input_tensor = detail::extract_args_to_vector<ttnn::Tensor>(args...);
        const OptionalConstTensors optional_input_tensors =
            detail::extract_args_to_vector<std::optional<const ttnn::Tensor>>(args...);
        std::vector<std::vector<ttnn::Tensor>> vec_input_tensors =
            detail::extract_args_to_vector<std::vector<ttnn::Tensor>>(args...);
        if (!(single_input_tensor.empty() || vec_input_tensors.empty())) {
            TT_THROW(
                "Only one of single_input_tensor or vec_input_tensors can be specified."
                "Ensure that your invoke function does not have both Tensor and std::vector<Tensor> as input "
                "parameters");
        }
        if (single_input_tensor.empty() && vec_input_tensors.size() > 1) {
            TT_THROW(
                "You have more than one std::vector<Tensor> input parameters in the invoke. Only one vector is "
                "allowed");
        }

        auto& input_tensors = !vec_input_tensors.empty() ? vec_input_tensors[0] : single_input_tensor;

        auto output_tensors = detail::create_async_output_tensors<operation_t, execute_on_worker_thread_return_t>(
            input_tensors, optional_input_tensors, args...);

        const OptionalTensors optional_output_tensors =
            detail::extract_args_to_vector<std::optional<ttnn::Tensor>>(args...);

        tt::tt_metal::operation::launch_op(
            [args...](
                const Tensors& input_tensors,
                const OptionalConstTensors& optional_input_tensors,
                const OptionalTensors& optional_output_tensors) {
                auto execute_on_worker_thread_args = detail::map_launch_op_args_to_execute_on_worker_thread_args(
                    input_tensors, optional_input_tensors, optional_output_tensors, args...);
                return std::apply(
                    [](auto&&... args) {
                        return detail::map_execute_on_worker_thread_return_to_launch_op_return<operation_t>(
                            operation_t::invoke(std::forward<decltype(args)>(args)...));
                    },
                    execute_on_worker_thread_args);
            },
            input_tensors,
            output_tensors,
            optional_input_tensors,
            optional_output_tensors);

        if constexpr (std::is_same_v<std::decay_t<execute_on_worker_thread_return_t>, Tensor>) {
            return output_tensors.at(0);
        } else if constexpr (std::is_same_v<execute_on_worker_thread_return_t, Tensors>) {
            return output_tensors;
        } else if constexpr (std::is_same_v<execute_on_worker_thread_return_t, OptionalTensors>) {
            return output_tensors;
        } else if constexpr (detail::is_homogenous_tuple<execute_on_worker_thread_return_t, Tensor>()) {
            return detail::make_tuple_from_vector<execute_on_worker_thread_return_t>(output_tensors);
        } else {
            static_assert(
                tt::stl::concepts::always_false_v<operation_t>,
                "Operation is expecting the operator() method to return either a single Tensor or a "
                "vector of "
                "Tensor(s).");
        }
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
                "Composite operations must not be in the `ttnn::prim` namespace.");
        }
    }
}

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t, bool auto_launch_op>
constexpr auto register_operation_impl() {
    assert_operation_in_correct_namespace<cpp_fully_qualified_name, operation_t>();
    constexpr auto operation = registered_operation_t<cpp_fully_qualified_name, operation_t, auto_launch_op>{};
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
    return register_operation_impl<cpp_fully_qualified_name, operation_t, false>();
}

template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
constexpr auto register_operation_with_auto_launch_op() {
    return register_operation_impl<cpp_fully_qualified_name, operation_t, true>();
}

}  // namespace decorators

using ttnn::decorators::register_operation;
using ttnn::decorators::register_operation_with_auto_launch_op;

}  // namespace ttnn
