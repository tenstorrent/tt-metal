// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "ttnn/core.hpp"

namespace ttnn {
namespace decorators {

using Tensors = tt::tt_metal::operation::Tensors;
using OptionalTensors = tt::tt_metal::operation::OptionalTensors;
using OptionalConstTensors = tt::tt_metal::operation::OptionalConstTensors;

namespace detail {

template <class T, class... args_t>
using execute_on_worker_thread_return_t = decltype(T::execute_on_worker_thread(std::declval<args_t>()...));

template <class T, class... args_t>
constexpr bool has_execute_on_worker_thread() {
    return std::experimental::is_detected_v<execute_on_worker_thread_return_t, T, args_t&&...>;
}

template <class T, class... args_t>
using execute_on_main_thread_return_t = decltype(T::execute_on_main_thread(std::declval<args_t>()...));

template <class T, class... args_t>
constexpr bool has_execute_on_main_thread() {
    return std::experimental::is_detected_v<execute_on_main_thread_return_t, T, args_t&&...>;
}

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

template <typename concrete_operation_t, typename execute_on_worker_thread_return_t>
inline Tensors create_async_output_tensors(const Tensors& inputs, const OptionalConstTensors& optional_inputs) {
    bool enable_autoformat_device = false;

    constexpr bool custom_create_async_outputs =
        requires(const concrete_operation_t& t) { t.create_async_output_tensors(inputs, optional_inputs); };

    if constexpr (custom_create_async_outputs) {
        return concrete_operation_t::create_async_output_tensors(inputs, optional_inputs);
    } else if constexpr (std::is_same_v<std::decay_t<execute_on_worker_thread_return_t>, Tensor>) {
        return {Tensor(operation::get_workers_for_op_output(inputs, optional_inputs, enable_autoformat_device))};
    } else if constexpr (detail::is_homogenous_tuple<execute_on_worker_thread_return_t, Tensor>()) {
        Tensors output_tensors;
        output_tensors.reserve(std::tuple_size_v<execute_on_worker_thread_return_t>);
        for (auto index = 0; index < std::tuple_size_v<execute_on_worker_thread_return_t>; index++) {
            output_tensors.emplace_back(
                Tensor(operation::get_workers_for_op_output(inputs, optional_inputs, enable_autoformat_device)));
        }
        return output_tensors;
    } else {
        static_assert(
            tt::stl::concepts::always_false_v<concrete_operation_t>,
            "Operation is expecting the execute_on_worker_thread method to return either a single Tensor or a tuple "
            "of "
            "Tensor(s). If the operation returns a vector of Tensors, it must implement create_async_output_tensors.");
    }
}

template <typename... args_t>
auto map_launch_op_args_to_execute_on_worker_thread_args(
    const Tensors& input_tensors, const OptionalConstTensors& optional_input_tensors, const OptionalTensors& optional_output_tensors, args_t&&... args) {
    auto input_tensor_index = 0;
    auto optional_input_tensor_index = 0;
    auto optional_output_tensor_index = 0;
    return std::tuple{
        [&input_tensor_index, &input_tensors, &optional_input_tensor_index, &optional_input_tensors, &optional_output_tensor_index, &optional_output_tensors](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Tensor>) {
                return input_tensors.at(input_tensor_index++);
            } else if constexpr (std::is_same_v<T, std::optional<const Tensor>>) {
                return optional_input_tensors.at(optional_input_tensor_index++);
            } else if constexpr (std::is_same_v<T, std::optional<Tensor>>) {
                return optional_output_tensors.at(optional_output_tensor_index++);
            } else {
                return arg;
            }
        }(std::forward<args_t>(args))...};
}

template <typename concrete_operation_t, typename T>
auto map_execute_on_worker_thread_return_to_launch_op_return(const T&& value) -> Tensors {
    if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Tensors>) {
        return value;
    } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Tensor>) {
        return {value};
    } else if constexpr (is_homogenous_tuple<T, Tensor>()) {
        Tensors output_tensors;
        output_tensors.reserve(std::tuple_size_v<T>);
        std::apply(
            [&output_tensors](auto&&... args) {
                (output_tensors.emplace_back(std::forward<decltype(args)>(args)), ...);
            },
            value);
        return output_tensors;
    } else {
        static_assert(
            tt::stl::concepts::always_false_v<concrete_operation_t>,
            "Operation must return either a single Tensor or a vector of Tensors or implement "
            "map_execute_on_worker_thread_return_to_launch_op_return.");
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
static const std::string base_name(const char* cpp_fully_qualified_name) {
    auto cpp_fully_qualified_name_as_string = std::string(cpp_fully_qualified_name);
    auto last_token = cpp_fully_qualified_name_as_string.substr(cpp_fully_qualified_name_as_string.rfind("::") + 2);
    return last_token;
}

// Convert "ttnn::add" to "add_t"
static const std::string class_name(const char* cpp_fully_qualified_name) {
    return base_name(cpp_fully_qualified_name) + "_t";
}

// Convert "ttnn::add" to "ttnn.add"
static const std::string python_fully_qualified_name(const char* cpp_fully_qualified_name) {
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
    return replace(std::string{cpp_fully_qualified_name}, "::", ".");
}

}  // namespace detail

template <auto id, typename concrete_operation_t>
struct operation_t {
    const char* cpp_fully_qualified_name;  // TODO: move this to template args when C++20 is available

    template <typename... args_t>
    auto operator()(args_t&&... args) const {
        ZoneScopedN("Run ttnn operation (struct-based)");
        ZoneName(this->cpp_fully_qualified_name, std::strlen(this->cpp_fully_qualified_name));
        tt::log_debug(tt::LogOp, "Started   C++ ttnn operation: {}", this->cpp_fully_qualified_name);

        // #8479: Fix and re-enable logging in cpp operation decorator
        // detail::log("Arguments: ", std::forward<args_t>(args)...);

        static_assert(
            detail::has_execute_on_worker_thread<concrete_operation_t, args_t&&...>() xor
                detail::has_execute_on_main_thread<concrete_operation_t, args_t&&...>(),
            "Operation must either implement execute_on_worker_thread or execute_on_main_thread.");

        if constexpr (detail::has_execute_on_worker_thread<concrete_operation_t, args_t&&...>()) {
            using execute_on_worker_thread_return_t =
                detail::execute_on_worker_thread_return_t<concrete_operation_t, args_t&&...>;
            const Tensors input_tensors = detail::extract_args_to_vector<ttnn::Tensor>(std::forward<args_t>(args)...);
            const OptionalConstTensors optional_input_tensors =
                detail::extract_args_to_vector<std::optional<const ttnn::Tensor>>(std::forward<args_t>(args)...);

            auto output_tensors =
                detail::create_async_output_tensors<concrete_operation_t, execute_on_worker_thread_return_t>(
                    input_tensors, optional_input_tensors);


            const OptionalTensors optional_output_tensors =
                detail::extract_args_to_vector<std::optional<ttnn::Tensor>>(std::forward<args_t>(args)...);

            bool enable_autoformat = false;
            operation::launch_op(
                [cpp_fully_qualified_name = this->cpp_fully_qualified_name, args...](
                    const Tensors& input_tensors,
                    const OptionalConstTensors& optional_input_tensors,
                    const OptionalTensors& optional_output_tensors) mutable -> Tensors {
                    auto execute_on_worker_thread_args = detail::map_launch_op_args_to_execute_on_worker_thread_args(
                        input_tensors, optional_input_tensors, optional_output_tensors, std::forward<args_t>(args)...);
                    return std::apply(
                        [cpp_fully_qualified_name](auto&&... args) -> Tensors {
                            return detail::map_execute_on_worker_thread_return_to_launch_op_return<
                                concrete_operation_t>(
                                concrete_operation_t::execute_on_worker_thread(std::forward<decltype(args)>(args)...));
                        },
                        execute_on_worker_thread_args);
                },
                input_tensors,
                output_tensors,
                optional_input_tensors,
                optional_output_tensors,
                enable_autoformat);

            tt::log_debug(tt::LogOp, "Finished  C++ ttnn operation: {}", this->cpp_fully_qualified_name);

            if constexpr (std::is_same_v<std::decay_t<execute_on_worker_thread_return_t>, Tensor>) {
                return output_tensors.at(0);
            } else if constexpr (std::is_same_v<execute_on_worker_thread_return_t, Tensors>) {
                return output_tensors;
            } else if constexpr (detail::is_homogenous_tuple<execute_on_worker_thread_return_t, Tensor>()) {
                return detail::make_tuple_from_vector<execute_on_worker_thread_return_t>(output_tensors);
            } else {
                static_assert(
                    tt::stl::concepts::always_false_v<concrete_operation_t>,
                    "Operation is expecting the execute_on_worker_thread method to return either a single Tensor or a "
                    "vector of "
                    "Tensor(s).");
            }

        } else {
            auto output = concrete_operation_t::execute_on_main_thread(std::forward<decltype(args)>(args)...);
            tt::log_debug(tt::LogOp, "Finished  C++ ttnn operation: {}", this->cpp_fully_qualified_name);
            return output;
        }
    }


    // Get "add" from "ttnn::add"
    const std::string base_name() const { return detail::base_name(this->cpp_fully_qualified_name); }

    // Convert "ttnn::add" to "add_t"
    const std::string class_name() const { return detail::class_name(this->cpp_fully_qualified_name); }

    // Convert "ttnn::add" to "ttnn.add"
    const std::string python_fully_qualified_name() const {
        return detail::python_fully_qualified_name(this->cpp_fully_qualified_name);
    }
};

template <typename concrete_operation_t>
constexpr auto register_operation(const char* cpp_fully_qualified_name) {
    return operation_t<__COUNTER__, concrete_operation_t>{cpp_fully_qualified_name};
}

namespace detail {
template <auto lambda_t>
struct lambda_operation_t {
    static auto execute_on_main_thread(auto&&... args) { return lambda_t(std::forward<decltype(args)>(args)...); }
};
}  // namespace detail

// If you are feeling lazy, you can use this macro to create an operation struct from a lambda
// You  will have to implement async manually
#define REGISTER_OPERATION_FROM_FUNCTION(name, function)                                                        \
    (::ttnn::decorators::register_operation<::ttnn::decorators::detail::lambda_operation_t<[](auto&&... args) { \
        return function(std::forward<decltype(args)>(args)...);                                                 \
    }>>(name))

}  // namespace decorators

using ttnn::decorators::register_operation;

}  // namespace ttnn
