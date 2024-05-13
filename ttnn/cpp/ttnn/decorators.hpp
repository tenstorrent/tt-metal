// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {
namespace decorators {

namespace detail {

template <typename T, typename = void>
struct has_validate_api_arguments : std::false_type {};
template <typename T>
struct has_validate_api_arguments<T, std::void_t<decltype(T::validate_api_arguments)>> : std::true_type {};

template <typename T, typename = void>
struct has_create_async_output_tensors : std::false_type {};
template <typename T>
struct has_create_async_output_tensors<T, std::void_t<decltype(T::create_async_output_tensors)>> : std::true_type {};

template <class T, class... args_t>
using execute_return_t = decltype(T::execute(std::declval<args_t>()...));

template <class T, class... args_t>
constexpr bool has_execute() {
    return std::experimental::is_detected_v<execute_return_t, T, args_t&&...>;
}

template <class T, class... args_t>
using execute_async_return_t = decltype(T::execute_async(std::declval<args_t>()...));

template <class T, class... args_t>
constexpr bool has_execute_async() {
    return std::experimental::is_detected_v<execute_async_return_t, T, args_t&&...>;
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

template <typename concrete_operation_t, typename... args_t>
inline Tensors create_async_output_tensors(
    const Tensors&& inputs, const OptionalConstTensors&& optional_inputs, args_t&&... args) {
    if constexpr (has_create_async_output_tensors<concrete_operation_t>::value) {
        return concrete_operation_t::create_async_output_tensors(std::forward<args_t>(args)...);
    } else {
        bool enable_autoformat_device = false;
        return {Tensor(operation::get_workers_for_op_output(
            std::move(inputs), std::move(optional_inputs), enable_autoformat_device))};
    }
}

template <typename concrete_operation_t, typename... args_t>
constexpr auto validate(const char* cpp_fully_qualified_name, args_t&&... args) {
    if (ttnn::CONFIG.enable_fast_runtime_mode) {
        return;
    }
    auto tensors_to_validate = concrete_operation_t::input_tensors_to_validate(std::forward<args_t>(args)...);
    static_assert(
        std::tuple_size_v<decltype(tensors_to_validate)> ==
            std::tuple_size_v<decltype(concrete_operation_t::input_tensor_schemas())>,
        "Number of tensors to validate must match the number of input tensors schemas");
    [cpp_fully_qualified_name, &tensors_to_validate]<auto... Ns>(std::index_sequence<Ns...>) {
        (ttnn::validate_input_tensor(
             cpp_fully_qualified_name,
             std::get<Ns>(tensors_to_validate),
             concrete_operation_t::input_tensor_schemas().at(Ns)),
         ...);
    }(std::make_index_sequence<std::tuple_size_v<decltype(tensors_to_validate)>>{});

    if constexpr (has_validate_api_arguments<concrete_operation_t>::value) {
        concrete_operation_t::validate_api_arguments(std::forward<args_t>(args)...);
    }
}

template <typename... args_t>
auto map_launch_op_args_to_execute_args(
    const Tensors& input_tensors, const OptionalConstTensors& optional_input_tensors, args_t&&... args) {
    auto input_tensor_index = 0;
    auto optional_input_tensor_index = 0;
    return std::tuple{
        [&input_tensor_index, &input_tensors, &optional_input_tensor_index, &optional_input_tensors](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Tensor>) {
                return input_tensors.at(input_tensor_index++);
            } else if constexpr (std::is_same_v<T, std::optional<const Tensor>>) {
                return optional_input_tensors.at(optional_input_tensor_index++);
            } else {
                return arg;
            }
        }(std::forward<args_t>(args))...};
}

template <typename concrete_operation_t, typename T>
constexpr auto map_execute_return_to_launch_op_return(const T&& value) -> Tensors {
    if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Tensors>) {
        return value;
    } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Tensor>) {
        return {value};
    } else {
        static_assert(
            tt::stl::concepts::always_false_v<concrete_operation_t>,
            "Operation must return either a single Tensor or a vector of Tensors or implement "
            "map_execute_return_to_launch_op_return.");
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
}  // namespace detail

template <auto id, typename concrete_operation_t>
struct operation_t {
    const char* cpp_fully_qualified_name;  // TODO: move this to template args when C++20 is available

    template <typename... args_t>
    auto operator()(args_t&&... args) const {
        ZoneScopedN("ttnn::decorators::operation_t::operator()");
        tt::log_debug(tt::LogOp, "Started   C++ ttnn operation: {}", this->cpp_fully_qualified_name);
        detail::log("Arguments: ", std::forward<args_t>(args)...);

        static_assert(
            detail::has_execute<concrete_operation_t, args_t&&...>() xor
                detail::has_execute_async<concrete_operation_t, args_t&&...>(),
            "Operation must either implement execute or execute_async.");

        if constexpr (detail::has_execute<concrete_operation_t, args_t&&...>()) {
            using return_type_of_execute = detail::execute_return_t<concrete_operation_t, args_t&&...>;
            const Tensors input_tensors = detail::extract_args_to_vector<ttnn::Tensor>(std::forward<args_t>(args)...);
            const OptionalConstTensors optional_input_tensors =
                detail::extract_args_to_vector<std::optional<const ttnn::Tensor>>(std::forward<args_t>(args)...);

            auto output_tensors = detail::create_async_output_tensors<concrete_operation_t>(
                std::move(input_tensors), std::move(optional_input_tensors), std::forward<args_t>(args)...);

            // TODO: add support for optional_output_tensors
            // auto optional_output_tensors = extract_args_to_vector(std::forward<args_t>(args)...,
            // std::optional<ttnn::Tensor>);

            bool enable_autoformat = false;
            operation::launch_op(
                [cpp_fully_qualified_name = this->cpp_fully_qualified_name, args...](
                    const Tensors& input_tensors,
                    const OptionalConstTensors& optional_input_tensors,
                    const OptionalTensors&) mutable -> Tensors {
                    tt::log_debug(
                        tt::LogOp, "Launching C++ ttnn operation in async mode: {}", cpp_fully_qualified_name);
                    auto execute_args = detail::map_launch_op_args_to_execute_args(
                        input_tensors, optional_input_tensors, std::forward<args_t>(args)...);
                    return std::apply(
                        [cpp_fully_qualified_name](auto&&... args) -> Tensors {
                            detail::validate<concrete_operation_t>(
                                cpp_fully_qualified_name, std::forward<decltype(args)>(args)...);
                            return detail::map_execute_return_to_launch_op_return<concrete_operation_t>(
                                concrete_operation_t::execute(std::forward<decltype(args)>(args)...));
                        },
                        execute_args);
                },
                input_tensors,
                output_tensors,
                optional_input_tensors,
                {},
                enable_autoformat);

            tt::log_debug(tt::LogOp, "Finished  C++ ttnn operation: {}", this->cpp_fully_qualified_name);

            if constexpr (std::is_same_v<std::decay_t<return_type_of_execute>, Tensors>) {
                return output_tensors;
            } else if constexpr (std::is_same_v<std::decay_t<return_type_of_execute>, Tensor>) {
                return output_tensors.at(0);
            } else {
                static_assert(
                    tt::stl::concepts::always_false_v<concrete_operation_t>,
                    "Operation is expecting the execute method to return either a single Tensor or a vector of "
                    "Tensor(s).");
            }

        } else {
            detail::validate<concrete_operation_t>(cpp_fully_qualified_name, std::forward<decltype(args)>(args)...);
            tt::log_debug(tt::LogOp, "Launching C++ ttnn operation in async: {}", cpp_fully_qualified_name);
            auto output = concrete_operation_t::execute_async(std::forward<decltype(args)>(args)...);
            tt::log_debug(tt::LogOp, "Finished  C++ ttnn operation: {}", this->cpp_fully_qualified_name);
            return output;
        }
    }

    // Get "add" from "ttnn::add"
    const std::string name() const {
        auto cpp_fully_qualified_name = std::string(this->cpp_fully_qualified_name);
        auto last_token = cpp_fully_qualified_name.substr(cpp_fully_qualified_name.rfind("::") + 2);
        return last_token;
    }

    // Convert "ttnn::add" to "ttnn_add_t"
    const std::string class_name() const { return this->name() + "_t"; }

    // Convert "ttnn::add" to "ttnn.add"
    const std::string python_fully_qualified_name() const {
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
        return replace(std::string{this->cpp_fully_qualified_name}, "::", ".");
    }
};

template <typename concrete_operation_t>
constexpr auto register_operation(const char* name) {
    return operation_t<__COUNTER__, concrete_operation_t>{name};
}

}  // namespace decorators

using ttnn::decorators::register_operation;

}  // namespace ttnn
