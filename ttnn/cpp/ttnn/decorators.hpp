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

template <typename T, typename = void>
struct has_validate_api_arguments : std::false_type {};
template <typename T>
struct has_validate_api_arguments<T, std::void_t<decltype(T::validate_api_arguments)>> : std::true_type {};

template <typename T, typename = void>
struct has_create_async_output_tensors : std::false_type {};
template <typename T>
struct has_create_async_output_tensors<T, std::void_t<decltype(T::create_async_output_tensors)>> : std::true_type {};

template <typename, typename T, typename... Args>
struct has_map_launch_op_args_to_execute : std::false_type {};

template <typename T, typename... Args>
struct has_map_launch_op_args_to_execute<
    decltype((void)(T::map_launch_op_args_to_execute(
        std::declval<const Tensors&>(), std::declval<const OptionalConstTensors&>(), std::declval<Args>()...))),
    T,
    Args...> : std::true_type {};

template <typename T, typename = void>
struct has_map_execute_return_to_vector : std::false_type {};
template <typename T>
struct has_map_execute_return_to_vector<T, std::void_t<decltype(T::map_execute_return_to_vector)>> : std::true_type {};

template <class T, class... Args>
using execute_return_t = decltype(T::execute(std::declval<Args>()...));

template <class T, class... Args>
constexpr bool has_execute() {
    return std::experimental::is_detected_v<execute_return_t, T, Args&&...>;
}

template <class T, class... Args>
using execute_async_return_t = decltype(T::execute_async(std::declval<Args>()...));

template <class T, class... Args>
constexpr bool has_execute_async() {
    return std::experimental::is_detected_v<execute_async_return_t, T, Args&&...>;
}

// TODO: come back here and use this to automate map_launch_op_args_to_execute when this method is not implemented
template <typename... Args1, typename... Args2>
inline constexpr auto rearrange_tuples_to_match_argument_order(
    const std::tuple<Args1...>& order, const std::tuple<Args2...>& to_rearrange) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::make_tuple(std::get<Is>(to_rearrange)...);
    }(std::index_sequence<std::tuple_size_v<std::tuple<Args2...>>>{});
}

template <typename Include, typename... Args>
auto extract_args_to_vector(Args&&... args) {
    std::vector<Include> result;
    result.reserve(sizeof...(Args));

    auto process_arg = [&](auto&& arg) {
        using ArgType = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<ArgType, Include>) {
            result.push_back(arg);
        }
    };
    (process_arg(std::forward<Args>(args)), ...);
    return result;
}

template <typename T, typename... Excludes>
constexpr bool is_any_of = (... || std::is_same_v<std::decay_t<T>, Excludes>);

template <typename T, typename... Exclude>
constexpr auto conditional_tuple(T&& arg) {
    if constexpr (is_any_of<T, Exclude...>) {
        return std::tuple<>();
    } else {
        return std::tuple<std::decay_t<T>>{std::forward<T>(arg)};
    }
}

template <typename... Exclude, typename... Args>
constexpr auto extract_args_excluding_types_by_value(Args&&... args) {
    return std::tuple_cat(conditional_tuple<Args, Exclude...>(std::forward<Args>(args))...);
}

template <typename concrete_operation_t, typename... Args>
inline Tensors create_async_output_tensors(
    const Tensors&& inputs, const OptionalConstTensors&& optional_inputs, Args&&... args) {
    if constexpr (has_create_async_output_tensors<concrete_operation_t>::value) {
        return concrete_operation_t::create_async_output_tensors(std::forward<Args>(args)...);
    } else {
        bool enable_autoformat_device = false;
        return {Tensor(operation::get_workers_for_op_output(
            std::move(inputs), std::move(optional_inputs), enable_autoformat_device))};
    }
}

template <typename T, typename Return, typename... Args>
constexpr auto resolve_execute_method(Return (*launch)(Args...)) {
    return [](const T& self, Args... args) { return self(std::forward<Args>(args)...); };
}

namespace detail {
template <typename concrete_operation_t, typename... Args>
constexpr auto validate(const char* cpp_fully_qualified_name, Args&&... args) {
    if (ttnn::CONFIG.enable_fast_runtime_mode) {
        return;
    }
    auto tensors_to_validate = concrete_operation_t::input_tensors_to_validate(std::forward<Args>(args)...);
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
        concrete_operation_t::validate_api_arguments(std::forward<Args>(args)...);
    }
}

template <typename concrete_operation_t, typename T>
constexpr auto map_execute_return_to_vector(const T&& value) -> Tensors {
    if constexpr (has_map_execute_return_to_vector<concrete_operation_t>::value) {
        return concrete_operation_t::map_execute_return_to_vector(std::forward<decltype(value)>(value));
    } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Tensors>) {
        return value;
    } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Tensor>) {
        return {value};
    } else {
        static_assert(
            tt::stl::concepts::always_false_v<concrete_operation_t>,
            "Operation must return either a single Tensor or a vector of Tensors or implement "
            "map_execute_return_to_vector.");
    }
}
}  // namespace detail

template <auto id, typename concrete_operation_t>
struct operation_t {
    const char* cpp_fully_qualified_name;  // TODO: move this to template args when C++20 is available

    template <typename... Args>
    auto operator()(Args&&... args) const {
        ZoneScopedN("ttnn::decorators::operation_t::operator()");
        tt::log_debug(tt::LogOp, "Started   C++ ttnn operation: {}", this->cpp_fully_qualified_name);

        static_assert(
            has_execute<concrete_operation_t, Args&&...>() xor has_execute_async<concrete_operation_t, Args&&...>(),
            "Operation must either implement execute or execute_async.");

        if constexpr (has_execute<concrete_operation_t, Args&&...>()) {
            using return_type_of_execute = execute_return_t<concrete_operation_t, Args&&...>;
            const Tensors input_tensors = extract_args_to_vector<ttnn::Tensor>(std::forward<Args>(args)...);
            const OptionalConstTensors optional_input_tensors =
                extract_args_to_vector<std::optional<const ttnn::Tensor>>(std::forward<Args>(args)...);

            auto output_tensors = create_async_output_tensors<concrete_operation_t>(
                std::move(input_tensors), std::move(optional_input_tensors), std::forward<Args>(args)...);

            auto remaining_args =
                extract_args_excluding_types_by_value<ttnn::Tensor, std::optional<const ttnn::Tensor>>(
                    std::forward<Args>(args)...);

            // TODO: add support for optional_output_tensors
            // auto optional_output_tensors = extract_args_to_vector(std::forward<Args>(args)...,
            // std::optional<ttnn::Tensor>);

            bool enable_autoformat = false;
            operation::launch_op(
                [cpp_fully_qualified_name = this->cpp_fully_qualified_name, remaining_args](
                    const Tensors& input_tensors,
                    const OptionalConstTensors& optional_input_tensors,
                    const OptionalTensors&) mutable -> Tensors {
                    tt::log_debug(tt::LogOp, "Launching C++ ttnn operation in async: {}", cpp_fully_qualified_name);

                    return std::apply(
                        [cpp_fully_qualified_name, &input_tensors, &optional_input_tensors](auto&&... args) -> Tensors {
                            auto args_execute_tuple = concrete_operation_t::map_launch_op_args_to_execute(
                                input_tensors, optional_input_tensors, std::forward<decltype(args)>(args)...);
                            return std::apply(
                                [cpp_fully_qualified_name](auto&&... args) -> Tensors {
                                    detail::validate<concrete_operation_t>(
                                        cpp_fully_qualified_name, std::forward<decltype(args)>(args)...);
                                    return detail::map_execute_return_to_vector<concrete_operation_t>(
                                        concrete_operation_t::execute(std::forward<decltype(args)>(args)...));
                                },
                                args_execute_tuple);
                        },
                        remaining_args);
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
