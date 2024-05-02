// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/tensor.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {
namespace decorators {

template <auto id, typename concrete_operation_t, auto... launch_args_t>
struct operation_t {
    const char* fully_qualified_name;  // TODO: move this to template args when C++20 is available

    template <typename... Args>
    constexpr auto operator()(Args&&... args) const {
        tt::log_debug(tt::LogOp, "Started  C++ ttnn operation: {}", this->fully_qualified_name);

        constexpr auto validate_execute_arguments = [](Args&&... args) {
            if constexpr (sizeof...(launch_args_t) > 0) {
                concrete_operation_t::template validate_execute_arguments<launch_args_t...>(
                    std::forward<Args>(args)...);
            } else {
                concrete_operation_t::validate_execute_arguments(std::forward<Args>(args)...);
            }
        };
        constexpr auto execute = [](Args&&... args) {
            if constexpr (sizeof...(launch_args_t) > 0) {
                return concrete_operation_t::template execute<launch_args_t...>(std::forward<Args>(args)...);
            } else {
                return concrete_operation_t::execute(std::forward<Args>(args)...);
            }
        };

        if (not ttnn::CONFIG.enable_fast_runtime_mode) {
            validate_execute_arguments(std::forward<Args>(args)...);
        }
        auto output = execute(std::forward<Args>(args)...);

        tt::log_debug(tt::LogOp, "Finished C++ ttnn operation: {}", this->fully_qualified_name);
        return output;
    }
};

template <typename concrete_operation_t, auto... launch_args_t>
constexpr auto register_operation(const char* name) {
    return operation_t<__COUNTER__, concrete_operation_t, launch_args_t...>{name};
}

}  // namespace decorators

using ttnn::decorators::register_operation;

}  // namespace ttnn
