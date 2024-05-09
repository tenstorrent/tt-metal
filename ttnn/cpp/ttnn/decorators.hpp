// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/tensor.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {
namespace decorators {

template <auto id, typename concrete_operation_t>
struct operation_t {
    const char* cpp_fully_qualified_name;  // TODO: move this to template args when C++20 is available

    template <typename... Args>
    auto operator()(Args&&... args) const {
        ZoneScopedN("ttnn::decorators::operation_t::operator()");
        tt::log_debug(tt::LogOp, "Started  C++ ttnn operation: {}", this->cpp_fully_qualified_name);

        if (not ttnn::CONFIG.enable_fast_runtime_mode) {
            auto tensors_to_validate = concrete_operation_t::input_tensors_to_validate(std::forward<Args>(args)...);
            static_assert(
                std::tuple_size_v<decltype(tensors_to_validate)> ==
                    std::tuple_size_v<decltype(concrete_operation_t::input_tensor_schemas())>,
                "Number of tensors to validate must match the number of input tensors schemas");
            [this, &tensors_to_validate]<auto... Ns>(std::index_sequence<Ns...>) {
                (ttnn::validate_input_tensor(
                     this->cpp_fully_qualified_name,
                     std::get<Ns>(tensors_to_validate),
                     concrete_operation_t::input_tensor_schemas().at(Ns)),
                 ...);
            }(std::make_index_sequence<std::tuple_size_v<decltype(tensors_to_validate)>>{});
        }

        auto output = concrete_operation_t::execute(std::forward<Args>(args)...);

        tt::log_debug(tt::LogOp, "Finished C++ ttnn operation: {}", this->cpp_fully_qualified_name);
        return output;
    }

    // Get "add" from "ttnn::add"
    const std::string name() const {
        auto cpp_fully_qualified_name = std::string(this->cpp_fully_qualified_name);
        auto last_token = cpp_fully_qualified_name.substr(cpp_fully_qualified_name.rfind("::") + 2);
        return last_token;
    }

    // Convert "ttnn::add" to "ttnn_add_t"
    const std::string class_name() const {
        return this->name() + "_t";
    }

    // Convert "ttnn::add" to "ttnn.add"
    const std::string python_fully_qualified_name() const {
        auto replace = [](const std::string& input, const std::string& from, const std::string& to) {
            if(from.empty()) { return input; }
            auto output = input;
            size_t start = 0;
            while((start = output.find(from, start)) != std::string::npos) {
                output.replace(start, from.length(), to);
                start += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
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
