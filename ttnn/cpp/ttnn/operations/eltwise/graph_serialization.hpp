// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/graph/graph_registration.hpp"
#include <tt_stl/reflection.hpp>
#include <variant>

namespace ttnn::operations::unary {

// UnaryOpType enum serialization
inline std::ostream& operator<<(std::ostream& os, const UnaryOpType& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

// BasicUnaryWithParam template serialization
template <typename... Ts>
inline std::ostream& operator<<(std::ostream& os, const BasicUnaryWithParam<Ts...>& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

}  // namespace ttnn::operations::unary

// Automatic type registration
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::unary::BasicUnaryWithParam<float>);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::unary::BasicUnaryWithParam<int>);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::unary::BasicUnaryWithParam<unsigned int>);
TTNN_REGISTER_GRAPH_ARG(ttnn::operations::unary::BasicUnaryWithParam<float, int, unsigned int>);
TTNN_REGISTER_GRAPH_ARG(std::variant<std::string, ttnn::operations::unary::BasicUnaryWithParam<float>>);
