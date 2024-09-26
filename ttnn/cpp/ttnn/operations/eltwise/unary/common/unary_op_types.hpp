// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "tt_metal/tt_stl/reflection.hpp"

namespace ttnn::operations::unary {

// These operations have a corresponding LLK available
enum class UnaryOpType {
    EXP,
    RECIP,
    GELU,
    RELU,
    SQRT,
    SIGMOID,
    LOG,
    TANH,
    LOG2,
    LOG10,
    SIN,
    COS,
    ABS,
    SIGN,
    SQUARE,
    EQZ,
    NEZ,
    GTZ,
    LTZ,
    GEZ,
    LEZ,
    RELU_MAX,
    RELU_MIN,
    POWER,
    LEAKY_RELU,
    ELU,
    EXP2,
    HEAVISIDE,
    EXPM1,
    SIGNBIT,
    ASIN,
    ACOS,
    RSQRT,
    RELU6,
    ATAN,
    ERF,
    ERFC,
    ISINF,
    ISPOSINF,
    ISNEGINF,
    ISNAN,
    LOGICAL_NOT_UNARY,
    ISFINITE,
    ERFINV,
    I0,
    TAN,
    RSUB,
    RDIV,
    SILU,
    SOFTPLUS,
    IDENTITY,
    NEG,
    ADD_UNARY_SFPU,
    SUB_UNARY_SFPU,
    MUL_UNARY_SFPU,
    DIV_UNARY_SFPU,
    IDENTITY_UINT32,
    UNARY_NE,
    UNARY_GT,
    UNARY_LT,
    TILED_PROD,
    TYPECAST,
    BITWISE_XOR,
    BITWISE_NOT,
    BITWISE_AND,
    BITWISE_OR,
    RIGHT_SHIFT,
    FLOOR,
    CEIL,
    LEFT_SHIFT,
    REMAINDER,
    FMOD,
    DROPOUT,
    ZEROS,
};

struct UnaryWithParam {
    UnaryOpType op_type;
    std::vector<float> params;

    UnaryWithParam(UnaryOpType op_type, const std::vector<float>& params) : op_type{op_type}, params{params} {}
    UnaryWithParam(UnaryOpType op_type, float param) : op_type{op_type}, params{param} {}
    UnaryWithParam(UnaryOpType op_type) : op_type{op_type} {}

    bool has_parameter() const { return params.size() > 0; }

    static constexpr auto attribute_names = std::forward_as_tuple("op_type", "param");
    const auto attribute_values() const { return std::forward_as_tuple(this->op_type, this->params); }
};

using FusedActivations = std::vector<ttnn::operations::unary::UnaryWithParam>;

}

namespace tt::stl::json {

template <>
struct from_json_t<ttnn::operations::unary::UnaryWithParam> {
    auto operator()(const nlohmann::json& json_object) const {
        return ttnn::operations::unary::UnaryWithParam{
            from_json<ttnn::operations::unary::UnaryOpType>(json_object["op_type"]),
            from_json<std::vector<float>>(json_object["params"])};
    }
};
};  // namespace tt::stl::json
