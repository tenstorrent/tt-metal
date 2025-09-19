// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <tt_stl/reflection.hpp>

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
    LOG1P,
    TANH,
    LOG2,
    LOG10,
    SIN,
    COS,
    COSH,
    SINH,
    ABS,
    ABS_INT32,
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
    ACOSH,
    RSQRT,
    RELU6,
    ATAN,
    ASINH,
    ATANH,
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
    I1,
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
    UNARY_NE,
    UNARY_EQ,
    UNARY_GT,
    UNARY_LT,
    UNARY_GE,
    UNARY_LE,
    TILED_PROD,
    TYPECAST,
    BITWISE_XOR,
    BITWISE_NOT,
    BITWISE_AND,
    BITWISE_OR,
    RIGHT_SHIFT,
    FLOOR,
    CEIL,
    TRUNC,
    FRAC,
    ROUND,
    LEFT_SHIFT,
    REMAINDER,
    FMOD,
    DROPOUT,
    FILL,
    PRELU_SFPU,
    ZERO_POINT,
    ALT_COMPLEX_ROTATE90,
    MISH,
    MAXIMUM,
    MINIMUM,
    TANHSHRINK,
    THRESHOLD,
    SOFTSHRINK,
    HARDSHRINK,
    HARDTANH,
    HARDSIGMOID,
    HARDSWISH,
    WHERE_TSS,
    SOFTSIGN,
    CELU,
    CLAMP_TSS,
    SELU,
};

enum class VecMode {
    None = 0,
    R = 1,
    C = 2,
    RC = 4,
    RC_custom = 6,
    Invalid = 0xFF,
};

template <typename... Ts>
    requires(... and (std::integral<Ts> or std::floating_point<Ts>))
struct BasicUnaryWithParam {
    std::variant<BasicUnaryWithParam<Ts>...> base;

    template <typename T>
        requires(... or std::same_as<T, Ts>)
    BasicUnaryWithParam(UnaryOpType op_type, const std::vector<T>& params) :
        base{std::in_place_type<BasicUnaryWithParam<T>>, op_type, params} {}

    template <typename T>
        requires(... or std::same_as<T, Ts>)
    BasicUnaryWithParam(UnaryOpType op_type, std::initializer_list<T> params) :
        base{std::in_place_type<BasicUnaryWithParam<T>>, op_type, params} {}

    template <typename T>
        requires(... or std::same_as<T, Ts>)
    BasicUnaryWithParam(UnaryOpType op_type, T param) :
        base{std::in_place_type<BasicUnaryWithParam<T>>, op_type, param} {}

    BasicUnaryWithParam(UnaryOpType op_type) : base{std::in_place_index<0>, op_type} {}

    template <typename T>
        requires(... or std::same_as<T, Ts>)
    BasicUnaryWithParam(const BasicUnaryWithParam<T>& other) : base{other} {}

    UnaryOpType type() const noexcept {
        return std::visit([](const auto& activation) { return activation.type(); }, base);
    }

    bool has_parameter() const noexcept {
        return std::visit([](const auto& activation) { return activation.has_parameter(); }, base);
    }

    bool empty() const noexcept {
        return std::visit([](const auto& activation) { return activation.empty(); }, base);
    }

    std::variant<std::span<const Ts>...> get_params() const noexcept {
        return std::visit<decltype(get_params())>([](const auto& activation) { return activation.get_params(); }, base);
    }

    template <typename T>
        requires(... or std::same_as<T, Ts>)
    std::span<const T> get_params_if() const noexcept {
        if (const auto ptr = std::get_if<BasicUnaryWithParam<T>>(&base)) {
            return ptr->get_params();
        }

        return {};
    }

    template <typename T>
        requires(... or std::same_as<T, Ts>)
    std::optional<T> get_param_if(std::size_t index = 0) const noexcept {
        if (const auto ptr = std::get_if<BasicUnaryWithParam<T>>(&base)) {
            return ptr->get_param_if(index);
        }

        return std::nullopt;
    }

    static constexpr auto attribute_names = std::forward_as_tuple("base");
    auto attribute_values() const { return std::forward_as_tuple(this->base); }
};

template <typename T>
struct BasicUnaryWithParam<T> {
    UnaryOpType op_type;
    std::vector<T> params;

    BasicUnaryWithParam(UnaryOpType op_type, const std::vector<T>& params) : op_type{op_type}, params{params} {}
    BasicUnaryWithParam(UnaryOpType op_type, std::initializer_list<T> params) : op_type{op_type}, params{params} {}
    BasicUnaryWithParam(UnaryOpType op_type, T param) : op_type{op_type}, params{param} {}
    BasicUnaryWithParam(UnaryOpType op_type) : op_type{op_type} {}

    UnaryOpType type() const noexcept { return op_type; }

    bool has_parameter() const noexcept { return params.size() > 0; }

    bool empty() const noexcept { return params.empty(); }

    std::span<const T> get_params() const noexcept { return params; }

    std::optional<T> get_param_if(std::size_t index = 0) const noexcept {
        if (index >= params.size()) {
            return std::nullopt;
        }

        return params[index];
    }

    static constexpr auto attribute_names = std::forward_as_tuple("op_type", "param");
    auto attribute_values() const { return std::forward_as_tuple(this->op_type, this->params); }
};

using EltwiseUnaryWithParam = BasicUnaryWithParam<float, std::int32_t, std::uint32_t>;

using UnaryWithParam = BasicUnaryWithParam<float>;

template <typename... Ts>
using BasicFusedActivations = std::vector<ttnn::operations::unary::BasicUnaryWithParam<Ts...>>;

using EltwiseFusedActivations = std::vector<ttnn::operations::unary::EltwiseUnaryWithParam>;

using FusedActivations = std::vector<ttnn::operations::unary::UnaryWithParam>;

}  // namespace ttnn::operations::unary

template <typename T>
struct ttsl::json::from_json_t<ttnn::operations::unary::BasicUnaryWithParam<T>> {
    auto operator()(const nlohmann::json& json_object) const {
        return ttnn::operations::unary::BasicUnaryWithParam<T>{
            from_json<ttnn::operations::unary::UnaryOpType>(json_object["op_type"]),
            from_json<std::vector<T>>(json_object["params"])};
    }
};
