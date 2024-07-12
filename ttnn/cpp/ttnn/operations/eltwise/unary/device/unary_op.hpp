// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/experimental/tensor/host_buffer/functions.hpp"
#include "ttnn/experimental/tensor/tensor_utils.hpp"

#include "ttnn/experimental/tt_dnn/op_library/run_operation.hpp"


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
};

template <typename T>
bool is_parametrized_type(T val) {
    switch (val) {
        case UnaryOpType::RELU_MAX:
        case UnaryOpType::RELU_MIN:
        case UnaryOpType::POWER:
        case UnaryOpType::LEAKY_RELU:
        case UnaryOpType::ELU:
        case UnaryOpType::GELU:
        case UnaryOpType::RSQRT:
        case UnaryOpType::HEAVISIDE:
        case UnaryOpType::ERF:
        case UnaryOpType::ERFC:
        case UnaryOpType::RSUB:
        case UnaryOpType::RDIV:
        case UnaryOpType::EXP:
        case UnaryOpType::SOFTPLUS:
        case UnaryOpType::ADD_UNARY_SFPU:
        case UnaryOpType::SUB_UNARY_SFPU:
        case UnaryOpType::MUL_UNARY_SFPU:
        case UnaryOpType::DIV_UNARY_SFPU:
        case UnaryOpType::UNARY_NE:
        case UnaryOpType::UNARY_GT:
        case UnaryOpType::UNARY_LT:
        case UnaryOpType::TYPECAST:
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::BITWISE_NOT:
        case UnaryOpType::BITWISE_AND:
        case UnaryOpType::BITWISE_OR:
        case UnaryOpType::RIGHT_SHIFT:
        case UnaryOpType::LEFT_SHIFT:
        case UnaryOpType::REMAINDER:
        case UnaryOpType::FMOD: return true;
        default: return false;
    }
    return false;
}

constexpr uint8_t DefaultQueueId = 0;

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

inline UnaryWithParam string_to_unary_with_param(const std::string& name) {
    if (name == "relu")
        return UnaryWithParam(UnaryOpType::RELU);
    else if (name == "gelu")
        return UnaryWithParam(UnaryOpType::GELU, static_cast<float>(true));
    else if (name == "silu")
        return UnaryWithParam(UnaryOpType::SILU);
    else if (name == "sigmoid")
        return UnaryWithParam(UnaryOpType::SIGMOID);
    else if (name == "sqrt")
        return UnaryWithParam(UnaryOpType::SQRT);
    else if (name == "exp")
        return UnaryWithParam(UnaryOpType::EXP, static_cast<float>(true));
    else if (name == "recip")
        return UnaryWithParam(UnaryOpType::RECIP);
    else if (name == "log")
        return UnaryWithParam(UnaryOpType::LOG);
    else if (name == "tanh")
        return UnaryWithParam(UnaryOpType::TANH);
    else if (name == "log2")
        return UnaryWithParam(UnaryOpType::LOG2);
    else if (name == "log10")
        return UnaryWithParam(UnaryOpType::LOG10);
    else if (name == "sin")
        return UnaryWithParam(UnaryOpType::SIN);
    else if (name == "cos")
        return UnaryWithParam(UnaryOpType::COS);
    else if (name == "abs")
        return UnaryWithParam(UnaryOpType::ABS);
    else if (name == "sign")
        return UnaryWithParam(UnaryOpType::SIGN);
    else if (name == "square")
        return UnaryWithParam(UnaryOpType::SQUARE);
    else if (name == "softplus")
        return UnaryWithParam(UnaryOpType::SOFTPLUS);
    TT_THROW("Unknown unary op: " + name);
}

enum class UnaryOpParallelizationStrategy { MULTI_CORE, SHARDED_MULTI_CORE };

struct Unary {
    const std::vector<UnaryWithParam> op_chain;
    const MemoryConfig output_mem_config;
    bool fp32_dest_acc_en;
    bool preserve_fp32_precision;
    DataType output_dtype;

    void validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &optional_output_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    UnaryOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;

    const operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

namespace utils {

bool get_op_approx_mode(UnaryOpType op_type);
std::pair<string, string> get_op_init_and_func(UnaryOpType op_type, std::vector<float> params = {}, string idst = "0");
std::map<string, string> get_defines(
    UnaryOpType op_type, std::optional<std::vector<float>> params = std::nullopt, string id = "0", string idst = "0");
std::map<string, string> get_block_defines(
    const std::vector<UnaryWithParam>& op_chain, string block_id = "0", string idst = "0");
}  // namespace utils

}  // namespace ttnn::operations::unary
