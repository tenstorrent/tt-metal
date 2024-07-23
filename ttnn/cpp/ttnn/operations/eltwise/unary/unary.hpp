// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/unary_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/data_movement/downsample/device/downsample_op.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {

namespace operations {

namespace unary {

namespace detail {

inline Tensor execute_on_worker_thread(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    DataType output_dtype = (op_chain[0].op_type == UnaryOpType::TYPECAST) ? static_cast<DataType>(op_chain[0].params[1]) : input_tensor.get_dtype();
    bool preserve_fp32_precision = (op_chain[0].op_type == UnaryOpType::TYPECAST) and (input_tensor.get_dtype() == DataType::FLOAT32);
    bool fp32_dest_acc_en = preserve_fp32_precision or
                            output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or
                            output_dtype == DataType::FLOAT32 or
                            input_tensor.get_dtype() == DataType::UINT32 or
                            input_tensor.get_dtype() == DataType::INT32;  // MT: Currently only uint32/int32 is moved to
                                                                          // DST directly, fp32 is converted to fp16b

    auto output_memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config.value_or(input_tensor.memory_config());
    return operation::run(
               Unary{op_chain, output_memory_config, fp32_dest_acc_en, preserve_fp32_precision, output_dtype},
               {input_tensor}, {}, {optional_output_tensor}, queue_id).at(0);
}

}  // namespace detail

template <UnaryOpType... unary_op_types>
struct ExecuteUnary {
    static Tensor execute_on_worker_thread(
        uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(queue_id, input_tensor, {UnaryWithParam{unary_op_types}...}, memory_config, optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(DefaultQueueId, input_tensor, {UnaryWithParam{unary_op_types}...}, memory_config, optional_output_tensor);
    }
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithFastAndApproximateMode {
    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const bool parameter = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id,
            input_tensor,
            {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}},
            memory_config,
            optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const bool parameter = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId,
            input_tensor,
            {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}},
            memory_config,
            optional_output_tensor);
    }
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithFloatParameter {
    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const float parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id,
            input_tensor,
            {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}},
            memory_config,
            optional_output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const float parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId,
            input_tensor,
            {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}},
            memory_config,
            optional_output_tensor);
    }
};

struct Softplus {
    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input,
        const float beta,
        const float threshold,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        TT_ASSERT(input.device()->arch() != tt::ARCH::GRAYSKULL, "Softplus is not currently supported on Grayskull");
        return detail::execute_on_worker_thread(
            queue_id, input, {UnaryWithParam{UnaryOpType::SOFTPLUS, {beta, threshold}}}, memory_config, optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        const Tensor& input,
        const float beta,
        const float threshold,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        TT_ASSERT(input.device()->arch() != tt::ARCH::GRAYSKULL, "Softplus is not currently supported on Grayskull");
        return detail::execute_on_worker_thread(
            DefaultQueueId,
            input,
            {UnaryWithParam{UnaryOpType::SOFTPLUS, {beta, threshold}}},
            memory_config,
            optional_output_tensor);
    }
};

struct Sigmoid_accurate {
    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id, input, {UnaryWithParam(UnaryOpType::NEG),
                                    UnaryWithParam(UnaryOpType::EXP, 1.0f),
                                    UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f),
                                    UnaryWithParam(UnaryOpType::RECIP)},
                                    memory_config,
                                    optional_output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId,
            input,
            {UnaryWithParam(UnaryOpType::NEG),
             UnaryWithParam(UnaryOpType::EXP, 1.0f),
             UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f),
             UnaryWithParam(UnaryOpType::RECIP)},
            memory_config,
            optional_output_tensor);
    }
};

struct Unary_chain {
    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const std::vector<UnaryWithParam>& ops_chain,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {

        TT_FATAL(ops_chain.size() > 0, "Op chain cannot be empty");
        return detail::execute_on_worker_thread(
            queue_id, input_tensor, ops_chain, memory_config, optional_output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const std::vector<UnaryWithParam>& ops_chain,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {

        TT_FATAL(ops_chain.size() > 0, "Op chain cannot be empty");
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, ops_chain, memory_config, optional_output_tensor);
    }
};

struct Identity {
    static Tensor execute_on_worker_thread(
        uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
            UnaryOpType op_type = UnaryOpType::IDENTITY;
            if(input_tensor.get_dtype() == DataType::UINT32) {
                op_type = UnaryOpType::IDENTITY_UINT32;
            }

        return detail::execute_on_worker_thread(queue_id, input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
            UnaryOpType op_type = UnaryOpType::IDENTITY;
            if(input_tensor.get_dtype() == DataType::UINT32) {
                op_type = UnaryOpType::IDENTITY_UINT32;
            }

        return detail::execute_on_worker_thread(DefaultQueueId, input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
    }
};

template <UnaryOpType unary_op_type, typename T = int32_t >
struct ExecuteUnaryWithIntegerParameter {

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        T parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id, input_tensor, {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}}, memory_config, optional_output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        T parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}}, memory_config, optional_output_tensor);
    }
};

template <UnaryOpType unary_op_type, typename T = float>
struct SymmetricBinop {

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        T param,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id, input_tensor, {UnaryWithParam(unary_op_type, static_cast<float>(param))}, memory_config, optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        T param,
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id, input_tensor, {UnaryWithParam(unary_op_type, static_cast<float>(param))}, memory_config, optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        T param,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, {UnaryWithParam(unary_op_type, static_cast<float>(param))}, memory_config, optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        T param,
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, {UnaryWithParam(unary_op_type, static_cast<float>(param))}, memory_config, optional_output_tensor);
    }

};

template <UnaryOpType unary_op_type, UnaryOpType unary_op_rev_type, typename T = float>
struct AsymmetricBinop {

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        T param,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id, input_tensor, {UnaryWithParam(unary_op_type, static_cast<float>(param))}, memory_config,  optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        T param,
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id, input_tensor, {UnaryWithParam(unary_op_rev_type, static_cast<float>(param))}, memory_config, optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        T param,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, {UnaryWithParam(unary_op_type, static_cast<float>(param))}, memory_config,  optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        T param,
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, {UnaryWithParam(unary_op_rev_type, static_cast<float>(param))}, memory_config, optional_output_tensor);
    }
};


}  // namespace unary
}  // namespace operations

#define REGISTER_UNARY_OPERATION(operation_name, operation_type)                                      \
    constexpr auto operation_name = ttnn::register_operation<                                         \
        ttnn::operations::unary::ExecuteUnary<ttnn::operations::unary::UnaryOpType::operation_type>>( \
        "ttnn::" #operation_name);

#define REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(operation_name, operation_type)   \
    constexpr auto operation_name =                                                               \
        ttnn::register_operation<ttnn::operations::unary::ExecuteUnaryWithFastAndApproximateMode< \
            ttnn::operations::unary::UnaryOpType::operation_type>>("ttnn::" #operation_name);

#define REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(operation_name, operation_type)                                 \
    constexpr auto operation_name = ttnn::register_operation<ttnn::operations::unary::ExecuteUnaryWithFloatParameter< \
        ttnn::operations::unary::UnaryOpType::operation_type>>("ttnn::" #operation_name);

#define REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(operation_name, operation_type, data_type)                                 \
    constexpr auto operation_name = ttnn::register_operation<ttnn::operations::unary::ExecuteUnaryWithIntegerParameter< \
        ttnn::operations::unary::UnaryOpType::operation_type, data_type>>("ttnn::" #operation_name);

REGISTER_UNARY_OPERATION(abs, ABS);
REGISTER_UNARY_OPERATION(acos, ACOS);
REGISTER_UNARY_OPERATION(asin, ASIN);
REGISTER_UNARY_OPERATION(atan, ATAN);
REGISTER_UNARY_OPERATION(cos, COS);
REGISTER_UNARY_OPERATION(erfinv, ERFINV);
REGISTER_UNARY_OPERATION(exp2, EXP2);
REGISTER_UNARY_OPERATION(expm1, EXPM1);
REGISTER_UNARY_OPERATION(eqz, EQZ);
REGISTER_UNARY_OPERATION(floor, FLOOR);
REGISTER_UNARY_OPERATION(ceil, CEIL);
REGISTER_UNARY_OPERATION(gez, GEZ);
REGISTER_UNARY_OPERATION(gtz, GTZ);
REGISTER_UNARY_OPERATION(i0, I0);
REGISTER_UNARY_OPERATION(isfinite, ISFINITE);
REGISTER_UNARY_OPERATION(isinf, ISINF);
REGISTER_UNARY_OPERATION(isnan, ISNAN);
REGISTER_UNARY_OPERATION(isneginf, ISNEGINF);
REGISTER_UNARY_OPERATION(isposinf, ISPOSINF);
REGISTER_UNARY_OPERATION(lez, LEZ);
REGISTER_UNARY_OPERATION(log, LOG);
REGISTER_UNARY_OPERATION(log10, LOG10);
REGISTER_UNARY_OPERATION(log2, LOG2);
REGISTER_UNARY_OPERATION(logical_not, LOGICAL_NOT_UNARY);
REGISTER_UNARY_OPERATION(ltz, LTZ);
REGISTER_UNARY_OPERATION(neg, NEG);
REGISTER_UNARY_OPERATION(nez, NEZ);
REGISTER_UNARY_OPERATION(reciprocal, RECIP);
REGISTER_UNARY_OPERATION(relu, RELU);
REGISTER_UNARY_OPERATION(relu6, RELU6);
REGISTER_UNARY_OPERATION(sigmoid, SIGMOID);
REGISTER_UNARY_OPERATION(sign, SIGN);
REGISTER_UNARY_OPERATION(signbit, SIGNBIT);
REGISTER_UNARY_OPERATION(silu, SILU);
REGISTER_UNARY_OPERATION(sin, SIN);
REGISTER_UNARY_OPERATION(sqrt, SQRT);
REGISTER_UNARY_OPERATION(square, SQUARE);
REGISTER_UNARY_OPERATION(tan, TAN);
REGISTER_UNARY_OPERATION(tanh, TANH);

constexpr auto log_sigmoid = ttnn::register_operation<ttnn::operations::unary::ExecuteUnary<
    ttnn::operations::unary::UnaryOpType::SIGMOID,
    ttnn::operations::unary::UnaryOpType::LOG>>("ttnn::log_sigmoid");

// Unaries with fast_and_approximate_mode
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(exp, EXP);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erf, ERF);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erfc, ERFC);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(gelu, GELU);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(rsqrt, RSQRT);

// Unaries with float parameter
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(elu, ELU);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(rsub, RSUB);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(heaviside, HEAVISIDE);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(leaky_relu, LEAKY_RELU);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(relu_max, RELU_MAX);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(relu_min, RELU_MIN);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(remainder, REMAINDER);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(fmod, FMOD);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(gt_unary, UNARY_GT);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(lt_unary, UNARY_LT);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(ne_unary, UNARY_NE);

// Unaries with integer parameter
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(power, POWER, uint32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_left_shift, LEFT_SHIFT, int32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_right_shift, RIGHT_SHIFT, int32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_and, BITWISE_AND, int32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_or, BITWISE_OR, int32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_xor, BITWISE_XOR, int32_t);
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(bitwise_not, BITWISE_NOT, int32_t);

// Unaries used for composite implementation
REGISTER_UNARY_OPERATION(tiled_prod, TILED_PROD);

// Other unaries
constexpr auto identity = ttnn::register_operation<ttnn::operations::unary::Identity>("ttnn::identity");
constexpr auto softplus = ttnn::register_operation<ttnn::operations::unary::Softplus>("ttnn::softplus");
constexpr auto sigmoid_accurate = ttnn::register_operation<ttnn::operations::unary::Sigmoid_accurate>("ttnn::sigmoid_accurate");
constexpr auto unary_chain = ttnn::register_operation<ttnn::operations::unary::Unary_chain>("ttnn::unary_chain");

constexpr auto add_sfpu = ttnn::register_operation<ttnn::operations::unary::SymmetricBinop<ttnn::operations::unary::UnaryOpType::ADD_UNARY_SFPU>>("ttnn::add_sfpu");
constexpr auto mul_sfpu = ttnn::register_operation<ttnn::operations::unary::SymmetricBinop<ttnn::operations::unary::UnaryOpType::MUL_UNARY_SFPU>>("ttnn::mul_sfpu");

constexpr auto sub_sfpu = ttnn::register_operation<ttnn::operations::unary::AsymmetricBinop<ttnn::operations::unary::UnaryOpType::SUB_UNARY_SFPU, ttnn::operations::unary::UnaryOpType::RSUB>>("ttnn::sub_sfpu");
constexpr auto div_sfpu = ttnn::register_operation<ttnn::operations::unary::AsymmetricBinop<ttnn::operations::unary::UnaryOpType::DIV_UNARY_SFPU, ttnn::operations::unary::UnaryOpType::RDIV>>("ttnn::div_sfpu");

}  // namespace ttnn
