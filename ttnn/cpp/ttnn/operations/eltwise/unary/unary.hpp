// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/unary_op.hpp"
#include "tt_eager/tt_dnn/op_library/downsample/downsample_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {

namespace operations {

namespace unary {

namespace detail {

inline const std::array<ttnn::TensorSchema, 1> input_tensor_schemas() {
    return {ttnn::TensorSchema{
        2,
        4,
        {ttnn::bfloat16, ttnn::bfloat8_b},
        {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
        true,
        false,
        false,
        false}};
}

template <typename... Args>
inline auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
    return std::forward_as_tuple(input_tensor);
}

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
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }
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
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const bool parameter = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id, input_tensor, {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}}, memory_config, optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const bool parameter = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}}, memory_config, optional_output_tensor);
    }
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithFloatParameter {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const float parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id, input_tensor, {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}}, memory_config, optional_output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const float parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}}, memory_config, optional_output_tensor);
    }
};

struct Softplus {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input,
        const float beta,
        const float threshold,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        TT_ASSERT(input.device()->arch() != tt::ARCH::GRAYSKULL, "Softplus is not currently supported on Grayskull");
        return detail::execute_on_worker_thread(
            DefaultQueueId, input, {UnaryWithParam{UnaryOpType::SOFTPLUS, {beta, threshold}}}, memory_config, optional_output_tensor);
    }
};

struct Sigmoid_accurate {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input, {UnaryWithParam(UnaryOpType::NEG),
                                    UnaryWithParam(UnaryOpType::EXP, 1.0f),
                                    UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f),
                                    UnaryWithParam(UnaryOpType::RECIP)},
                                    memory_config,
                                    optional_output_tensor);
    }
};

struct Unary_chain {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const std::vector<UnaryWithParam>& ops_chain,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, ops_chain, memory_config, optional_output_tensor);
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

REGISTER_UNARY_OPERATION(abs, ABS);
REGISTER_UNARY_OPERATION(acos, ACOS);
REGISTER_UNARY_OPERATION(asin, ASIN);
REGISTER_UNARY_OPERATION(atan, ATAN);
REGISTER_UNARY_OPERATION(cos, COS);
REGISTER_UNARY_OPERATION(erfinv, ERFINV);
REGISTER_UNARY_OPERATION(exp2, EXP2);
REGISTER_UNARY_OPERATION(expm1, EXPM1);
REGISTER_UNARY_OPERATION(eqz, EQZ);
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
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(heaviside, HEAVISIDE);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(leaky_relu, LEAKY_RELU);

// Other unaries
constexpr auto softplus = ttnn::register_operation<ttnn::operations::unary::Softplus>("ttnn::softplus");
constexpr auto sigmoid_accurate = ttnn::register_operation<ttnn::operations::unary::Sigmoid_accurate>("ttnn::sigmoid_accurate");
constexpr auto unary_chain = ttnn::register_operation<ttnn::operations::unary::Unary_chain>("ttnn::unary_chain");

}  // namespace ttnn
