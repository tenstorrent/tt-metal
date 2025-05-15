
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn {

namespace operations::unary_backward {

Tensor change_layout_to_tile(const Tensor& temp, const MemoryConfig& output_mem_config);

struct ExecuteUnaryBackwardNeg {
    static std::vector<std::optional<Tensor>> invoke(
        QueueId queue_id,
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardThreshold {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float min,
        float max,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardRpow {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float scalar,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardDivNoNan {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float scalar,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardPolygamma {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        int scalar,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardLog {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardRound {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardFloor {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardLogit {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardRelu {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardAcosh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardCos {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardHardsigmoid {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardLgamma {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardMultigammaln {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardSoftplus {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float parameter_a,
        float parameter_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardHardtanh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float parameter_a,
        float parameter_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardHardshrink {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float parameter_a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardSoftshrink {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float parameter_a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardLeakyRelu {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float parameter_a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardElu {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float parameter_a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardCelu {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float parameter_a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardLogiteps {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float parameter_a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardTan {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardSquare {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardSelu {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardRelu6 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardI0 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardFillZero {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardLogSigmoid {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardTrunc {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardFrac {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardRad2deg {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardAtan {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardAcos {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardErfc {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardErfinv {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardDigamma {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardExpm1 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardExp2 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardSign {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardLog2 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardCosh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardSoftsign {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardCeil {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardSigmoid {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardLog1p {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardLog10 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardSinh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardSin {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardAsinh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardAsin {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardAtanh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardTanhshrink {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardHardswish {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardDeg2rad {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardErf {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardRsqrt {
    static std::vector<std::optional<Tensor>> invoke(
        QueueId queue_id,
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardClamp {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        std::optional<float> min = std::nullopt,
        std::optional<float> max = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        std::optional<Tensor> min = std::nullopt,
        std::optional<Tensor> max = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardClip {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        std::optional<float> min = std::nullopt,
        std::optional<float> max = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        std::optional<Tensor> min = std::nullopt,
        std::optional<Tensor> max = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardRdiv {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float parameter_a,
        const std::optional<string>& parameter_b = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardRepeat {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const ttnn::Shape& parameter_a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardPow {
    static std::vector<std::optional<Tensor>> invoke(
        QueueId queue_id,
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardExp {
    static std::vector<std::optional<Tensor>> invoke(
        QueueId queue_id,
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardTanh {
    static std::vector<std::optional<Tensor>> invoke(
        QueueId queue_id,
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardSqrt {
    static std::vector<std::optional<Tensor>> invoke(
        QueueId queue_id,
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardSilu {
    static std::vector<std::optional<Tensor>> invoke(
        QueueId queue_id,
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardFill {
    static std::vector<std::optional<Tensor>> invoke(
        QueueId queue_id,
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardProd {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<int64_t> dim = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteUnaryBackwardRecip {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    static std::vector<ComplexTensor> invoke(
        const ComplexTensor& grad_tensor_arg,
        const ComplexTensor& input_tensor_a_arg,
        const MemoryConfig& memory_config);
};

struct ExecuteUnaryBackwardAbs {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    static std::vector<ComplexTensor> invoke(
        const Tensor& grad_tensor_arg, const ComplexTensor& input_tensor_a_arg, const MemoryConfig& memory_config);
};

struct ExecuteUnaryBackwardGelu {
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        QueueId queue_id,
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const string& parameter_a,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

}  // namespace operations::unary_backward

constexpr auto acos_bw =
    ttnn::register_operation<"ttnn::acos_bw", operations::unary_backward::ExecuteUnaryBackwardAcos>();
constexpr auto atan_bw =
    ttnn::register_operation<"ttnn::atan_bw", operations::unary_backward::ExecuteUnaryBackwardAtan>();
constexpr auto rad2deg_bw =
    ttnn::register_operation<"ttnn::rad2deg_bw", operations::unary_backward::ExecuteUnaryBackwardRad2deg>();
constexpr auto frac_bw =
    ttnn::register_operation<"ttnn::frac_bw", operations::unary_backward::ExecuteUnaryBackwardFrac>();
constexpr auto trunc_bw =
    ttnn::register_operation<"ttnn::trunc_bw", operations::unary_backward::ExecuteUnaryBackwardTrunc>();
constexpr auto log_sigmoid_bw =
    ttnn::register_operation<"ttnn::log_sigmoid_bw", operations::unary_backward::ExecuteUnaryBackwardLogSigmoid>();
constexpr auto fill_zero_bw =
    ttnn::register_operation<"ttnn::fill_zero_bw", operations::unary_backward::ExecuteUnaryBackwardFillZero>();
constexpr auto i0_bw = ttnn::register_operation<"ttnn::i0_bw", operations::unary_backward::ExecuteUnaryBackwardI0>();
constexpr auto relu6_bw =
    ttnn::register_operation<"ttnn::relu6_bw", operations::unary_backward::ExecuteUnaryBackwardRelu6>();
constexpr auto selu_bw =
    ttnn::register_operation<"ttnn::selu_bw", operations::unary_backward::ExecuteUnaryBackwardSelu>();
constexpr auto square_bw =
    ttnn::register_operation<"ttnn::square_bw", operations::unary_backward::ExecuteUnaryBackwardSquare>();
constexpr auto tan_bw = ttnn::register_operation<"ttnn::tan_bw", operations::unary_backward::ExecuteUnaryBackwardTan>();
constexpr auto sigmoid_bw =
    ttnn::register_operation<"ttnn::sigmoid_bw", operations::unary_backward::ExecuteUnaryBackwardSigmoid>();
constexpr auto ceil_bw =
    ttnn::register_operation<"ttnn::ceil_bw", operations::unary_backward::ExecuteUnaryBackwardCeil>();
constexpr auto softsign_bw =
    ttnn::register_operation<"ttnn::softsign_bw", operations::unary_backward::ExecuteUnaryBackwardSoftsign>();
constexpr auto cosh_bw =
    ttnn::register_operation<"ttnn::cosh_bw", operations::unary_backward::ExecuteUnaryBackwardCosh>();
constexpr auto log2_bw =
    ttnn::register_operation<"ttnn::log2_bw", operations::unary_backward::ExecuteUnaryBackwardLog2>();
constexpr auto sign_bw =
    ttnn::register_operation<"ttnn::sign_bw", operations::unary_backward::ExecuteUnaryBackwardSign>();
constexpr auto exp2_bw =
    ttnn::register_operation<"ttnn::exp2_bw", operations::unary_backward::ExecuteUnaryBackwardExp2>();
constexpr auto expm1_bw =
    ttnn::register_operation<"ttnn::expm1_bw", operations::unary_backward::ExecuteUnaryBackwardExpm1>();
constexpr auto digamma_bw =
    ttnn::register_operation<"ttnn::digamma_bw", operations::unary_backward::ExecuteUnaryBackwardDigamma>();
constexpr auto erfinv_bw =
    ttnn::register_operation<"ttnn::erfinv_bw", operations::unary_backward::ExecuteUnaryBackwardErfinv>();
constexpr auto erf_bw = ttnn::register_operation<"ttnn::erf_bw", operations::unary_backward::ExecuteUnaryBackwardErf>();
constexpr auto deg2rad_bw =
    ttnn::register_operation<"ttnn::deg2rad_bw", operations::unary_backward::ExecuteUnaryBackwardDeg2rad>();
constexpr auto hardswish_bw =
    ttnn::register_operation<"ttnn::hardswish_bw", operations::unary_backward::ExecuteUnaryBackwardHardswish>();
constexpr auto tanhshrink_bw =
    ttnn::register_operation<"ttnn::tanhshrink_bw", operations::unary_backward::ExecuteUnaryBackwardTanhshrink>();
constexpr auto atanh_bw =
    ttnn::register_operation<"ttnn::atanh_bw", operations::unary_backward::ExecuteUnaryBackwardAtanh>();
constexpr auto asin_bw =
    ttnn::register_operation<"ttnn::asin_bw", operations::unary_backward::ExecuteUnaryBackwardAsin>();
constexpr auto asinh_bw =
    ttnn::register_operation<"ttnn::asinh_bw", operations::unary_backward::ExecuteUnaryBackwardAsinh>();
constexpr auto sin_bw = ttnn::register_operation<"ttnn::sin_bw", operations::unary_backward::ExecuteUnaryBackwardSin>();
constexpr auto sinh_bw =
    ttnn::register_operation<"ttnn::sinh_bw", operations::unary_backward::ExecuteUnaryBackwardSinh>();
constexpr auto log10_bw =
    ttnn::register_operation<"ttnn::log10_bw", operations::unary_backward::ExecuteUnaryBackwardLog10>();
constexpr auto log1p_bw =
    ttnn::register_operation<"ttnn::log1p_bw", operations::unary_backward::ExecuteUnaryBackwardLog1p>();
constexpr auto erfc_bw =
    ttnn::register_operation<"ttnn::erfc_bw", operations::unary_backward::ExecuteUnaryBackwardErfc>();
constexpr auto threshold_bw =
    ttnn::register_operation<"ttnn::threshold_bw", operations::unary_backward::ExecuteUnaryBackwardThreshold>();
constexpr auto fill_bw =
    ttnn::register_operation<"ttnn::fill_bw", operations::unary_backward::ExecuteUnaryBackwardFill>();
constexpr auto rsqrt_bw =
    ttnn::register_operation<"ttnn::rsqrt_bw", operations::unary_backward::ExecuteUnaryBackwardRsqrt>();
constexpr auto neg_bw = ttnn::register_operation<"ttnn::neg_bw", operations::unary_backward::ExecuteUnaryBackwardNeg>();
constexpr auto multigammaln_bw =
    ttnn::register_operation<"ttnn::multigammaln_bw", operations::unary_backward::ExecuteUnaryBackwardMultigammaln>();
constexpr auto lgamma_bw =
    ttnn::register_operation<"ttnn::lgamma_bw", operations::unary_backward::ExecuteUnaryBackwardLgamma>();
constexpr auto hardsigmoid_bw =
    ttnn::register_operation<"ttnn::hardsigmoid_bw", operations::unary_backward::ExecuteUnaryBackwardHardsigmoid>();
constexpr auto cos_bw = ttnn::register_operation<"ttnn::cos_bw", operations::unary_backward::ExecuteUnaryBackwardCos>();
constexpr auto acosh_bw =
    ttnn::register_operation<"ttnn::acosh_bw", operations::unary_backward::ExecuteUnaryBackwardAcosh>();
constexpr auto clamp_bw =
    ttnn::register_operation<"ttnn::clamp_bw", operations::unary_backward::ExecuteUnaryBackwardClamp>();
constexpr auto clip_bw =
    ttnn::register_operation<"ttnn::clip_bw", operations::unary_backward::ExecuteUnaryBackwardClip>();
constexpr auto rdiv_bw =
    ttnn::register_operation<"ttnn::rdiv_bw", operations::unary_backward::ExecuteUnaryBackwardRdiv>();
constexpr auto gelu_bw =
    ttnn::register_operation<"ttnn::gelu_bw", operations::unary_backward::ExecuteUnaryBackwardGelu>();
constexpr auto repeat_bw =
    ttnn::register_operation<"ttnn::repeat_bw", operations::unary_backward::ExecuteUnaryBackwardRepeat>();
constexpr auto pow_bw = ttnn::register_operation<"ttnn::pow_bw", operations::unary_backward::ExecuteUnaryBackwardPow>();
constexpr auto exp_bw = ttnn::register_operation<"ttnn::exp_bw", operations::unary_backward::ExecuteUnaryBackwardExp>();
constexpr auto tanh_bw =
    ttnn::register_operation<"ttnn::tanh_bw", operations::unary_backward::ExecuteUnaryBackwardTanh>();
constexpr auto sqrt_bw =
    ttnn::register_operation<"ttnn::sqrt_bw", operations::unary_backward::ExecuteUnaryBackwardSqrt>();
constexpr auto silu_bw =
    ttnn::register_operation<"ttnn::silu_bw", operations::unary_backward::ExecuteUnaryBackwardSilu>();
constexpr auto relu_bw =
    ttnn::register_operation<"ttnn::relu_bw", operations::unary_backward::ExecuteUnaryBackwardRelu>();
constexpr auto logit_bw =
    ttnn::register_operation<"ttnn::logit_bw", operations::unary_backward::ExecuteUnaryBackwardLogit>();
constexpr auto floor_bw =
    ttnn::register_operation<"ttnn::floor_bw", operations::unary_backward::ExecuteUnaryBackwardFloor>();
constexpr auto round_bw =
    ttnn::register_operation<"ttnn::round_bw", operations::unary_backward::ExecuteUnaryBackwardRound>();
constexpr auto log_bw = ttnn::register_operation<"ttnn::log_bw", operations::unary_backward::ExecuteUnaryBackwardLog>();
constexpr auto logiteps_bw =
    ttnn::register_operation<"ttnn::logiteps_bw", operations::unary_backward::ExecuteUnaryBackwardLogiteps>();
constexpr auto celu_bw =
    ttnn::register_operation<"ttnn::celu_bw", operations::unary_backward::ExecuteUnaryBackwardCelu>();
constexpr auto elu_bw = ttnn::register_operation<"ttnn::elu_bw", operations::unary_backward::ExecuteUnaryBackwardElu>();
constexpr auto leaky_relu_bw =
    ttnn::register_operation<"ttnn::leaky_relu_bw", operations::unary_backward::ExecuteUnaryBackwardLeakyRelu>();
constexpr auto softshrink_bw =
    ttnn::register_operation<"ttnn::softshrink_bw", operations::unary_backward::ExecuteUnaryBackwardSoftshrink>();
constexpr auto hardshrink_bw =
    ttnn::register_operation<"ttnn::hardshrink_bw", operations::unary_backward::ExecuteUnaryBackwardHardshrink>();
constexpr auto hardtanh_bw =
    ttnn::register_operation<"ttnn::hardtanh_bw", operations::unary_backward::ExecuteUnaryBackwardHardtanh>();
constexpr auto softplus_bw =
    ttnn::register_operation<"ttnn::softplus_bw", operations::unary_backward::ExecuteUnaryBackwardSoftplus>();
constexpr auto rpow_bw =
    ttnn::register_operation<"ttnn::rpow_bw", operations::unary_backward::ExecuteUnaryBackwardRpow>();
constexpr auto div_no_nan_bw =
    ttnn::register_operation<"ttnn::div_no_nan_bw", operations::unary_backward::ExecuteUnaryBackwardDivNoNan>();
constexpr auto polygamma_bw =
    ttnn::register_operation<"ttnn::polygamma_bw", operations::unary_backward::ExecuteUnaryBackwardPolygamma>();
constexpr auto reciprocal_bw =
    ttnn::register_operation<"ttnn::reciprocal_bw", operations::unary_backward::ExecuteUnaryBackwardRecip>();
constexpr auto abs_bw = ttnn::register_operation<"ttnn::abs_bw", operations::unary_backward::ExecuteUnaryBackwardAbs>();
constexpr auto prod_bw =
    ttnn::register_operation<"ttnn::prod_bw", operations::unary_backward::ExecuteUnaryBackwardProd>();

}  // namespace ttnn
