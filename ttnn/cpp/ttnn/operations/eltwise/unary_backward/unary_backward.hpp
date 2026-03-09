
// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn {

namespace operations::unary_backward {

Tensor change_layout_to_tile(const Tensor& input_tensor, const MemoryConfig& output_mem_config);

struct ExecuteUnaryBackwardNeg {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardThreshold {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float threshold,
        float value,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardRpow {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float exponent,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardDivNoNan {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float scalar,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardPolygamma {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        int n,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardLog {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardRound {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardFloor {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardLogit {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardRelu {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardAcosh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardCos {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardHardsigmoid {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardLgamma {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardMultigammaln {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardSoftplus {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float beta,
        float threshold,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardHardtanh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float min,
        float max,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardHardshrink {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float lambd,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardSoftshrink {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float lambd,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardLeakyRelu {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float negative_slope,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardElu {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float alpha,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardCelu {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float alpha,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardLogiteps {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float eps,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardTan {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardSquare {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardSelu {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardRelu6 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardI0 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardFillZero {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardLogSigmoid {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardTrunc {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardFrac {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardRad2deg {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardAtan {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardAcos {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardErfc {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardErfinv {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardDigamma {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardExpm1 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardExp2 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardSign {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardLog2 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardCosh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardSoftsign {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardCeil {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardSigmoid {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardLog1p {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardLog10 {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardSinh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardSin {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardAsinh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardAsin {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardAtanh {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardTanhshrink {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardHardswish {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardDeg2rad {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardErf {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardRsqrt {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardClamp {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        std::optional<float> min = std::nullopt,
        std::optional<float> max = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        std::optional<Tensor> min = std::nullopt,
        std::optional<Tensor> max = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardClip {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        std::optional<float> min = std::nullopt,
        std::optional<float> max = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        std::optional<Tensor> min = std::nullopt,
        std::optional<Tensor> max = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardRdiv {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float scalar,
        const std::optional<std::string>& rounding_mode = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardRepeat {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const ttnn::Shape& shape,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardPow {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        float exponent,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardExp {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardTanh {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<Tensor>& input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardSqrt {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardSilu {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardFill {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<Tensor>& input_grad = std::nullopt);
};

struct ExecuteUnaryBackwardProd {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        std::optional<int64_t> dim = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteUnaryBackwardRecip {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

    static std::vector<ComplexTensor> invoke(
        const ComplexTensor& grad_tensor_arg,
        const ComplexTensor& input_tensor_a_arg,
        const MemoryConfig& output_mem_config);
};

struct ExecuteUnaryBackwardAbs {
    static std::vector<Tensor> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

    static std::vector<ComplexTensor> invoke(
        const Tensor& grad_tensor_arg, const ComplexTensor& input_tensor_a_arg, const MemoryConfig& output_mem_config);
};

struct ExecuteUnaryBackwardGelu {
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const Tensor& grad_tensor_arg,
        const Tensor& input_tensor_arg,
        const std::string& approximate,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        std::optional<Tensor> input_grad = std::nullopt);
};

}  // namespace operations::unary_backward

std::vector<std::optional<Tensor>> neg_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<Tensor> threshold_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float threshold,
    float value,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> rpow_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float exponent,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> div_no_nan_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float scalar,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> polygamma_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    int n,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> log_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> round_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> floor_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> logit_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> relu_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> acosh_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> cos_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> hardsigmoid_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> lgamma_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> multigammaln_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> softplus_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float beta,
    float threshold,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> hardtanh_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float min,
    float max,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> hardshrink_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float lambd,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> softshrink_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float lambd,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> leaky_relu_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float negative_slope,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> elu_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> celu_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float alpha,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> logiteps_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float eps,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> tan_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> square_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> selu_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> relu6_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> i0_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> fill_zero_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> log_sigmoid_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> trunc_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> frac_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> rad2deg_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> atan_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> acos_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> erfc_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> erfinv_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> digamma_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> expm1_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> exp2_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> sign_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> log2_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> cosh_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> softsign_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> ceil_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> sigmoid_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> log1p_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> log10_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> sinh_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> sin_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> asinh_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> asin_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> atanh_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> tanhshrink_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> hardswish_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> deg2rad_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> erf_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<std::optional<Tensor>> rsqrt_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<Tensor> clamp_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    std::optional<float> min = std::nullopt,
    std::optional<float> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> clamp_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    std::optional<Tensor> min = std::nullopt,
    std::optional<Tensor> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> clip_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    std::optional<float> min = std::nullopt,
    std::optional<float> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> clip_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    std::optional<Tensor> min = std::nullopt,
    std::optional<Tensor> max = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> rdiv_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float scalar,
    const std::optional<std::string>& rounding_mode = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> repeat_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const ttnn::Shape& shape,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<std::optional<Tensor>> pow_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float exponent,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<std::optional<Tensor>> exp_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<std::optional<Tensor>> tanh_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<std::optional<Tensor>> sqrt_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<std::optional<Tensor>> silu_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    std::optional<Tensor> input_grad = std::nullopt);

std::vector<std::optional<Tensor>> fill_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt);

std::vector<Tensor> prod_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    std::optional<int64_t> dim = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> reciprocal_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<ComplexTensor> reciprocal_bw(
    const ComplexTensor& grad_tensor_arg,
    const ComplexTensor& input_tensor_a_arg,
    const MemoryConfig& output_mem_config);

std::vector<Tensor> abs_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<ComplexTensor> abs_bw(
    const Tensor& grad_tensor_arg, const ComplexTensor& input_tensor_a_arg, const MemoryConfig& output_mem_config);

std::vector<std::optional<ttnn::Tensor>> gelu_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::string& approximate,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    std::optional<Tensor> input_grad = std::nullopt);

}  // namespace ttnn
