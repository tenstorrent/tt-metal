// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_unary_backward_op.hpp"

#include "ttnn/operations/eltwise/complex_binary/device/complex_binary_op.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite_op.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/creation/creation.hpp"

namespace ttnn::operations::complex_unary_backward {

// polar
// z = r * cos(theta) + i * r * sin(theta)
// grad_abs = Re(conj(grad) * sgn(z)) = sign(r) * (grad_r * cos(theta) + grad_i * sin(theta))
// grad_angle = Re(conj(grad) * i * z) = r * (grad_i * cos(theta) - grad_r * sin(theta))
std::vector<ComplexTensor> _polar_bw(
    const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;

    const Tensor& r = input.real();
    const Tensor& theta = input.imag();
    const Tensor& grad_r = grad.real();
    const Tensor& grad_i = grad.imag();

    Tensor cos_theta = ttnn::cos(theta, output_mem_config);
    Tensor sin_theta = ttnn::sin(theta, output_mem_config);

    Tensor unit_proj = ttnn::add(
        ttnn::multiply(grad_r, cos_theta, std::nullopt, output_mem_config),
        ttnn::multiply(grad_i, sin_theta, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    Tensor sign_r = ttnn::sign(r, output_mem_config);
    Tensor grad_abs = ttnn::multiply(unit_proj, sign_r, std::nullopt, output_mem_config);
    sign_r.deallocate();
    unit_proj.deallocate();

    Tensor ortho_proj = ttnn::subtract(
        ttnn::multiply(grad_i, cos_theta, std::nullopt, output_mem_config),
        ttnn::multiply(grad_r, sin_theta, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    cos_theta.deallocate();
    sin_theta.deallocate();
    Tensor grad_angle = ttnn::multiply(r, ortho_proj, std::nullopt, output_mem_config);
    ortho_proj.deallocate();

    ComplexTensor grad_result = ComplexTensor({grad_abs, grad_angle});
    grad_abs.deallocate();
    grad_angle.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// complex imag
// imag: at::imag(grad)
std::vector<ComplexTensor> _imag_bw(
    const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    // Only a template for the zeros below, so take the component directly: ttnn::real would copy it
    // when the caller's config differs from the component's own.
    const Tensor& real_input = input.real();
    Tensor r = ttnn::zeros_like(real_input, real_input.dtype(), real_input.layout(), std::nullopt, output_mem_config);
    ComplexTensor grad_result = ComplexTensor({r, grad});
    r.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// complex real
// real: at::real(grad)
std::vector<ComplexTensor> _real_bw(
    const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    // As in _imag_bw: a zeros template only, so no need to place a copy of the component.
    const Tensor& real_input = input.real();
    Tensor i = ttnn::zeros_like(real_input, real_input.dtype(), real_input.layout(), std::nullopt, output_mem_config);
    ComplexTensor grad_result = ComplexTensor({grad, i});
    i.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// angle: at::where(self == 0.0, at::zeros({}, self.options()), grad * (-y, x) / (x^2 + y^2))
// Evaluated via Smith's scaled division so |z| > 2^63 and 0 < |z| < 2^-63 stay in normal range.
std::vector<ComplexTensor> _angle_bw(
    const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;

    const Tensor& inp_r = input.real();
    const Tensor& inp_i = input.imag();
    Tensor condition_zero = ttnn::logical_and(
        ttnn::eqz(inp_r, output_mem_config),
        ttnn::eqz(inp_i, output_mem_config),
        std::nullopt,
        output_mem_config);
    Tensor abs_r = ttnn::abs(inp_r, output_mem_config);
    Tensor abs_i = ttnn::abs(inp_i, output_mem_config);
    Tensor r_ge_i = ttnn::ge(abs_r, abs_i, std::nullopt, output_mem_config);
    abs_r.deallocate();
    abs_i.deallocate();

    Tensor big = ttnn::where(r_ge_i, inp_r, inp_i, output_mem_config);
    Tensor small = ttnn::where(r_ge_i, inp_i, inp_r, output_mem_config);
    Tensor ratio = ttnn::divide(small, big, std::nullopt, output_mem_config);
    Tensor denom =
        ttnn::add(big, ttnn::multiply(small, ratio, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    big.deallocate();
    small.deallocate();
    Tensor scaled_grad = ttnn::divide(grad, denom, std::nullopt, output_mem_config);
    denom.deallocate();
    Tensor scaled_ratio = ttnn::multiply(scaled_grad, ratio, std::nullopt, output_mem_config);
    ratio.deallocate();

    Tensor res_real = ttnn::where(
        condition_zero,
        0.0f,
        ttnn::neg(ttnn::where(r_ge_i, scaled_ratio, scaled_grad, output_mem_config), output_mem_config),
        output_mem_config);
    Tensor res_imag = ttnn::where(
        condition_zero,
        0.0f,
        ttnn::where(r_ge_i, scaled_grad, scaled_ratio, output_mem_config),
        output_mem_config);
    condition_zero.deallocate();
    r_ge_i.deallocate();
    scaled_grad.deallocate();
    scaled_ratio.deallocate();
    ComplexTensor grad_result = ComplexTensor({res_real, res_imag});
    res_real.deallocate();
    res_imag.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// complex conj
// self: grad.conj()
std::vector<ComplexTensor> _conj_bw(
    const ComplexTensor& grad, const ComplexTensor& /*input*/, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_result = ttnn::conj(grad, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// complex abs
// self: grad * self.sgn()
std::vector<ComplexTensor> _complex_abs_bw(
    const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor result = ttnn::abs(input, output_mem_config);
    Tensor grad_inp_r = ttnn::where(
        ttnn::eqz(result, output_mem_config),
        0.0f,
        ttnn::multiply(
            grad,
            ttnn::multiply(input.real(), ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    Tensor grad_inp_i = ttnn::where(
        ttnn::eqz(result, output_mem_config),
        0.0f,
        ttnn::multiply(
            grad,
            ttnn::multiply(input.imag(), ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    ComplexTensor grad_inp = ComplexTensor({grad_inp_r, grad_inp_i});
    result.deallocate();
    grad_inp_r.deallocate();
    grad_inp_i.deallocate();
    grad_tensor.emplace_back(grad_inp);
    return grad_tensor;
}

}  // namespace ttnn::operations::complex_unary_backward
