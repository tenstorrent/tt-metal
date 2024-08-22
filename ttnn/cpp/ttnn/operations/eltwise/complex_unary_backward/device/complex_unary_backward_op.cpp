// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "complex_unary_backward_op.hpp"

#include "ttnn/operations/eltwise/complex_binary/device/complex_binary_op.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"
#include "ttnn/operations/creation.hpp"


namespace ttnn::operations::complex_unary_backward {

// polar
// grad_abs = torch.real(grad_conj * torch.sgn(result))
// result_mul_1_j = result * torch.tensor(0.0 + 1.0j)
// grad_angle = torch.real(grad_conj * result_mul_1_j)
// polar fwd op uses sin and cos hence input_b range is (0, 2*pi)
std::vector<ComplexTensor> _polar_bw(const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor result = ttnn::polar(input, output_mem_config);
    Tensor abs_result = ttnn::abs(result, output_mem_config);
    Tensor sgn_result_r = ttnn::where(ttnn::eqz(abs_result, output_mem_config), ttnn::zeros_like(result.real(), result.real().get_dtype(), result.real().get_layout(), std::nullopt, output_mem_config), ttnn::multiply(result.real(), ttnn::reciprocal(abs_result, output_mem_config), std::nullopt, output_mem_config), output_mem_config );
    Tensor sgn_result_i = ttnn::where(ttnn::eqz(abs_result, output_mem_config), ttnn::zeros_like(result.imag(), result.imag().get_dtype(), result.imag().get_layout(), std::nullopt, output_mem_config), ttnn::multiply(result.imag(), ttnn::reciprocal(abs_result, output_mem_config), std::nullopt, output_mem_config), output_mem_config );
    abs_result.deallocate();
    ComplexTensor sgn_result = ComplexTensor({ sgn_result_r, sgn_result_i });
    sgn_result_r.deallocate();
    sgn_result_i.deallocate();
    Tensor grad_abs = ttnn::real(ttnn::operations::complex_binary::_mul(ttnn::conj(grad, output_mem_config), sgn_result, output_mem_config), output_mem_config);
    sgn_result.deallocate();
    ComplexTensor flip_tensor = ComplexTensor({ttnn::zeros_like(input.real(), input.real().get_dtype(), input.real().get_layout(), std::nullopt, output_mem_config), ttnn::full_like(input.imag(), 1.0f) });
    Tensor grad_angle = ttnn::real(ttnn::operations::complex_binary::_mul(ttnn::conj(grad, output_mem_config), ttnn::operations::complex_binary::_mul(result, flip_tensor, output_mem_config), output_mem_config), output_mem_config);
    result.deallocate();
    flip_tensor.deallocate();
    ComplexTensor grad_result = ComplexTensor({grad_abs, grad_angle});
    grad_abs.deallocate();
    grad_angle.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// complex imag
// imag: at::imag(grad)
std::vector<ComplexTensor> _imag_bw(const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor real_input = ttnn::real(input, output_mem_config);
    Tensor r = ttnn::zeros_like(real_input, real_input.get_dtype(), real_input.get_layout(), std::nullopt, output_mem_config);
    ComplexTensor grad_result = ComplexTensor({r,grad});
    r.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// complex real
// real: at::real(grad)
std::vector<ComplexTensor> _real_bw(const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor real_input = ttnn::real(input, output_mem_config);
    Tensor i = ttnn::zeros_like(real_input, real_input.get_dtype(), real_input.get_layout(), std::nullopt, output_mem_config);
    ComplexTensor grad_result = ComplexTensor({grad, i});
    i.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// angle at::where(self == 0.0, at::zeros({}, self.options()), grad * self / self.abs().pow(2)
std::vector<ComplexTensor> _angle_bw(const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    const Tensor &inp_r = input.real();
    const Tensor &inp_i = input.imag();
    Tensor condition_zero = ttnn::logical_and(ttnn::eqz(input.real(),output_mem_config), ttnn::eqz(input.imag(),output_mem_config), std::nullopt, output_mem_config);
    Tensor abs_squared = ttnn::reciprocal(ttnn::add(ttnn::square(inp_r, output_mem_config), ttnn::square(inp_i, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
    Tensor res_real = ttnn::where(condition_zero, ttnn::zeros_like(inp_r, inp_r.get_dtype(), inp_r.get_layout(), std::nullopt, output_mem_config), ttnn::multiply(grad, ttnn::multiply(ttnn::neg(inp_i, output_mem_config), abs_squared, std::nullopt, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
    Tensor res_imag = ttnn::where(condition_zero, ttnn::zeros_like(inp_i, inp_i.get_dtype(), inp_i.get_layout(), std::nullopt, output_mem_config), ttnn::multiply(grad, ttnn::multiply(inp_r, abs_squared, std::nullopt, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
    condition_zero.deallocate();
    abs_squared.deallocate();
    ComplexTensor grad_result = ComplexTensor({res_real, res_imag});
    res_real.deallocate();
    res_imag.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// complex conj
// self: grad.conj()
std::vector<ComplexTensor> _conj_bw(const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_result = ttnn::conj(grad, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// complex abs
// self: grad * self.sgn()
std::vector<ComplexTensor> _complex_abs_bw(const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor result = ttnn::abs(input, output_mem_config);
    Tensor grad_inp_r = ttnn::where(ttnn::eqz(result, output_mem_config), ttnn::zeros_like(result, result.get_dtype(), result.get_layout(), std::nullopt, output_mem_config), ttnn::multiply(grad, ttnn::multiply(input.real(), ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),std::nullopt, output_mem_config), output_mem_config );
    Tensor grad_inp_i = ttnn::where(ttnn::eqz(result, output_mem_config), ttnn::zeros_like(result, result.get_dtype(), result.get_layout(), std::nullopt, output_mem_config), ttnn::multiply(grad, ttnn::multiply(input.imag(), ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),std::nullopt, output_mem_config), output_mem_config );
    ComplexTensor grad_inp = ComplexTensor({ grad_inp_r, grad_inp_i});
    result.deallocate();
    grad_inp_r.deallocate();
    grad_inp_i.deallocate();
    grad_tensor.emplace_back(grad_inp);
    return grad_tensor;
}


}  // namespace ttnn::operations::complex_unary_backward
