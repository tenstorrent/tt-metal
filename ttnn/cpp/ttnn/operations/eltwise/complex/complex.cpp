// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "complex.hpp"


namespace ttnn {
namespace operations::complex {


ComplexTensor::ComplexTensor(std::array<Tensor, 2> val): m_real_imag(val) {
            TT_ASSERT( m_real_imag[0].get_legacy_shape() == m_real_imag[1].get_legacy_shape() , "Tensor shapes of real and imag should be identical");
        }

const Tensor& ComplexTensor::operator[](uint32_t index) const {
            return m_real_imag[index];
        }

const Tensor& ComplexTensor::real() const {
            return m_real_imag[0];
        }

const Tensor& ComplexTensor::imag() const {
            return m_real_imag[1];
        }

void ComplexTensor::deallocate() {
            m_real_imag[0].deallocate();
            m_real_imag[1].deallocate();
        }


ComplexTensor CreateComplexTensor::operator()(
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg) {
            return ComplexTensor({input_tensor_a_arg, input_tensor_b_arg});
    }

}  // namespace operations::complex

} // namespace ttnn
