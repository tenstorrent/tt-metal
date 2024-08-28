// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "complex.hpp"


namespace ttnn {
namespace operations::complex {


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


ComplexTensor CreateComplexTensor::invoke(
        const Tensor &real,
        const Tensor &imag) {
            TT_ASSERT(real.get_legacy_shape() == imag.get_legacy_shape() , "Tensor shapes of real and imag should be identical");
            return ComplexTensor({real, imag});
    }

}  // namespace operations::complex

} // namespace ttnn
