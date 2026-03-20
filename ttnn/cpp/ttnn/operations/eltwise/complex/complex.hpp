// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations::complex {

struct ComplexTensor {
    std::array<Tensor, 2> m_real_imag;

    ComplexTensor(const std::tuple<const Tensor&, const Tensor&>& real_imag);

    const Tensor& operator[](uint32_t index) const;
    const Tensor& real() const;
    const Tensor& imag() const;
    void deallocate();
};

template <std::size_t I>
const Tensor& get(const ComplexTensor&);

ComplexTensor complex_tensor(const Tensor& real, const Tensor& imag);

}  // namespace operations::complex

using ComplexTensor = operations::complex::ComplexTensor;

}  // namespace ttnn

template <>
struct std::tuple_size<ttnn::operations::complex::ComplexTensor> {
    static constexpr std::size_t value = 2;
};

template <std::size_t I>
struct std::tuple_element<I, ttnn::operations::complex::ComplexTensor> {
    using type = ttnn::Tensor;
};
