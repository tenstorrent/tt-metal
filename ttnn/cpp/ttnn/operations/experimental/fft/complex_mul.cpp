// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/fft/complex_mul.hpp"

#include "device/complex_mul_device_operation.hpp"

namespace ttnn::operations::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> complex_mul(
    const ttnn::Tensor& a_real,
    const ttnn::Tensor& a_imag,
    const ttnn::Tensor& b_real,
    const ttnn::Tensor& b_imag)
{
    return ttnn::prim::complex_mul(a_real, a_imag, b_real, b_imag);
}

}  // namespace ttnn::operations::experimental
