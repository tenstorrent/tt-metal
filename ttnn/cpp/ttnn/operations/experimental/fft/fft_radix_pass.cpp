// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/fft/fft_radix_pass.hpp"

#include "device/fft_radix_pass_device_operation.hpp"

namespace ttnn::operations::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> fft_radix_pass(
    const ttnn::Tensor& input_real,
    const std::optional<ttnn::Tensor>& input_imag,
    uint32_t P,
    uint32_t twiddle_N2,
    uint32_t stride,
    float    output_scale)
{
    return ttnn::prim::fft_radix_pass(
        input_real, input_imag, P, twiddle_N2, stride, output_scale);
}

}  // namespace ttnn::operations::experimental
