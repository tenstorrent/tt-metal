// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/fft/apply_twiddles.hpp"

#include "device/apply_twiddles_device_operation.hpp"

namespace ttnn::operations::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> apply_twiddles(
    const ttnn::Tensor& input_real,
    const ttnn::Tensor& input_imag,
    uint32_t N1,
    uint32_t N2)
{
    return ttnn::prim::apply_twiddles(input_real, input_imag, N1, N2);
}

}  // namespace ttnn::operations::experimental
