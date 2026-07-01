// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/fft/apply_twiddles_xl.hpp"

#include "device/apply_twiddles_xl_device_operation.hpp"

namespace ttnn::operations::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> apply_twiddles_xl(
    const ttnn::Tensor& input_real,
    const ttnn::Tensor& input_imag,
    uint32_t P,
    uint32_t big_modulus,
    uint32_t full_N)
{
    return ttnn::prim::apply_twiddles_xl(input_real, input_imag, P, big_modulus, full_N);
}

}  // namespace ttnn::operations::experimental
