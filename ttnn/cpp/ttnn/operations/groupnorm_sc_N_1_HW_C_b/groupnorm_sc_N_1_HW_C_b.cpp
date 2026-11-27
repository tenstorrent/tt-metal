// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_sc_N_1_HW_C_b.hpp"

#include <stdexcept>

#include <fmt/format.h>

#include "device/groupnorm_sc_N_1_HW_C_b_device_operation.hpp"

namespace ttnn {

Tensor groupnorm_sc_N_1_HW_C_b(
    const Tensor& input_tensor,
    int num_groups,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    float eps,
    bool in_place,
    const std::optional<Tensor>& output_tensor) {
    using namespace operations::groupnorm_sc_N_1_HW_C_b;
    if (num_groups < 1) {
        throw std::invalid_argument(fmt::format(
            "groupnorm_sc_N_1_HW_C_b: num_groups={} must be >= 1 and divide C={}",
            num_groups,
            input_tensor.logical_shape()[-1]));
    }
    const GroupnormScN1HwCbParams params{
        .num_groups = static_cast<uint32_t>(num_groups),
        .eps = eps,
        .in_place = in_place,
    };
    // The framework allocates the output before it validates: refuse here, before anything is allocated.
    check_support(params, {.input = input_tensor, .gamma = gamma, .beta = beta, .output = output_tensor});
    return ttnn::prim::groupnorm_sc_N_1_HW_C_b(input_tensor, gamma, beta, output_tensor, params);
}

}  // namespace ttnn
