// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::ssm::detail {

operation::ProgramWithCallbacks multi_core_ssm_eltwise_mul(
    const Tensor& a,
    const Tensor& b,
    Tensor& output,
    const uint32_t hidden_size,
    MathFidelity math_fidelity,
    CoreCoord compute_with_storage_grid_size);

}  // namespace ttnn::operations::experimental::ssm::detail
