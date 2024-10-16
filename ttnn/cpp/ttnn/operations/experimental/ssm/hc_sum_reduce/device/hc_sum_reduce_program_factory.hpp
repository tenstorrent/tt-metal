// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::ssm::detail {

operation::ProgramWithCallbacks multi_core_ssm_1d_sum_reduce(const Tensor& a,
                                                             Tensor& output,
                                                             MathFidelity math_fidelity,
                                                             CoreCoord compute_with_storage_grid_size);

}  // namespace ttnn::operations::experimental::ssm::detail
