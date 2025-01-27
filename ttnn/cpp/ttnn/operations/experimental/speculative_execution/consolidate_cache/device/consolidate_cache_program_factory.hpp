// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn::operations::experimental::speculative_execution::detail {

operation::ProgramWithCallbacks consolidate_cache(
    const Tensor& input_tensor,
    const Tensor& other_tensor,
    const Tensor& priority_tensor,
    const Tensor& other_priority_tensor,
    const Tensor& output_tensor);

}  // namespace ttnn::operations::experimental::speculative_execution::detail
