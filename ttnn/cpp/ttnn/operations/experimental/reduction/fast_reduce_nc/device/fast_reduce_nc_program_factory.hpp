// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::reduction::detail {

tt::tt_metal::operation::ProgramWithCallbacks reduce_nc_factory(
    const ttnn::Tensor& input,
    const ttnn::Tensor& output,
    int64_t dim,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::operations::experimental::reduction::detail
