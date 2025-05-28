// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::detail {

tt::tt_metal::operation::ProgramWithCallbacks slice_write_multi_core(
    const Tensor& a,
    const Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step);

}  // namespace ttnn::operations::experimental::detail
