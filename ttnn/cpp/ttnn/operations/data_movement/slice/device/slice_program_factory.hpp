// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks slice_multi_core(
    const Tensor& a,
    Tensor& output,
    const ttnn::SimpleShape& output_tensor_start,
    const ttnn::SimpleShape& output_tensor_end,
    const ttnn::SimpleShape& step);

}  // namespace ttnn::operations::data_movement::detail
