// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks slice_rm_multi_core_sharded(
    const Tensor& a, Tensor& output, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end);

}  // namespace ttnn::operations::data_movement::detail