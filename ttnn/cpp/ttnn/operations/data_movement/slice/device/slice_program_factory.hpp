// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "slice_rm_multi_core.hpp"
#include "slice_rm_strided_single_core_n_dims.hpp"
#include "slice_rm_multi_core_sharded.hpp"
#include "slice_tile_multi_core.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks slice_multi_core(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step);

}  // namespace ttnn::operations::data_movement::detail
