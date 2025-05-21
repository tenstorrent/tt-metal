// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::reduction::detail {

tt::tt_metal::operation::ProgramWithCallbacks topk_single_core_interleaved(
    const Tensor& input_tensor,
    const uint32_t k,
    const int8_t dim,
    const bool largest,
    const bool sorted,
    const CoreRangeSet& sub_core_grids,
    Tensor& value_tensor,
    Tensor& index_tensor);

tt::tt_metal::operation::ProgramWithCallbacks topk_multicore_interleaved(
    const Tensor& input_tensor,
    const uint32_t k,
    const int8_t dim,
    const bool largest,
    const bool sorted,
    const CoreRangeSet& sub_core_grids,
    Tensor& value_tensor,
    Tensor& index_tensor);
}  // namespace ttnn::operations::reduction::detail
