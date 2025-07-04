// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::reduction::detail {

tt::tt_metal::operation::ProgramWithCallbacks topk_single_core_interleaved(
    const Tensor& input_tensor,
    uint32_t k,
    int8_t dim,
    bool largest,
    bool sorted,
    bool uint16_output,
    const CoreRangeSet& sub_core_grids,
    Tensor& value_tensor,
    Tensor& index_tensor);

tt::tt_metal::operation::ProgramWithCallbacks topk_multicore_interleaved(
    const Tensor& input_tensor,
    const std::optional<Tensor>& indices_tensor,
    uint32_t k,
    int8_t dim,
    bool largest,
    bool sorted,
    const CoreRangeSet& sub_core_grids,
    Tensor& value_tensor,
    Tensor& index_tensor);
}  // namespace ttnn::operations::reduction::detail
