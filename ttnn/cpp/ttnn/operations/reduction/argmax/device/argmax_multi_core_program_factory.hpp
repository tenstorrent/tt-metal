// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::reduction::detail {

tt::tt_metal::operation::ProgramWithCallbacks argmax_multi_core(
    const Tensor& input,
    const Tensor& output,
    std::optional<uint32_t> dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::reduction::detail
