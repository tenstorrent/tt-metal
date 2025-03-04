// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::reduction::detail {

using namespace tt::constants;

tt::tt_metal::operation::ProgramWithCallbacks argmax_single_core(
    const Tensor& input, const Tensor& output, const std::optional<uint32_t> dim);

tt::tt_metal::operation::ProgramWithCallbacks argmax_multi_core(
    const Tensor& input,
    const Tensor& output,
    const std::optional<uint32_t> dim,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::reduction::detail
