// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::detail {

using namespace tt::constants;

tt::tt_metal::operation::ProgramWithCallbacks plusone_single_core(
    const Tensor& input, const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::experimental::detail
