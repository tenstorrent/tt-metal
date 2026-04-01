// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/plusone_device_operation.hpp"
#include "ttnn/operations/experimental/plusone/plusone.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor plus_one(
    const Tensor& input_tensor, const std::optional<CoreRangeSet>& sub_core_grids, bool skip_negative_entries) {
    return ttnn::prim::plus_one(input_tensor, sub_core_grids, skip_negative_entries);
}

}  // namespace ttnn::operations::experimental
