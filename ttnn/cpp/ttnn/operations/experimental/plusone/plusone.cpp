// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/container/vector.hpp>
#include <algorithm>
#include <vector>

#include "device/plusone_op.hpp"
#include <tt-metalium/buffer_constants.hpp>
#include "ttnn/operations/experimental/plusone/plusone.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor PlusOneOperation::invoke(
    QueueId queue_id, const Tensor& input_tensor, const std::optional<CoreRangeSet>& sub_core_grids) {
    return tt::tt_metal::operation::run(PlusOne{sub_core_grids}, {input_tensor}, {}, {}, queue_id).at(0);
}

}  // namespace ttnn::operations::experimental
