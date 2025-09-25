// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/sampling_op.hpp"
#include "ttnn/operations/reduction/sampling/sampling.hpp"

#include <utility>

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::reduction {

ttnn::Tensor SamplingOperation::invoke(
    QueueId queue_id,
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const Tensor& k,
    const Tensor& p,
    const Tensor& temp,
    const std::optional<uint32_t>& seed,
    const std::optional<CoreRangeSet>& sub_core_grids,
    std::optional<Tensor> optional_output_tensor) {
    return tt::tt_metal::operation::run(
               Sampling{seed, sub_core_grids},
               {input_values_tensor, input_indices_tensor, k, p, temp},
               {},
               {std::move(optional_output_tensor)},
               queue_id)
        .at(0);
}

}  // namespace ttnn::operations::reduction
