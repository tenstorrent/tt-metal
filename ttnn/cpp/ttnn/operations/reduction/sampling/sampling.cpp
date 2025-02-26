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
    const std::vector<uint16_t>& k,
    const std::vector<float>& p,
    const uint32_t seed,
    const std::optional<CoreRangeSet>& sub_core_grids,
    std::optional<Tensor> optional_output_tensor) {
    return tt::tt_metal::operation::run(
               Sampling{k, p, seed, sub_core_grids},
               {input_values_tensor, input_indices_tensor},
               {},
               {std::move(optional_output_tensor)},
               queue_id)
        .at(0);
}

ttnn::Tensor SamplingOperation::invoke(
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const std::vector<uint16_t>& k,
    const std::vector<float>& p,
    const uint32_t seed,
    const std::optional<CoreRangeSet>& sub_core_grids,
    std::optional<Tensor> optional_output_tensor) {
    return invoke(
        DefaultQueueId,
        input_values_tensor,
        input_indices_tensor,
        k,
        p,
        seed,
        sub_core_grids,
        std::move(optional_output_tensor));
}

}  // namespace ttnn::operations::reduction
