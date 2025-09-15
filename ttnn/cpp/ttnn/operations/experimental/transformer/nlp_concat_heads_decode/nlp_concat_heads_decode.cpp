// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_concat_heads_decode_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "nlp_concat_heads_decode.hpp"

#include <utility>

namespace ttnn::operations::experimental::transformer {

ttnn::Tensor NLPConcatHeadsDecodeOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const uint32_t num_heads,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    bool on_subcoregrids = false;
    if (input_tensor.is_sharded()) {
        const auto& input_core_ranges = input_tensor.shard_spec().value().grid.ranges();
        if (input_core_ranges.size() > 1 || !(input_core_ranges[0].start_coord == CoreCoord{0, 0})) {
            on_subcoregrids = true;
        }
    }
    return tt::tt_metal::operation::run(
               NLPConcatHeadsDecodeDeviceOperation{num_heads, on_subcoregrids},
               {input_tensor},
               {},
               {std::move(optional_output_tensor)})
        .at(0);
}

}  // namespace ttnn::operations::experimental::transformer
