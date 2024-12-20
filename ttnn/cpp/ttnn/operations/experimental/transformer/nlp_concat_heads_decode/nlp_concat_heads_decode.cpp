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
    uint8_t queue_id,
    const Tensor& input_tensor,
    const uint32_t num_heads,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    bool on_subcoregrids_val = false;
    if (input_tensor.is_sharded()) {
        const auto& input_core_ranges = input_tensor.shard_spec().value().grid.ranges();
        if (input_core_ranges.size() > 1 || !(input_core_ranges[0].start_coord == CoreCoord{0, 0})) {
            on_subcoregrids_val = true;
        }
    }
    const bool on_subcoregrids = on_subcoregrids_val;

    return operation::run(
               NLPConcatHeadsDecodeDeviceOperation{num_heads, on_subcoregrids},
               {input_tensor},
               {},
               {std::move(optional_output_tensor)})
        .at(0);
}

ttnn::Tensor NLPConcatHeadsDecodeOperation::invoke(
    const Tensor& input_tensor,
    const uint32_t num_heads,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return invoke(ttnn::DefaultQueueId, input_tensor, num_heads, memory_config, std::move(optional_output_tensor));
}
};  // namespace ttnn::operations::experimental::transformer
