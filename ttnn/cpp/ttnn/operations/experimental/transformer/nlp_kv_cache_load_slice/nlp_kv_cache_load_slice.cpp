// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/nlp_kv_cache_load_slice_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "nlp_kv_cache_load_slice.hpp"

namespace ttnn::operations::experimental::transformer {

    ttnn::Tensor NLPKVCacheLoadSliceOperation::invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const uint32_t seq_len_start,
        const uint32_t seq_len_end,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor) {
            auto input_tensor_shape = input_tensor.get_shape().with_tile_padding();
            auto dim0 = input_tensor_shape[0];
            auto dim1 = input_tensor_shape[1];
            auto head_dim = input_tensor_shape[3];

            const ttnn::Shape output_tensor_start = {
                0,
                0,
                seq_len_start,
                0,
            };

            const ttnn::Shape output_tensor_end = {
                dim0-1,
                dim1-1,
                seq_len_end-1,
                head_dim-1,
            };

            const ttnn::Shape output_tensor_shape = {
                output_tensor_end[0] - output_tensor_start[0] + 1,
                output_tensor_end[1] - output_tensor_start[1] + 1,
                output_tensor_end[2] - output_tensor_start[2] + 1,
                output_tensor_end[3] - output_tensor_start[3] + 1,
            };
            return operation::run(NlpKVCacheLoadSliceDeviceOperation{output_tensor_start, output_tensor_end, output_tensor_shape, input_tensor_shape}, {input_tensor}, {}, {optional_output_tensor}).at(0);
    }

    ttnn::Tensor NLPKVCacheLoadSliceOperation::invoke(
        const Tensor& input_tensor,
        const uint32_t seq_len_start,
        const uint32_t seq_len_end,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor) {
        return invoke(
            ttnn::DefaultQueueId, input_tensor, seq_len_start, seq_len_end, memory_config, optional_output_tensor);
    }
};  // namespace ttnn::operations::experimental::transformer
