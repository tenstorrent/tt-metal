// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_halo_v2.hpp"

#include "device/untilize_with_halo_v2_op.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor ExecuteUntilizeWithHaloV2::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const Tensor& padding_config1,
    const Tensor& padding_config2,
    const Tensor& local_config1,
    const Tensor& local_config2,
    const Tensor& remote_config1,
    const Tensor& remote_config2,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t max_out_nsticks_per_core,
    const std::optional<MemoryConfig>& memory_config,
    const bool remote_read,
    const bool transpose_mcast) {
    TT_ASSERT(input_tensor.memory_config().is_sharded());
    TT_ASSERT(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);

    return operation::run(
               UntilizeWithHaloV2{
                   pad_val,
                   ncores_nhw,
                   max_out_nsticks_per_core,
                   memory_config.value_or(input_tensor.memory_config()),
                   remote_read,
                   transpose_mcast},
               {input_tensor,
                padding_config1,
                padding_config2,
                local_config1,
                local_config2,
                remote_config1,
                remote_config2},
               {},
               {},
               queue_id)
        .at(0);
}

ttnn::Tensor ExecuteUntilizeWithHaloV2::invoke(
    const ttnn::Tensor& input_tensor,
    const Tensor& padding_config1,
    const Tensor& padding_config2,
    const Tensor& local_config1,
    const Tensor& local_config2,
    const Tensor& remote_config1,
    const Tensor& remote_config2,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t max_out_nsticks_per_core,
    const std::optional<MemoryConfig>& memory_config,
    const bool remote_read,
    const bool transpose_mcast) {
    return invoke(
        DefaultQueueId,
        input_tensor,
        padding_config1,
        padding_config2,
        local_config1,
        local_config2,
        remote_config1,
        remote_config2,
        pad_val,
        ncores_nhw,
        max_out_nsticks_per_core,
        memory_config,
        remote_read,
        transpose_mcast);
}

}  // namespace ttnn::operations::data_movement
