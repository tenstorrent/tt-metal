// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "halo.hpp"

#include <utility>
#include "device/halo_device_operation.hpp"
namespace ttnn::operations::sliding_window::halo {
Tensor HaloOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const SlidingWindowConfig& config,
    uint32_t pad_val,
    bool remote_read,
    bool transpose_mcast,
    uint32_t reshard_num_cores_nhw,
    const MemoryConfig& output_memory_config,
    bool is_out_tiled) {
    return halo_op(
        input_tensor,
        config,
        pad_val,
        remote_read,
        transpose_mcast,
        reshard_num_cores_nhw,
        std::move(output_memory_config),
        is_out_tiled);
}
};  // namespace ttnn::operations::sliding_window::halo
