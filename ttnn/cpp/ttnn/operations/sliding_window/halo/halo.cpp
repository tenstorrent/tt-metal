// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "halo.hpp"

#include <utility>
#include "device/halo_device_operation.hpp"
namespace ttnn::operations::sliding_window::halo {
Tensor HaloOperation::invoke(
    const Tensor& input_tensor,
    const SlidingWindowConfig& config,
    uint32_t pad_val,
    bool remote_read,
    bool transpose_mcast,
    const MemoryConfig& output_memory_config,
    bool is_out_tiled,
    bool in_place,
    bool config_tensors_in_dram) {
    return halo_op(
        input_tensor,
        config,
        pad_val,
        remote_read,
        transpose_mcast,
        output_memory_config,
        is_out_tiled,
        in_place,
        config_tensors_in_dram);
}
};  // namespace ttnn::operations::sliding_window::halo
