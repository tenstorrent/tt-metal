// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteUntilizeWithHaloV2 {
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const ttnn::Tensor& input_tensor,
                               const Tensor& padding_config,
                               const Tensor& local_config,
                               const Tensor& remote_config,
                               const uint32_t pad_val,
                               const uint32_t ncores_nhw,
                               const uint32_t max_out_nsticks_per_core,
                               const std::optional<MemoryConfig>& memory_config,
                               const bool remote_read,
                               const bool transpose_mcast);

    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor,
                               const Tensor& padding_config,
                               const Tensor& local_config,
                               const Tensor& remote_config,
                               const uint32_t pad_val,
                               const uint32_t ncores_nhw,
                               const uint32_t max_out_nsticks_per_core,
                               const std::optional<MemoryConfig>& memory_config,
                               const bool remote_read,
                               const bool transpose_mcast);
};

}  // namespace operations::data_movement

constexpr auto untilize_with_halo_v2 =
    ttnn::register_operation_with_auto_launch_op<"ttnn::untilize_with_halo_v2",
                                                 ttnn::operations::data_movement::ExecuteUntilizeWithHaloV2>();

}  // namespace ttnn
