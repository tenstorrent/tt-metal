// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/multi_device.hpp"
#include "ttnn/operations/ccl/ccl_fabric.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteAllGather::invoke(const ttnn::Tensor& input_tensor, const uint32_t dim, const uint32_t num_links, const std::optional<ttnn::MemoryConfig>& memory_config, ttnn::ccl::OpFabricMode fabric_mode) {
    return ttnn::operations::ccl::all_gather(input_tensor, dim, num_links, memory_config, fabric_mode);
}

}  // namespace ttnn::operations::ccl
