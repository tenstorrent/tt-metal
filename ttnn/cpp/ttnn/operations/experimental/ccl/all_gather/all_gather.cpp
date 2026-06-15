// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather.hpp"

#include "ttnn/operations/experimental/ccl/all_gather/device/all_gather_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::experimental {

ttnn::Tensor all_gather(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // TODO fix this
    // if (composite_common::use_composite_all_gather(input_tensor, gather_dim, memory_config)) {
    if (false) {
        log_info(
            tt::LogOp,
            "Using slower composite all_gather since input_tensor has unaligned pages / padded tiles on gather dim "
            "xxx");

        // Query the Fabric setup
        auto* mesh_device = input_tensor.device();
        TT_FATAL(mesh_device != nullptr, "Input tensor should be on device for all_gather operation");
        uint32_t num_links = ttnn::operations::ccl::common::get_num_links(*mesh_device, cluster_axis);

        return composite_common::composite_all_gather(
            input_tensor, dim, num_links, memory_config, /*subdevice_id*/ std::nullopt, cluster_axis);
    }

    return ttnn::prim::all_gather_experimental(
        input_tensor, persistent_output_tensor, dim, memory_config, cluster_axis, subdevice_id, sub_core_grid);
}

}  // namespace ttnn::experimental
