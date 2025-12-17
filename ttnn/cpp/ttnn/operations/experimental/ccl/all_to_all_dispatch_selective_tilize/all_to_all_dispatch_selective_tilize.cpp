// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_selective_tilize.hpp"
#include "device/all_to_all_dispatch_selective_tilize_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

namespace ttnn::operations::experimental::ccl {

std::array<ttnn::Tensor, 2> ExecuteAllToAllDispatchSelectiveTilize::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology,
    const std::optional<uint32_t>& output_concat_dim) {
    auto* mesh_device = input_tensor.device();

    uint32_t num_links_ = num_links.value_or(::ttnn::operations::ccl::common::get_num_links(*mesh_device, axis));
    log_debug(tt::LogOp, "num_links: {}", num_links_);
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, axis);

    return ttnn::prim::all_to_all_dispatch_selective_tilize(
        input_tensor, expert_indices_tensor, expert_scores_tensor, expert_mapping_tensor, axis, num_links_, topology_);
}

}  // namespace ttnn::operations::experimental::ccl
