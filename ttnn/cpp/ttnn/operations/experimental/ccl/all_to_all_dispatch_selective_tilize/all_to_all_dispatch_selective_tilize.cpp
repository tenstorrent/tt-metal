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

ttnn::Tensor ExecuteAllToAllDispatchSelectiveTilize::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    uint32_t tokens_per_chunk,
    const std::optional<CoreRangeSet>& selective_tilize_core_range_set,
    const std::optional<CoreRangeSet>& matmul_core_range_set,
    const std::optional<CoreRangeSet>& combine_core_range_set) {
    return ttnn::prim::all_to_all_dispatch_selective_tilize(
        input_tensor,
        expert_indices_tensor,
        expert_scores_tensor,
        expert_mapping_tensor,
        axis,
        tokens_per_chunk,
        selective_tilize_core_range_set,
        matmul_core_range_set,
        combine_core_range_set);
}

}  // namespace ttnn::operations::experimental::ccl
