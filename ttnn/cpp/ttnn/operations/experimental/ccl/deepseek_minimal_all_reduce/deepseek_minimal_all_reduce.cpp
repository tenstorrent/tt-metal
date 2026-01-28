// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_minimal_all_reduce.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/deepseek_minimal_all_reduce_device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteDeepseekMinimalAllReduce::invoke(
    const ttnn::Tensor& input_tensor,
    const uint32_t num_links,
    const ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    const std::optional<ttnn::Tensor>& intermediate_tensor,
    const std::optional<ttnn::Tensor>& residual_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_tensor) {
    return ttnn::prim::deepseek_minimal_all_reduce(
        input_tensor,
        num_links,
        topology,
        cluster_axis,
        intermediate_tensor,
        residual_tensor,
        persistent_output_tensor);
}

}  // namespace ttnn::operations::experimental::ccl
