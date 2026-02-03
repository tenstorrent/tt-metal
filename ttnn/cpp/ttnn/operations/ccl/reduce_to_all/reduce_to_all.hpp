// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#pragma once
#include <tt-metalium/mesh_coord.hpp>

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace operations::ccl {

struct ExecuteReduceToAll {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_l,
        const ttnn::Tensor& input_tensor_ms,  // Combined: col 0 = max, col 1 = sum
        float scale_fp32,
        tt::tt_fabric::Topology topology,
        const std::optional<ttnn::Tensor>& optional_output_tensor_l = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_fw_intermediate_tensor = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_bw_intermediate_tensor = std::nullopt,
        const std::optional<std::vector<ttnn::CoreCoord>>& input_forwarder_cores = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_forwarder_scratch_tensor = std::nullopt);
};

ttnn::TensorSpec reduce_to_all_tensor_spec(
    const ttnn::Tensor& input_tensor_l,
    const ttnn::Tensor& input_tensor_ms,  // Combined: col 0 = max, col 1 = sum
    float scale_fp32,
    tt::tt_fabric::Topology topology,
    const std::optional<std::vector<ttnn::CoreCoord>>& input_forwarder_cores);

}  // namespace operations::ccl

constexpr auto reduce_to_all =
    ttnn::register_operation<"ttnn::reduce_to_all", ttnn::operations::ccl::ExecuteReduceToAll>();

}  // namespace ttnn
