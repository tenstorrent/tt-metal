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

struct ExecuteReduceToRoot {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor_l,
        const ttnn::Tensor& input_tensor_s,
        const ttnn::Tensor& input_tensor_m,
        const MeshCoordinate& root_coord,
        float scale_fp32,
        tt::tt_fabric::Topology topology,
        const std::optional<ttnn::Tensor>& optional_output_tensor_l = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor_s = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor_m = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_intermediate_tensor = std::nullopt,
        const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores = std::nullopt);
};

std::vector<ttnn::TensorSpec> reduce_to_root_tensor_spec(
    const ttnn::Tensor& input_tensor_l,
    const ttnn::Tensor& input_tensor_s,
    const ttnn::Tensor& input_tensor_m,
    const MeshCoordinate& root_coord,
    float scale_fp32,
    tt::tt_fabric::Topology topology,
    const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores);

}  // namespace operations::ccl

constexpr auto reduce_to_root =
    ttnn::register_operation<"ttnn::reduce_to_root", ttnn::operations::ccl::ExecuteReduceToRoot>();

}  // namespace ttnn
