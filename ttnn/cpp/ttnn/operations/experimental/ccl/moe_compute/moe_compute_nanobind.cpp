// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "moe_compute_nanobind.hpp"
#include "moe_compute.hpp"

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_moe_compute(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& tilize_input_tensor,
               const ttnn::Tensor& tilize_expert_indices_tensor,
               const ttnn::Tensor& tilize_expert_scores_tensor,
               const ttnn::Tensor& tilize_expert_mapping_tensor,
               const ttnn::Tensor& matmul_w0_w1_tensor,
               const ttnn::Tensor& matmul_w2_tensor,
               const uint32_t layer_id,
               const uint32_t output_height_shard_dim,
               const uint32_t output_width_shard_dim,
               const std::vector<ttnn::CoreCoord>& output_shard_cores,
               const std::optional<uint32_t> cluster_axis) {
                return self(
                    tilize_input_tensor,
                    tilize_expert_indices_tensor,
                    tilize_expert_scores_tensor,
                    tilize_expert_mapping_tensor,
                    matmul_w0_w1_tensor,
                    matmul_w2_tensor,
                    layer_id,
                    output_height_shard_dim,
                    output_width_shard_dim,
                    output_shard_cores,
                    cluster_axis);
            },
            nb::arg("tilize_input_tensor").noconvert(),
            nb::arg("tilize_expert_indices_tensor").noconvert(),
            nb::arg("tilize_expert_scores_tensor").noconvert(),
            nb::arg("tilize_expert_mapping_tensor").noconvert(),
            nb::arg("matmul_w0_w1_tensor").noconvert(),
            nb::arg("matmul_w2_tensor").noconvert(),
            nb::arg("layer_id"),
            nb::arg("output_height_shard_dim"),
            nb::arg("output_width_shard_dim"),
            nb::arg("output_shard_cores"),
            nb::kw_only(),
            nb::arg("cluster_axis") = nb::none()});
}

}  // namespace

void bind_moe_compute(nb::module_& mod) {
    bind_moe_compute(
        mod,
        ttnn::experimental::moe_compute,
        R"doc(
        Experimental, high-performance MoE operation for DeepSeek.
        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
