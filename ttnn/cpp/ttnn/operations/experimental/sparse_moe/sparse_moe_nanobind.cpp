// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "sparse_moe_nanobind.hpp"
#include <cstdint>
#include <nanobind/nanobind.h>
#include "ttnn/operations/experimental/sparse_moe/sparse_moe.hpp"

namespace ttnn::operations::experimental::sparse_moe::detail {

void bind_sparse_moe_expert(nb::module_& mod) {
    mod.def(
        "sparse_moe_expert",
        &ttnn::experimental::sparse_moe_expert,
        nb::arg("input"),
        nb::arg("expert_gu"),
        nb::arg("expert_dw"),
        nb::arg("expert_mask"),
        nb::kw_only(),
        nb::arg("num_experts"),
        nb::arg("expert_inter_dim"),
        nb::arg("hidden_dim"),
        nb::arg("batch_size"),
        "Sparse MoE expert matmul: reads only active expert weights from DRAM.");
}

}  // namespace ttnn::operations::experimental::sparse_moe::detail
