// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace operations::transformer::sdpa_prefill {

struct ExecuteDistributedMLA {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::transformer::sdpa_prefill

namespace prim {
ttnn::Tensor distributed_mla(
    const ttnn::Tensor& input_tensor, std::optional<uint32_t> cluster_axis, const ttnn::MemoryConfig& memory_config);
}  // namespace prim

namespace transformer::sdpa_prefill {

constexpr auto distributed_mla = ttnn::register_operation<
    "ttnn::transformer::sdpa_prefill::distributed_mla",
    ttnn::operations::transformer::sdpa_prefill::ExecuteDistributedMLA>();

}  // namespace transformer::sdpa_prefill

}  // namespace ttnn
