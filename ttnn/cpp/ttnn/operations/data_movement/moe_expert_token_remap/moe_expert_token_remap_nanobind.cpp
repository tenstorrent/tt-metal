// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_expert_token_remap_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/fabric_edm_types.hpp>

#include "moe_expert_token_remap.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_moe_expert_token_remap(nb::module_& mod) {
    auto doc = R"doc(

Remap MoE CCL Metadata from global experts to local device experts

Args:
    topk_tensor (ttnn.Tensor): tensor of MoE topk scores, `[devices/devices, batch, seq, experts]`
    expert_mapping_tensor (ttnn.Tensor): tensor that maps MoE experts to devices, `[1, 1, experts, devices]`
    expert_metadata_tensor (ttnn.Tensor): tensor that maps tokens to global experts `[devices/devices, batch, seq, select_experts_k]``

Keyword Args:
    memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
    output_mapping_tensor (ttnn.Tensor, optional): Preallocated output mapping tensor. Defaults to `None`.
    output_reduced_tensor (ttnn.Tensor, optional): Preallocated output reduced tensor. Defaults to `None`.
    reduction_size (int, optional): reduction chunk size

Returns:
    Tuple:
    ttnn.Tensor: Tensor that maps batch tokens to local experts, `[devices/devices, batch, seq, experts_per_device]`
    ttnn.Tensor: Bool Tensor that reduces the mapping tensor by chunks of `reduction_size`, `[devices/devices, batch*seq/reduction_size, experts_per_device]`

    )doc";

    using OperationType = decltype(ttnn::moe_expert_token_remap);
    ttnn::bind_registered_operation(
        mod,
        ttnn::moe_expert_token_remap,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& topk_tensor,
               const ttnn::Tensor& expert_mapping_tensor,
               const ttnn::Tensor& expert_metadata_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& optional_output_mapping_tensor,
               const std::optional<ttnn::Tensor>& optional_output_reduced_tensor,
               const uint32_t reduction_size) {
                return self(
                    topk_tensor,
                    expert_mapping_tensor,
                    expert_metadata_tensor,
                    memory_config,
                    optional_output_mapping_tensor,
                    optional_output_reduced_tensor,
                    reduction_size);
            },
            nb::arg("topk_tensor").noconvert(),
            nb::arg("expert_indices_tensor").noconvert(),
            nb::arg("expert_mapping_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = std::nullopt,
            nb::arg("optional_output_mapping_tensor") = std::nullopt,
            nb::arg("optional_output_reduced_tensor") = std::nullopt,
            nb::arg("reduction_size") = ExecuteMoeExpertTokenRemap::REDUCTION_SIZE});
}

}  // namespace ttnn::operations::data_movement::detail
