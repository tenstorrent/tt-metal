// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_expert_token_remap_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "moe_expert_token_remap.hpp"
#include "device/moe_expert_token_remap_device_operation.hpp"

namespace ttnn::operations::data_movement::detail {

void bind_moe_expert_token_remap(nb::module_& mod) {
    const auto* doc = R"doc(

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

    ttnn::bind_function<"moe_expert_token_remap">(
        mod,
        doc,
        ttnn::overload_t(
            &ttnn::moe_expert_token_remap,
            nb::arg("topk_tensor").noconvert(),
            nb::arg("expert_mapping_tensor").noconvert(),
            nb::arg("expert_metadata_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = std::nullopt,
            nb::arg("optional_output_mapping_tensor") = std::nullopt,
            nb::arg("optional_output_reduced_tensor") = std::nullopt,
            nb::arg("reduction_size") = MoeExpertTokenRemapDeviceOperation::REDUCTION_SIZE));
}

}  // namespace ttnn::operations::data_movement::detail
