// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
        Remaps MoE routing weights from all experts to the experts of each device. A device's local experts are the
        experts that `expert_mapping_tensor` places on it, in increasing index order. For each token, the first output
        holds the `topk_tensor` weight of each local expert that the token selects in `expert_metadata_tensor`, and 0
        for the other local experts. The second output flags, for each group of `reduction_size` consecutive tokens,
        the local experts that a token of the group selects. Tokens are in ``(batch, seq)`` row-major order.

        Args:
            topk_tensor (ttnn.Tensor): the routing weights, of shape ``[1, batch, seq, experts]`` on each device: a weight for every expert, for each token.
            expert_mapping_tensor (ttnn.Tensor): the placement of experts on devices, of shape ``[1, 1, experts, devices]``, where ``devices`` is the number of devices in the mesh. Element ``[0, 0, e, d]`` is 1 when expert ``e`` is on device ``d``, counting devices in row-major mesh order, and 0 otherwise. Each device must hold ``experts / devices`` experts, so on a single device every element is 1.
            expert_metadata_tensor (ttnn.Tensor): the selected experts, of shape ``[1, batch, seq, k]`` on each device: the indices of the ``k`` experts that each token selects.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration of the outputs that are not preallocated. It must be interleaved. Defaults to `None`, which uses DRAM.
            optional_output_mapping_tensor (ttnn.Tensor, optional): Preallocated tensor for the first output. It must have that output's shape, dtype and layout, and be interleaved. Defaults to `None`.
            optional_output_reduced_tensor (ttnn.Tensor, optional): Preallocated tensor for the second output. It must have that output's shape, dtype and layout, and be interleaved. Defaults to `None`.
            reduction_size (int, optional): The number of consecutive tokens in each group of the second output. Must be greater than 0 and divide ``batch * seq``. Defaults to `16`.

        Returns:
            List[ttnn.Tensor]: two ROW_MAJOR tensors on each device. The first, of shape ``[1, batch, seq, experts / devices]`` with the dtype of `topk_tensor`, holds the weights of the local experts. The second, a UINT16 tensor of shape ``[1, 1, batch * seq / reduction_size, experts / devices]``, holds in row ``g`` a 1 for each local expert that a token of group ``g`` selects and 0 for the others.

        Note:
            Supported dtypes and layouts for `topk_tensor`:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32, UINT16
                 - ROW_MAJOR

            Supported dtypes and layouts for `expert_mapping_tensor` and `expert_metadata_tensor`:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - UINT16
                 - ROW_MAJOR

            A `topk_tensor` in DRAM needs ``experts`` times its element size to be a multiple of 64 bytes.

        Memory Support:
            - Interleaved: DRAM and L1
    )doc";

    ttnn::bind_function<"moe_expert_token_remap">(
        mod,
        doc,
        &ttnn::moe_expert_token_remap,
        nb::arg("topk_tensor").noconvert(),
        nb::arg("expert_mapping_tensor").noconvert(),
        nb::arg("expert_metadata_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("memory_config") = std::nullopt,
        nb::arg("optional_output_mapping_tensor") = std::nullopt,
        nb::arg("optional_output_reduced_tensor") = std::nullopt,
        nb::arg("reduction_size") = MoeExpertTokenRemapDeviceOperation::REDUCTION_SIZE);
}

}  // namespace ttnn::operations::data_movement::detail
