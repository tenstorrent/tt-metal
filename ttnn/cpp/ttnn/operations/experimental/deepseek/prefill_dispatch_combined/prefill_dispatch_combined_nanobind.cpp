// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_dispatch_combined_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "prefill_dispatch_combined.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::experimental::deepseek::prefill_dispatch_combined::detail {

void bind_prefill_dispatch_combined(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Combined prefill dispatch operation for DeepSeek MoE models.

        Routes tokens from their original positions to expert-specific buffers,
        sending metadata and payload in a single transfer per token-expert pair.

        The output is a single combined buffer where each row contains
        [padded_metadata | payload] instead of separate metadata and payload tensors.

        Args:
            input_tensor (ttnn.Tensor): Input tensor of shape (num_chips, seq_len, hidden_dim)
            weights_tensor (ttnn.Tensor): Router weights of shape (num_chips, seq_len, num_experts_per_tok)
            indices_tensor (ttnn.Tensor): Expert indices of shape (num_chips, seq_len, num_experts_per_tok)
            chip_to_n_routed_expert_offset_tensor (ttnn.Tensor): Base offset for each expert, shape (num_chips, n_routed_experts), dtype int32

        Keyword Args:
            num_chips (int): Number of chips in the system
            experts_per_chip (int): Number of experts per chip
            n_routed_experts (int): Total number of routed experts across all chips
            num_experts_per_tok (int): Number of experts each token is routed to
            metadata_len (int): Length of metadata per token
            max_dispatched_tokens_per_expert (int): Maximum tokens per expert
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice ID
            cluster_axis (int, optional): Mesh axis (default 0)
            num_links (int, optional): Number of ethernet links (default 1)
            topology (ttnn.Topology, optional): Fabric topology (default Linear)

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]:
                - combined: Combined metadata+payload buffer of shape
                  (experts_per_chip, max_dispatched_tokens_per_expert, padded_metadata_bf16 + hidden_dim)
                - experts_counter: Counter tracking tokens per expert of shape (experts_per_chip,)
        )doc";

    using OperationType = decltype(ttnn::prefill_dispatch_combined);
    ttnn::bind_registered_operation(
        mod,
        ttnn::prefill_dispatch_combined,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& weights_tensor,
               const ttnn::Tensor& indices_tensor,
               const ttnn::Tensor& chip_to_n_routed_expert_offset_tensor,
               uint32_t num_chips,
               uint32_t experts_per_chip,
               uint32_t n_routed_experts,
               uint32_t num_experts_per_tok,
               uint32_t metadata_len,
               uint32_t max_dispatched_tokens_per_expert,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               std::optional<uint32_t> cluster_axis,
               std::optional<uint32_t> num_links,
               std::optional<tt::tt_fabric::Topology> topology) {
                return self(
                    input_tensor,
                    weights_tensor,
                    indices_tensor,
                    chip_to_n_routed_expert_offset_tensor,
                    num_chips,
                    experts_per_chip,
                    n_routed_experts,
                    num_experts_per_tok,
                    metadata_len,
                    max_dispatched_tokens_per_expert,
                    memory_config,
                    subdevice_id,
                    cluster_axis,
                    num_links,
                    topology);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("weights_tensor").noconvert(),
            nb::arg("indices_tensor").noconvert(),
            nb::arg("chip_to_n_routed_expert_offset_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("num_chips"),
            nb::arg("experts_per_chip"),
            nb::arg("n_routed_experts"),
            nb::arg("num_experts_per_tok"),
            nb::arg("metadata_len"),
            nb::arg("max_dispatched_tokens_per_expert"),
            nb::arg("memory_config") = nb::none(),
            nb::arg("subdevice_id") = nb::none(),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("num_links") = 1,
            nb::arg("topology") = nb::cast(tt::tt_fabric::Topology::Linear)});
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_dispatch_combined::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_prefill_dispatch_combined(::nanobind::module_& mod) {
    prefill_dispatch_combined::detail::bind_prefill_dispatch_combined(mod);
}

}  // namespace ttnn::operations::experimental::deepseek::detail
