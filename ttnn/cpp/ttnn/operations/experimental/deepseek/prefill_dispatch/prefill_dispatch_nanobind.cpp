// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_dispatch_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "prefill_dispatch.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::operations::experimental::deepseek::prefill_dispatch::detail {

void bind_prefill_dispatch(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Prefill dispatch operation for DeepSeek MoE models.

        Routes tokens from their original positions to expert-specific buffers.
        Each token is routed to multiple experts based on router indices.
        Tokens are gathered into per-expert buffers with metadata tracking their origin.

        Args:
            input_tensor (ttnn.Tensor): Input tensor of shape (num_chips, seq_len, hidden_dim)
            weights_tensor (ttnn.Tensor): Router weights of shape (num_chips, seq_len, num_experts_per_tok)
            indices_tensor (ttnn.Tensor): Expert indices of shape (num_chips, seq_len, num_experts_per_tok)
            chip_to_n_routed_expert_offset_tensor (ttnn.Tensor): Base offset for each expert from each chip in the dispatched buffer, shape (num_chips, n_routed_experts), dtype int32

        Keyword Args:
            num_chips (int): Number of chips in the system
            experts_per_chip (int): Number of experts per chip
            n_routed_experts (int): Total number of routed experts across all chips
            num_experts_per_tok (int): Number of experts each token is routed to
            metadata_len (int): Length of metadata per token (stores: chip, token, topk_indice, routed_expert, weight)
            max_dispatched_tokens_per_expert (int): Maximum number of tokens that can be dispatched to each expert
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Defaults to None.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice ID for core allocation. Defaults to None.

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
                - dispatched: Dispatched tokens of shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
                - metadata: Metadata tensor of shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, metadata_len)
                - experts_counter: Counter tracking tokens per expert of shape (num_chips, experts_per_chip)

        Example:
            >>> dispatched, metadata, experts_counter = ttnn.experimental.deepseek.prefill_dispatch(
                    input_tensor,
                    weights_tensor,
                    indices_tensor,
                    chip_to_n_routed_expert_offset_tensor,
                    num_chips=2,
                    experts_per_chip=8,
                    n_routed_experts=16,
                    metadata_len=5,
                    max_dispatched_tokens_per_expert=256)
        )doc";

    using OperationType = decltype(ttnn::prefill_dispatch);
    ttnn::bind_registered_operation(
        mod,
        ttnn::prefill_dispatch,
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
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id) {
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
                    subdevice_id);
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
            nb::arg("subdevice_id") = nb::none()});
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_dispatch::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_prefill_dispatch(::nanobind::module_& mod) { prefill_dispatch::detail::bind_prefill_dispatch(mod); }

}  // namespace ttnn::operations::experimental::deepseek::detail
