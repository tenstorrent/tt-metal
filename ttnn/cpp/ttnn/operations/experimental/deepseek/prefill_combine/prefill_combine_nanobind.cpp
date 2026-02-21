// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_combine_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "prefill_combine.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::operations::experimental::deepseek::prefill_combine::detail {

void bind_prefill_combine(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Prefill combine operation for DeepSeek MoE models.

        Combines expert outputs back to original token positions.
        Uses metadata from dispatch to route expert results back to their originating tokens.
        Accumulates contributions from multiple experts per token.

        Args:
            dispatched_tensor (ttnn.Tensor): Expert outputs of shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
            metadata_tensor (ttnn.Tensor): Metadata tensor containing token routing information
            experts_counter_tensor (ttnn.Tensor): Counter tracking tokens per expert of shape (num_chips, experts_per_chip)

        Keyword Args:
            num_chips (int): Number of chips in the system
            experts_per_chip (int): Number of experts per chip
            num_experts_per_tok (int): Number of experts each token is routed to
            seq_len_per_chip (int): Sequence length per chip
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Defaults to None.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice ID for core allocation. Defaults to None.

        Returns:
            ttnn.Tensor: Combined output tensor of shape (num_chips, seq_len_per_chip, num_experts_per_tok, hidden_dim)

        Example:
            >>> output = ttnn.experimental.deepseek.prefill_combine(
                    dispatched_tensor,
                    metadata_tensor,
                    experts_counter_tensor,
                    num_chips=2,
                    experts_per_chip=8,
                    num_experts_per_tok=4,
                    seq_len_per_chip=512)
        )doc";

    using OperationType = decltype(ttnn::prefill_combine);
    ttnn::bind_registered_operation(
        mod,
        ttnn::prefill_combine,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& dispatched_tensor,
               const ttnn::Tensor& metadata_tensor,
               const ttnn::Tensor& experts_counter_tensor,
               uint32_t num_chips,
               uint32_t experts_per_chip,
               uint32_t num_experts_per_tok,
               uint32_t seq_len_per_chip,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id) {
                return self(
                    dispatched_tensor,
                    metadata_tensor,
                    experts_counter_tensor,
                    num_chips,
                    experts_per_chip,
                    num_experts_per_tok,
                    seq_len_per_chip,
                    memory_config,
                    subdevice_id);
            },
            nb::arg("dispatched_tensor").noconvert(),
            nb::arg("metadata_tensor").noconvert(),
            nb::arg("experts_counter_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("num_chips"),
            nb::arg("experts_per_chip"),
            nb::arg("num_experts_per_tok"),
            nb::arg("seq_len_per_chip"),
            nb::arg("memory_config") = nb::none(),
            nb::arg("subdevice_id") = nb::none()});
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_combine::detail

namespace ttnn::operations::experimental::deepseek::detail {

void bind_prefill_combine(::nanobind::module_& mod) { prefill_combine::detail::bind_prefill_combine(mod); }

}  // namespace ttnn::operations::experimental::deepseek::detail
