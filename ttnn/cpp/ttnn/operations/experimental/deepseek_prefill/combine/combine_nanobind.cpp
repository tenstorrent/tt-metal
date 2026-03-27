// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "combine.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine::detail {

void bind_combine(nb::module_& mod) {
    ttnn::bind_function<"combine", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Prefill combine operation for DeepSeek MoE models.

        Combines expert outputs back to original token positions.
        Uses metadata from dispatch to route expert results back to their originating tokens.
        Accumulates contributions from multiple experts per token.

        Args:
            dispatched_buffer (ttnn.Tensor): Expert outputs of shape (dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
            dispatched_metadata (ttnn.Tensor): Metadata tensor containing token routing information
            expert_token_counts (ttnn.Tensor): Counter tracking tokens per expert of shape (dispatch_group_size, experts_per_chip)

        Keyword Args:
            dispatch_group_size (int): Number of chips in the dispatch group
            experts_per_chip (int): Number of experts per chip
            num_experts_per_tok (int): Number of experts each token is routed to
            seq_len_per_chip (int): Sequence length per chip
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Defaults to None.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice ID for core allocation. Defaults to None.
            cluster_axis (int, optional): Mesh axis to operate along (0=rows, 1=cols). Defaults to 0. Currently only 0 is tested.
            num_links (int, optional): Number of ethernet links to use for fabric communication. Defaults to 1. Currently only 1 is tested.
            topology (ttnn.Topology, optional): Fabric topology (Linear or Ring). Defaults to Linear. Currently only Linear is tested.
            init_zeros (bool, optional): Whether to zero-initialize the output buffer. Defaults to True.

        Returns:
            ttnn.Tensor: Combined output tensor of shape (dispatch_group_size, seq_len_per_chip, num_experts_per_tok, hidden_dim)

        Example:
            >>> output = ttnn.experimental.deepseek_prefill.combine(
                    dispatched_buffer,
                    dispatched_metadata,
                    expert_token_counts,
                    dispatch_group_size=2,
                    experts_per_chip=8,
                    num_experts_per_tok=4,
                    seq_len_per_chip=512)
        )doc",
        &combine,
        nb::arg("dispatched_buffer").noconvert(),
        nb::arg("dispatched_metadata").noconvert(),
        nb::arg("expert_token_counts").noconvert(),
        nb::kw_only(),
        nb::arg("dispatch_group_size"),
        nb::arg("experts_per_chip"),
        nb::arg("num_experts_per_tok"),
        nb::arg("seq_len_per_chip"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("subdevice_id") = nb::none(),
        nb::arg("cluster_axis") = nb::none(),
        nb::arg("num_links") = 1,
        nb::arg("topology") = nb::cast(tt::tt_fabric::Topology::Linear),
        nb::arg("init_zeros") = true);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_combine(::nanobind::module_& mod) { combine::detail::bind_combine(mod); }

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
