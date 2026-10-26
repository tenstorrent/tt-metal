// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hybrid_routed_expert_ffn_nanobind.hpp"

#include <limits>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "hybrid_routed_expert_ffn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::detail {

void bind_hybrid_routed_expert_ffn(nb::module_& mod) {
    // RoutedExpertActivation is already registered by unified_routed_expert_ffn, whose header this
    // op includes rather than redeclaring the enum. Binding it a second time would make a distinct
    // Python type that no caster accepts, which fails at import rather than at the call.
    // That is not hypothetical -- redeclaring it here is what it took to learn.
    ttnn::bind_function<"hybrid_routed_expert_moe", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Both routed-expert implementations in ONE dispatch.

        Each expert is routed at runtime from the device-resident token counts against
        hybrid_token_threshold: at or below it the expert runs on the fused implementation,
        above it on the unified one. One program, so the layer can be overlapped with combine.

        Passing dispatched_metadata overlaps combine_fabric2d in the same program and returns
        combine's output. dispatched_buffer must then be bfloat16 ROW_MAJOR (the routed expert hands
        combine bfloat8_b tiles, as the two ops do back to back), and expert_offsets,
        replicated_global_expert_idx_table (the full table on every device), expert_region_offsets,
        combine_axis, combine_num_links, num_experts_per_tok and seq_len_per_chip are combine's.
        The overlap also needs fwd_arrived_semaphore, final_arrived_semaphore, expert_go_semaphore and
        channel_semaphores (hybrid_routed_expert_moe_channel_semaphore_count of them): global semaphores
        created once, initialised to zero, and passed on every call, since neighbouring chips bump them
        across launches and the routed expert's L1 arena leaves nothing to allocate them from by the time
        combine is built. Everything else it places in L1 is freed when
        the call returns. forwarding_buffer is combine's DRAM staging buffer from
        hybrid_routed_expert_moe_forwarding_buffer; omitted, combine allocates its own at the first
        call and keeps it for the program's lifetime.
        )doc",
        &hybrid_routed_expert_moe,
        nb::arg("dispatched_buffer").noconvert(),
        nb::arg("expert_region_offsets").noconvert(),
        nb::arg("expert_token_counts").noconvert(),
        nb::arg("global_expert_idx_table").noconvert(),
        nb::arg("gate_projs").noconvert(),
        nb::arg("up_projs").noconvert(),
        nb::arg("down_projs").noconvert(),
        nb::arg("max_dispatched_tokens_per_expert"),
        nb::kw_only(),
        nb::arg("hybrid_token_threshold") = 0,
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("activation") = RoutedExpertActivation::Silu,
        nb::arg("gate_biases") = nb::none(),
        nb::arg("up_biases") = nb::none(),
        nb::arg("down_biases") = nb::none(),
        nb::arg("dispatched_metadata") = nb::none(),
        nb::arg("expert_offsets") = nb::none(),
        nb::arg("replicated_global_expert_idx_table") = nb::none(),
        nb::arg("combine_axis") = 0,
        nb::arg("combine_num_links") = 2,
        nb::arg("num_experts_per_tok") = 0,
        nb::arg("seq_len_per_chip") = 0,
        nb::arg("fwd_arrived_semaphore") = nb::none(),
        nb::arg("final_arrived_semaphore") = nb::none(),
        nb::arg("expert_go_semaphore") = nb::none(),
        nb::arg("forwarding_buffer") = nb::none(),
        nb::arg("channel_semaphores") = std::vector<GlobalSemaphore>{});

    ttnn::bind_function<
        "hybrid_routed_expert_moe_channel_semaphore_count",
        "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        How many channel_semaphores hybrid_routed_expert_moe needs when overlapped with combine.

        Create that many global semaphores once per mesh, initialised to zero, before the first
        overlapped forward, and pass the same list on every call.
        )doc",
        &hybrid_routed_expert_moe_channel_semaphore_count,
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("combine_axis") = 0);

    ttnn::bind_function<"hybrid_routed_expert_moe_forwarding_buffer", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        Allocate combine's DRAM forwarding buffer for hybrid_routed_expert_moe overlapped with combine.

        Create it once per mesh, before the first overlapped forward, and pass it as forwarding_buffer
        on every call. The arguments are the ones the overlapped call passes; emb_dim is the routed
        token width.
        )doc",
        &hybrid_routed_expert_moe_forwarding_buffer,
        nb::arg("mesh_device"),
        nb::kw_only(),
        nb::arg("emb_dim"),
        nb::arg("experts_per_chip"),
        nb::arg("num_experts_per_tok"),
        nb::arg("seq_len_per_chip"),
        nb::arg("combine_axis") = 0,
        nb::arg("combine_num_links") = 2);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_hybrid_routed_expert_ffn(::nanobind::module_& mod) {
    hybrid_routed_expert_ffn::detail::bind_hybrid_routed_expert_ffn(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
