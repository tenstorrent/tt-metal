// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_combine_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "unified_routed_expert_combine.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine::detail {

void bind_unified_routed_expert_combine(nb::module_& mod) {
    // RoutedExpertActivation is already registered by unified_routed_expert_ffn; binding it again would make a
    // distinct Python type no caster accepts.
    ttnn::bind_function<"unified_routed_expert_moe_combine", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
        The routed expert on unified_routed_expert_moe's wave-planned program, alone or overlapped with
        combine_fabric2d in ONE program.

        Small experts run at once on slices of the grid, each with its own weight stream;
        big_first_threshold (0, or at least 96 tokens) runs the experts above that many tokens first.
        dispatched_buffer must be bfloat16 ROW_MAJOR, and alone the op returns the routed expert's
        bfloat8_b TILE output.

        Passing dispatched_metadata overlaps combine_fabric2d: the routed expert runs on rows 2-9,
        combine on rows 0-1 ships each expert's tokens as soon as the routed expert has written them, in
        the order the experts actually finish, and the op returns combine's output. expert_offsets,
        replicated_global_expert_idx_table (the full table on every device), combine_axis,
        combine_num_links, num_experts_per_tok and seq_len_per_chip are then combine_fabric2d's, and
        fwd_arrived_semaphore, final_arrived_semaphore, expert_go_semaphore and channel_semaphores
        (hybrid_routed_expert_moe_channel_semaphore_count of them) are global semaphores created once,
        initialised to zero, and passed on every call, since neighbouring chips bump them across
        launches. forwarding_buffer is combine's DRAM staging buffer from
        hybrid_routed_expert_moe_forwarding_buffer; omitted, combine allocates its own at the first call
        and keeps it for the program's lifetime.
        )doc",
        &unified_routed_expert_moe_combine,
        nb::arg("dispatched_buffer").noconvert(),
        nb::arg("expert_region_offsets").noconvert(),
        nb::arg("expert_token_counts").noconvert(),
        nb::arg("global_expert_idx_table").noconvert(),
        nb::arg("gate_projs").noconvert(),
        nb::arg("up_projs").noconvert(),
        nb::arg("down_projs").noconvert(),
        nb::arg("max_dispatched_tokens_per_expert"),
        nb::kw_only(),
        nb::arg("big_first_threshold") = 0,
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
        nb::arg("channel_semaphores") = std::vector<GlobalSemaphore>{},
        nb::arg("forwarding_buffer") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_unified_routed_expert_combine(::nanobind::module_& mod) {
    unified_routed_expert_combine::detail::bind_unified_routed_expert_combine(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
