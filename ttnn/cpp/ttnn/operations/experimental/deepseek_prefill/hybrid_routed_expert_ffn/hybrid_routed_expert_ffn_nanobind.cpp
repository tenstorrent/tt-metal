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
        nb::arg("down_biases") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {

void bind_hybrid_routed_expert_ffn(::nanobind::module_& mod) {
    hybrid_routed_expert_ffn::detail::bind_hybrid_routed_expert_ffn(mod);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
