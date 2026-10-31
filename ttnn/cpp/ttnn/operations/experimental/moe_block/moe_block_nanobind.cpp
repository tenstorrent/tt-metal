// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "block/moe_block_arena.hpp"
#include "common/common_nanobind.hpp"
#include "common/moe_block_config.hpp"
#include "common/moe_block_resources.hpp"
#include "flat_routed_expert/flat_routed_expert_nanobind.hpp"
#include "fused_head/fused_head_nanobind.hpp"
#include "fused_tail/fused_tail_nanobind.hpp"
#include "routed_experts_combine/routed_experts_combine_nanobind.hpp"
#include "sparse_dispatch_2d/sparse_dispatch_2d_nanobind.hpp"

namespace ttnn::operations::experimental::moe_block {

void bind_moe_block(nb::module_& experimental) {
    auto mod = experimental.def_submodule(
        "moe_block", "The MoE block's ops (prefill): head, dispatch, experts, combine, tail");
    bind_common(mod);
    fused_head::bind_fused_head(mod);
    sparse_dispatch_2d::bind_sparse_dispatch_2d(mod);
    flat_routed_expert::bind_flat_routed_expert(mod);
    routed_experts_combine::bind_routed_experts_combine(mod);
    fused_tail::bind_fused_tail(mod);

    mod.def(
        "moe_block_arena_spec",
        &moe_block_arena_spec,
        nb::arg("resources"),
        nb::kw_only(),
        nb::arg("dtypes"),
        nb::arg("accumulators"),
        nb::arg("max_tokens_per_expert"),
        nb::arg("activation"),
        nb::arg("layout") = flat_routed_expert::FlatLayoutOptions{},
        nb::arg("combine") = routed_experts_combine::CombineConfig{},
        nb::arg("l1_free") = nb::none(),
        R"doc(
        The TensorSpec of the block's one L1 arena a call (ttnn.allocate_tensor_on_device): passed to
        fused_head(..., arena=) and routed_experts_combine(..., arena=), freed after the combine. The combine's arena
        (every worker core, height-sharded L1) at max(the combine's need, the fused head's arena bytes) a core.

        Args:
            resources: the block's MoEBlockResources.
            dtypes, accumulators, l1_free: as fused_head's (plans the head on first use).
            max_tokens_per_expert, activation, layout, combine: as routed_experts_combine's (plans it on first use;
                combine.n_acc, when set, must equal accumulators).
        )doc");
}

}  // namespace ttnn::operations::experimental::moe_block
