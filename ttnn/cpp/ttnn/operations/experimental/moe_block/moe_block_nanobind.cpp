// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "common/common_nanobind.hpp"
#include "flat_routed_expert/flat_routed_expert_nanobind.hpp"
#include "fused_head/fused_head_nanobind.hpp"
#include "fused_tail/fused_tail_nanobind.hpp"
#include "routed_experts_combine/routed_experts_combine_nanobind.hpp"
#include "sparse_dispatch_2d/sparse_dispatch_2d_nanobind.hpp"

namespace ttnn::operations::experimental::moe_block {

void bind_moe_block(nb::module_& experimental) {
    auto mod = experimental.def_submodule("moe_block", "The MoE block's ops (prefill): head, dispatch, experts, combine, tail");
    bind_common(mod);
    fused_head::bind_fused_head(mod);
    sparse_dispatch_2d::bind_sparse_dispatch_2d(mod);
    flat_routed_expert::bind_flat_routed_expert(mod);
    routed_experts_combine::bind_routed_experts_combine(mod);
    fused_tail::bind_fused_tail(mod);
}

}  // namespace ttnn::operations::experimental::moe_block
