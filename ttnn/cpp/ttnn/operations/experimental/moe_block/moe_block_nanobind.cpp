// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "common/common_nanobind.hpp"
#include "common/moe_block_config.hpp"
#include "common/moe_block_resources.hpp"
#include "moe_head/moe_head_nanobind.hpp"
#include "sparse_dispatch_2d/sparse_dispatch_2d_nanobind.hpp"

namespace ttnn::operations::experimental::moe_block {

void bind_moe_block(nb::module_& experimental) {
    auto mod = experimental.def_submodule(
        "moe_block", "The MoE block's ops (prefill): head, dispatch, experts, combine, tail");
    bind_common(mod);
    moe_head::bind_moe_head(mod);
    sparse_dispatch_2d::bind_sparse_dispatch_2d(mod);
}

}  // namespace ttnn::operations::experimental::moe_block
