// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_block_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "common/common_nanobind.hpp"
#include "common/moe_block_config.hpp"
#include "common/moe_block_resources.hpp"

namespace ttnn::operations::experimental::moe_block {

void bind_moe_block(nb::module_& experimental) {
    auto mod = experimental.def_submodule(
        "moe_block", "The MoE block's ops (prefill): head, dispatch, experts, combine, tail");
    bind_common(mod);
}

}  // namespace ttnn::operations::experimental::moe_block
