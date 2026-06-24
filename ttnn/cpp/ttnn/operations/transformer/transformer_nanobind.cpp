// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "transformer_nanobind.hpp"

#include <cstddef>
#include <optional>
#include <tt_stl/reflection.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "concatenate_heads/concatenate_heads_nanobind.hpp"
#include "gated_delta_attn/gated_delta_attn_nanobind.hpp"

namespace ttnn::operations::transformer {

void py_module(nb::module_& mod) {
    // NOTE: SDPAProgramConfig registration removed — the sdpa op was nuked for
    // the agent-regen baseline.
    bind_concatenate_heads(mod);

    bind_gated_delta_attn_seq(mod);
}

}  // namespace ttnn::operations::transformer
