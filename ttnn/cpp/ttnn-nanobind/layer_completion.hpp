// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::layer_completion {
namespace nb = nanobind;

// Register the pipelined-prefill layer-completion ring/router/consumer bindings onto `mod` (a submodule
// of _ttnn). Folded into the main ttnn module — was a standalone `_layer_completion` extension. The
// tt_metal types are consumed via the sanctioned `tt_metal/api/internal/disaggregation/` surface.
void bind_layer_completion_api(nb::module_& mod);

}  // namespace ttnn::layer_completion
