// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::experimental::llama31_8b_spec {

namespace nb = nanobind;
void bind_rms_norm_residual_decode(nb::module_& llama31_8b_spec);

}  // namespace ttnn::experimental::llama31_8b_spec
