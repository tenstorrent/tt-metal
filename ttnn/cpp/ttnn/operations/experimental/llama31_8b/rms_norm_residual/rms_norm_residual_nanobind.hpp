// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::experimental::llama31_8b {

namespace nb = nanobind;
void bind_rms_norm_residual(nb::module_& llama31_8b);

}  // namespace ttnn::experimental::llama31_8b
