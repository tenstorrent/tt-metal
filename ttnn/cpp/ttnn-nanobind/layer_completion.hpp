// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::disaggregation {

namespace nb = nanobind;
void bind_layer_completion(nb::module_& mod);

}  // namespace ttnn::disaggregation
