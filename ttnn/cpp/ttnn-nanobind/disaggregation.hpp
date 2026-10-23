// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::disaggregation {

namespace nb = nanobind;
void bind_disaggregation_api(nb::module_& mod);

}  // namespace ttnn::disaggregation
