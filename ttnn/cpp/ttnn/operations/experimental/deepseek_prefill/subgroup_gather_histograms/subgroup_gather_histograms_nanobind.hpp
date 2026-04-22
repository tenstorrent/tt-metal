// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms::detail {

namespace nb = nanobind;

void bind_experimental_subgroup_gather_histograms_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms::detail
