// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::experimental::kda::sigmoid_gated_rms_norm::detail {

void bind_sigmoid_gated_rms_norm(nb::module_& mod);

}  // namespace ttnn::operations::experimental::kda::sigmoid_gated_rms_norm::detail
