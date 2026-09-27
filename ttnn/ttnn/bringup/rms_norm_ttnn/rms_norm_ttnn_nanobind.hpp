// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn::detail {
namespace nb = nanobind;
void bind_rms_norm_ttnn(nb::module_& mod);
}  // namespace ttnn::operations::bringup::rms_norm_ttnn::detail
