// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::gather::detail {
namespace nb = nanobind;
void bind_gather_operation(nb::module_& mod);
}  // namespace ttnn::operations::experimental::gather::detail
