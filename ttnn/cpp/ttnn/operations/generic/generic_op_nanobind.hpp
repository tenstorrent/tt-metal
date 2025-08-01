// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::generic {
namespace nb = nanobind;
void bind_generic_operation(nb::module_& mod);
}  // namespace ttnn::operations::generic
