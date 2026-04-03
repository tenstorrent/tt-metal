// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::point_to_point {
namespace nb = nanobind;
void bind_point_to_point(nb::module_& mod);

}  // namespace ttnn::operations::point_to_point
