// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::reduce_planner {
namespace nb = nanobind;

void py_module(nb::module_& planner);

}  // namespace ttnn::reduce_planner
