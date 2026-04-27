// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn-nanobind/nanobind_fwd.hpp>

namespace ttnn::operations::rotate {
namespace nb = nanobind;
void py_module(nb::module_& mod);

}  // namespace ttnn::operations::rotate
