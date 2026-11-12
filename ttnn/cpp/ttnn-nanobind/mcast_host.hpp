// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::mcast_host {
namespace nb = nanobind;
void py_module_types(nb::module_& mod);
void py_module(nb::module_& mod);

}  // namespace ttnn::mcast_host
