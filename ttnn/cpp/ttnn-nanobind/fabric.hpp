// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::fabric {

namespace nb = nanobind;
void bind_fabric_api(nb::module_& mod);

}  // namespace ttnn::fabric
