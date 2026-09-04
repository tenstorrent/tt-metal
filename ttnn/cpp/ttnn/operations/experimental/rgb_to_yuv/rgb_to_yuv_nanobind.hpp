// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace nb = nanobind;

namespace ttnn::experimental::detail {
void bind_rgb_to_yuv(nb::module_& mod);
}
