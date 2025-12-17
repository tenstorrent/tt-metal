// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::transformer {

namespace nb = nanobind;
void bind_rotate_half(nb::module_& mod);

}  // namespace ttnn::operations::experimental::transformer
