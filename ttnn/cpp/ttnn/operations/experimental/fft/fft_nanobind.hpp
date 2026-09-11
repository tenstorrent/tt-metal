// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::fft_binding::detail {

namespace nb = nanobind;

void bind_experimental_fft_operation(nb::module_& mod);

}  // namespace ttnn::operations::experimental::fft_binding::detail
