// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::quasar::conv_transpose2d_impl {
namespace nb = nanobind;
void bind_conv_transpose2d(nb::module_& mod);
}  // namespace ttnn::operations::experimental::quasar::conv_transpose2d_impl
