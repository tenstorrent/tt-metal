// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::math::fft {

void py_module(nanobind::module_& mod);

}  // namespace ttnn::operations::math::fft
