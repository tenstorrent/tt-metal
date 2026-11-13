// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::wavelet {

namespace nb = nanobind;

void bind_wavelet_operations(nb::module_& mod);

}  // namespace ttnn::operations::wavelet
