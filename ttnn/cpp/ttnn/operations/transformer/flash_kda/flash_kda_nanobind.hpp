// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::transformer {
namespace nb = nanobind;
void bind_flash_kda(nb::module_& mod);
}  // namespace ttnn::operations::transformer
