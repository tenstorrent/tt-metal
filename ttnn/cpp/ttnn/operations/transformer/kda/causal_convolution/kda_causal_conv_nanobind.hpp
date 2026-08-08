// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <nanobind/nanobind.h>
namespace nb = nanobind;
namespace ttnn::operations::transformer {
void bind_kda_causal_conv(nb::module_&);
}
