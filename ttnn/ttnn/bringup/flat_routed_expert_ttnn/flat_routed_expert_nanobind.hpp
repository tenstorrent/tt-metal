// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::bringup::detail {

void bind_flat_routed_expert(::nanobind::module_& mod);

}  // namespace ttnn::operations::bringup::detail
