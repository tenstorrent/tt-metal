// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::bringup::detail {

void bind_rotary_embedding_indexed(::nanobind::module_& mod);

}  // namespace ttnn::operations::bringup::detail
