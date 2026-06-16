// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::h2d_socket_sync::detail {

void bind_h2d_socket_sync(nb::module_& mod);

}  // namespace ttnn::operations::experimental::h2d_socket_sync::detail
