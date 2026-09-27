// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::bringup::dispatch::detail {
namespace nb = nanobind;
void bind_dispatch(nb::module_& mod);

}  // namespace ttnn::operations::bringup::dispatch::detail

namespace ttnn::operations::bringup::detail {
void bind_dispatch(::nanobind::module_& mod);
}  // namespace ttnn::operations::bringup::detail
