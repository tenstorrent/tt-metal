// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::bringup::post_combine_reduce::detail {
namespace nb = nanobind;
void bind_post_combine_reduce(nb::module_& mod);
}  // namespace ttnn::operations::bringup::post_combine_reduce::detail

namespace ttnn::operations::bringup::detail {
void bind_post_combine_reduce(::nanobind::module_& mod);
}  // namespace ttnn::operations::bringup::detail
