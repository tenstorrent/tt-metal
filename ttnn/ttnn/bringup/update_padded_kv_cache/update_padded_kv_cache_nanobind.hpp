// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::bringup::update_padded_kv_cache::detail {
namespace nb = nanobind;
void bind_update_padded_kv_cache(nb::module_& mod);
}  // namespace ttnn::operations::bringup::update_padded_kv_cache::detail

namespace ttnn::operations::bringup::detail {
void bind_update_padded_kv_cache(::nanobind::module_& mod);
}  // namespace ttnn::operations::bringup::detail
