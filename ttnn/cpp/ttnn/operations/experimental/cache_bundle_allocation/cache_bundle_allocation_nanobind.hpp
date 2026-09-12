// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::cache_bundle_allocation::detail {

void bind_cache_bundle_allocation_operation(nanobind::module_& mod);

}  // namespace ttnn::operations::experimental::cache_bundle_allocation::detail
