// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::indexed_page_cache::detail {

namespace nb = nanobind;
void bind_experimental_indexed_page_cache_operations(nb::module_& mod);

}  // namespace ttnn::operations::experimental::indexed_page_cache::detail
