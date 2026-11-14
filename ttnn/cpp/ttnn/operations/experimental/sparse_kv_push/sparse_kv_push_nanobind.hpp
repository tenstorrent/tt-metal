// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::sparse_kv_push::detail {

namespace nb = nanobind;
void bind_experimental_sparse_kv_push_operations(nb::module_& mod);

}  // namespace ttnn::operations::experimental::sparse_kv_push::detail
