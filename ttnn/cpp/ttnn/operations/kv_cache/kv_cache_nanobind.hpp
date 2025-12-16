// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::kv_cache {

namespace nb = nanobind;
void bind_kv_cache(nb::module_& mod);

}  // namespace ttnn::operations::kv_cache
