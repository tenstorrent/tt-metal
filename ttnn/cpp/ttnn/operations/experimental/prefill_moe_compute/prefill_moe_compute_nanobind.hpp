// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::prefill_moe_compute::detail {
namespace nb = nanobind;
void bind_prefill_moe_compute(nb::module_& mod);
}  // namespace ttnn::operations::experimental::prefill_moe_compute::detail
