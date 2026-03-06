// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::topk_router_gpt::detail {
namespace nb = nanobind;
void bind_topk_router_gpt(nb::module_& mod);
}  // namespace ttnn::operations::experimental::topk_router_gpt::detail
