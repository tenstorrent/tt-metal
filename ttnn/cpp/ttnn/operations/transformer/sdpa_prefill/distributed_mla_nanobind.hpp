// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::transformer::sdpa_prefill {
namespace nb = nanobind;
void bind_distributed_mla(nb::module_& mod);

}  // namespace ttnn::operations::transformer::sdpa_prefill
