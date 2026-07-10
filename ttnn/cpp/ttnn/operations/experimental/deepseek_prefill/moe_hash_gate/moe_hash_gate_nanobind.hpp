// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate::detail {
namespace nb = nanobind;
void bind_moe_hash_gate(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate::detail
