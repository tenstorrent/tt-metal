// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_post_mix::detail {
void bind_experimental_mhc_post_mix_operation(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_post_mix::detail
