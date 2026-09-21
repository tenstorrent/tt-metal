// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::compressor_state_exchange::detail {
namespace nb = nanobind;
void bind_compressor_state_exchange(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::compressor_state_exchange::detail

namespace ttnn::operations::experimental::deepseek_prefill::detail {
void bind_compressor_state_exchange(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
