// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::deepseek::mla::mla_wqkv_ab::detail {
namespace nb = nanobind;
void bind_mla_wqkv_ab(nb::module_& mod);

}  // namespace ttnn::operations::experimental::deepseek::mla::mla_wqkv_ab::detail

namespace ttnn::operations::experimental::deepseek::mla::detail {
void bind_mla_wqkv_ab(::nanobind::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek::mla::detail
