// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn-nanobind/nanobind_fwd.hpp"
namespace ttnn::operations::experimental {
void bind_dequantize_nvfp4(nanobind::module_& mod);
}
