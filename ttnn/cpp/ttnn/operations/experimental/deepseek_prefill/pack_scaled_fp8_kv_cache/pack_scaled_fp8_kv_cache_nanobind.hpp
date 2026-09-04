// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache::detail {
void bind_pack_scaled_fp8_kv_cache(nanobind::module_& mod);
}
