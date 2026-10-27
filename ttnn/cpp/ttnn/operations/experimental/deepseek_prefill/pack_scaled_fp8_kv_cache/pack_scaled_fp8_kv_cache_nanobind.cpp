// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pack_scaled_fp8_kv_cache_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "pack_scaled_fp8_kv_cache.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache::detail {

void bind_pack_scaled_fp8_kv_cache(nanobind::module_& mod) {
    namespace nb = nanobind;
    ttnn::bind_function<"pack_scaled_fp8_kv_cache", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(Pack one sparse-MLA KV row as [512 FP8 latent bytes | 4 FP32 scales | 64 BF16 RoPE values].)doc",
        &pack_scaled_fp8_kv_cache,
        nb::arg("latent"),
        nb::arg("scales"),
        nb::arg("rope"),
        nb::kw_only(),
        nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache::detail
