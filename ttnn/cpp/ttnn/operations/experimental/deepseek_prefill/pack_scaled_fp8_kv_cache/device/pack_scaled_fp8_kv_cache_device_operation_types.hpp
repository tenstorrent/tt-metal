// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::pack_scaled_fp8_kv_cache {

struct PackScaledFp8KvCacheParams {
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct PackScaledFp8KvCacheInputs {
    const Tensor& latent;
    const Tensor& scales;
    const Tensor& rope;
};

}  // namespace ttnn::experimental::prim::pack_scaled_fp8_kv_cache
