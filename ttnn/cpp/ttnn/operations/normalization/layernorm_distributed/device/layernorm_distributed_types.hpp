// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::normalization {

struct LayerNormDistributedDefaultProgramConfig {
    bool legacy_reduction = true;
    bool legacy_rsqrt = true;
};

enum class LayerNormDistributedType { LAYERNORM, RMSNORM };

}  // namespace ttnn::operations::normalization
