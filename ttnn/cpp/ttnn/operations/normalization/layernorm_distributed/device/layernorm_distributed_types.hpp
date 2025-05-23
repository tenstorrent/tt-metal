// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::normalization {

enum class LayerNormDistributedType { LAYERNORM, RMSNORM };

}  // namespace ttnn::operations::normalization
