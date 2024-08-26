// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "tt_metal/common/core_coord.h"

namespace ttnn::operations::normalization {

enum class LayerNormDistributedType {
    LAYERNORM, RMSNORM
};

}  // namespace ttnn::operations::normalization
