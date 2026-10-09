// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The Quasar layer_norm clone shares the program-config / enum types with the mainline op so the
// Python objects (ttnn.LayerNormDefaultProgramConfig, ttnn.LayerNormShardedMultiCoreProgramConfig,
// ...) are the same on both; they are re-exported into ttnn::prim::qsr for the cloned sources.
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

namespace ttnn::prim::qsr {

using ttnn::prim::DistributedLayerNormStage;
using ttnn::prim::LayerNormDefaultProgramConfig;
using ttnn::prim::LayerNormProgramConfig;
using ttnn::prim::LayerNormShardedMultiCoreProgramConfig;
using ttnn::prim::LayerNormType;

}  // namespace ttnn::prim::qsr
