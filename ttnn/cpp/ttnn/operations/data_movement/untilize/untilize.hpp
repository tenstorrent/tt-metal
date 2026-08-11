// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/data_movement/common/codegen_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// `implementation` selects the host dispatch. See codegen/untilize_codegen_supported.hpp.
ttnn::Tensor untilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool use_multicore = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    operations::data_movement::ImplementationSelector implementation =
        operations::data_movement::ImplementationSelector::Auto);

}  // namespace ttnn
