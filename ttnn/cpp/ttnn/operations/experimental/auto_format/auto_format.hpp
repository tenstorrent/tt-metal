// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/constants.hpp>

#include <optional>

#include <tt-metalium/math.hpp>

namespace ttnn::operations::experimental::auto_format {
using PadValue = tt::tt_metal::PadValue;

class AutoFormat {
private:
    AutoFormat() = default;

public:
    /**
     * Formats an input tensor to meet device and layout requirements
     * @param input Input tensor
     * @param device Target device
     * @param padded_shape Required padded shape
     * @param pad_value Value to use for padding
     * @param target_layout Desired layout
     * @param target_mem_config Optional memory configuration
     * @return Formatted tensor
     */
    static Tensor format_tensor(
        const Tensor& input,
        tt::tt_metal::distributed::MeshDevice* device,
        const ttnn::Shape& padded_shape,
        PadValue pad_value,
        tt::tt_metal::Layout target_layout,
        std::optional<tt::tt_metal::MemoryConfig> target_mem_config = std::nullopt);

    /**
     * Formats an input tensor a given layout + memory_config. If padding is needed it uses pad_value (e.g. RM->TILE)
     * @param input Input tensor
     * @param pad_value Value to use for padding
     * @param target_layout Desired layout
     * @param target_mem_config Optional memory configuration
     * @return Formatted tensor
     */
    static Tensor format_tensor(
        const Tensor& input,
        PadValue pad_value,
        tt::tt_metal::Layout target_layout,
        std::optional<tt::tt_metal::MemoryConfig> target_mem_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::auto_format
