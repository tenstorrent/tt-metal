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

struct FormatParams {
    ttnn::Shape pad_shape;
    float pad_value{};
    tt::tt_metal::Layout target_layout{tt::tt_metal::Layout::INVALID};
};

class AutoFormat {
private:
    inline static tt::tt_metal::distributed::MeshDevice* device = nullptr;

    AutoFormat() = default;

public:
    /**
     * Sets the default device to be used for auto-formatting operations
     * @param dev Pointer to the device to be used
     */
    static void SetDefaultDevice(tt::tt_metal::distributed::MeshDevice* dev);

    /**
     * Gets the default device used for auto-formatting operations
     * @return Pointer to the default device
     */
    static tt::tt_metal::distributed::MeshDevice* GetDefaultDevice();

    /**
     * Pads a shape to align with tile dimensions
     * @param unpadded_shape Original shape to be padded
     * @param pad_c Whether to pad the channel dimension
     * @param pad_n Whether to pad the batch dimension
     * @param pad_h Whether to pad the height dimension
     * @param pad_w Whether to pad the width dimension
     * @return Padded shape aligned to tile dimensions
     */
    static ttnn::Shape pad_to_tile_shape(const ttnn::Shape& unpadded_shape);

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
