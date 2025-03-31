// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operation.hpp"

#include <optional>

#include <tt-metalium/math.hpp>

namespace ttnn::operations::experimental::auto_format {

struct FormatParams {
    ttnn::Shape pad_shape;
    float pad_value;
    tt::tt_metal::Layout target_layout;
};

class AutoFormat {
private:
    inline static tt::tt_metal::IDevice* device = nullptr;

    AutoFormat() {}

public:
    /**
     * Sets the default device to be used for auto-formatting operations
     * @param dev Pointer to the device to be used
     */
    static void SetDefaultDevice(tt::tt_metal::IDevice* dev);

    /**
     * Gets the default device used for auto-formatting operations
     * @return Pointer to the default device
     */
    static tt::tt_metal::IDevice* GetDefaultDevice();

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
     * Checks if a tensor matches the required format specifications
     * @param a Input tensor to check
     * @param shape Required shape
     * @param target_layout Required layout
     * @return True if tensor matches all format requirements
     */
    static bool check_input_tensor_format(
        const Tensor& a, const ttnn::Shape& shape, tt::tt_metal::Layout target_layout = tt::tt_metal::Layout::TILE);

    // This code is a workaround for cases where we need to remove autoformat but other dependent ops
    // are not quite ready. So here we basically just put the tensor back on device.
    // Used in backward_ops.cpp
    // See: Remove auto format within permute_op.cpp #9404
    /**
     * Moves a tensor to device memory and pads if necessary
     * @param input Input tensor
     * @param device Target device
     * @param target_layout Desired layout
     * @param target_mem_config Optional memory configuration
     * @return Formatted tensor on device
     */
    static Tensor move_tensor_to_device_and_pad(
        const Tensor& input,
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Layout target_layout,
        std::optional<tt::tt_metal::MemoryConfig> target_mem_config);

    /**
     * Moves a tensor to device memory
     * @param input Input tensor
     * @param device Target device
     * @param mem_config Memory configuration
     * @return Tensor on device
     */
    static Tensor move_tensor_to_device(
        const Tensor& input,
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::MemoryConfig& mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

    /**
     * Updates tensor memory configuration
     * @param input Input tensor
     * @param mem_config Target memory configuration
     * @return Tensor with updated memory configuration
     */
    static Tensor move_tensor_to_mem_config(const Tensor& input, const tt::tt_metal::MemoryConfig& mem_config);

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
    static Tensor format_input_tensor(
        const Tensor& input,
        tt::tt_metal::IDevice* device,
        const ttnn::Shape& padded_shape,
        float pad_value,
        tt::tt_metal::Layout target_layout,
        std::optional<tt::tt_metal::MemoryConfig> target_mem_config = std::nullopt);

    /**
     * Formats an output tensor to meet shape and layout requirements
     * @param output Output tensor
     * @param shape Target shape
     * @param device Target device
     * @param target_layout Desired layout
     * @param target_mem_config Optional memory configuration
     * @return Formatted output tensor
     */
    static Tensor format_output_tensor(
        const Tensor& output,
        const ttnn::Shape& shape,
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Layout target_layout,
        std::optional<tt::tt_metal::MemoryConfig> target_mem_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::auto_format
