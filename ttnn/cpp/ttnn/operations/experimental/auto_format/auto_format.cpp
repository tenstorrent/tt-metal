// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/data_transfer/data_transfer.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"


namespace ttnn::operations::experimental::auto_format{

Tensor AutoFormat::move_tensor_to_device(const Tensor& input, Device* device, const MemoryConfig& mem_config) {
    if (input.storage_type() != StorageType::DEVICE) {
        return ttnn::data_transfer_to_device(input, device, mem_config);
    } else {
        return input;
    }
}

Tensor AutoFormat::move_tensor_to_mem_config(const Tensor& input, const MemoryConfig& mem_config) {
    if (input.storage_type() != StorageType::DEVICE) {
        return ttnn::data_transfer_to_device(input, AutoFormat::GetDefaultDevice(), mem_config);
    } else if (input.memory_config() != mem_config) {
        return ttnn::clone(input, mem_config);
    } else {
        return input;
    }
}

// This code is a workaround for cases where we need to remove autoformat but other dependent ops
// are not quite ready. So here we basically just put the tensor back on device.
// Used in backward_ops.cpp
// See: Remove auto format within permute_op.cpp #9404
Tensor AutoFormat::move_tensor_to_device_and_pad(const Tensor& input, Device *device, Layout target_layout, std::optional<MemoryConfig> target_mem_config){
    using namespace tt::constants;
    const auto intended_shape = input.get_shape();
    const auto device_shape = input.get_legacy_shape();
    const auto new_intended_shape = std::array<std::uint32_t, 4>{intended_shape[0], intended_shape[1], intended_shape[-2], intended_shape[-1]};
    const auto new_device_shape =  std::array<std::uint32_t, 4>{
        device_shape[0],
        device_shape[1],
        (device_shape[-2] % TILE_HEIGHT != 0 ? (device_shape[-2] / TILE_HEIGHT + 1) * TILE_HEIGHT : device_shape[-2]),
        (device_shape[-1] % TILE_WIDTH != 0 ? (device_shape[-1] / TILE_WIDTH + 1) * TILE_WIDTH : device_shape[-1])
        };
    const auto new_shape = tt::tt_metal::Shape(new_intended_shape, new_device_shape);
    return AutoFormat::format_input_tensor(input, device, new_shape, 0.0, target_layout, target_mem_config);
}

Tensor AutoFormat::format_input_tensor(
    const Tensor& input,
    Device* device,
    const tt::tt_metal::Shape& padded_shape,
    float pad_value,
    Layout target_layout,
    std::optional<MemoryConfig> target_mem_config) {
    bool pad_input = input.get_legacy_shape() != padded_shape;
    bool convert_layout = input.get_layout() != target_layout;

    if (!pad_input && !convert_layout) {
        return AutoFormat::move_tensor_to_device(input, device);
    }

    MemoryConfig mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    if (target_mem_config.has_value()) {
        mem_config = target_mem_config.value();
    } else if (input.storage_type() == StorageType::DEVICE) {
        mem_config = input.memory_config();
    }

    Tensor formatted_input = input;
    auto shape = formatted_input.get_legacy_shape();

    // TODO: Profile if it is faster to put host tensor to device and then pad/convert if possible
    // Device side conversions
    if (formatted_input.storage_type() == StorageType::DEVICE) {
        if (convert_layout && !pad_input) {
            if (target_layout == Layout::TILE && formatted_input.get_layout() == Layout::ROW_MAJOR) {
                return ttnn::tilize(formatted_input, mem_config);
            } else if (target_layout == Layout::ROW_MAJOR && formatted_input.get_layout() == Layout::TILE) {
                return ttnn::untilize(formatted_input, mem_config);
            }
        } else if (!convert_layout && pad_input) {
            if (formatted_input.get_layout() == Layout::ROW_MAJOR || formatted_input.get_layout() == Layout::TILE) {
                return ttnn::pad(0, (const ttnn::Tensor) formatted_input, padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), pad_value, false, mem_config);
            }
        } else if (convert_layout && pad_input) {
            if (formatted_input.get_layout() == Layout::ROW_MAJOR && target_layout == Layout::TILE) {
                return ttnn::tilize_with_val_padding(formatted_input, padded_shape, pad_value, mem_config);
            } else if (formatted_input.get_layout() == Layout::TILE && target_layout == Layout::ROW_MAJOR) {
                formatted_input = ttnn::untilize(formatted_input, mem_config);
                return ttnn::pad(0, (const ttnn::Tensor) formatted_input, padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), pad_value, false, mem_config);
            }
        }
        // Fall back to host conversions
        formatted_input = ttnn::data_transfer_to_host(formatted_input);
    }

    // Host side conversions
    if (pad_input) {
        if (formatted_input.get_layout() != Layout::ROW_MAJOR) {
            formatted_input = formatted_input.to(Layout::ROW_MAJOR);
            convert_layout = formatted_input.get_layout() != target_layout;
        }
        formatted_input = ttnn::pad((const ttnn::Tensor)formatted_input, padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), pad_value);
    }

    if (convert_layout) {
        formatted_input = formatted_input.to(target_layout);
    }

    return AutoFormat::move_tensor_to_device(formatted_input, device, mem_config);
}

Tensor AutoFormat::format_output_tensor(
    const Tensor& output,
    const tt::tt_metal::Shape& shape,
    Device* device,
    Layout target_layout,
    std::optional<MemoryConfig> target_mem_config) {
    bool unpad_output = output.get_legacy_shape() != shape;
    bool convert_layout = output.get_layout() != target_layout;

    if (!unpad_output && !convert_layout) {
        return output;
    }
    MemoryConfig mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    if (target_mem_config.has_value()) {
        mem_config = target_mem_config.value();
    } else if (output.storage_type() == StorageType::DEVICE) {
        mem_config = output.memory_config();
    }

    Tensor formatted_output = output;
    // Device side conversions
    if (formatted_output.storage_type() == StorageType::DEVICE) {
        if (!unpad_output && convert_layout) {
            // If target layout is tile but shape does not support tile, we don't do any conversions
            if (target_layout == Layout::TILE && formatted_output.get_layout() == Layout::ROW_MAJOR) {
                if (AutoFormat::legal_tile_shape(formatted_output.get_legacy_shape())) {
                    formatted_output = ttnn::tilize(formatted_output, mem_config);
                }
                return formatted_output;
            } else if (target_layout == Layout::ROW_MAJOR && formatted_output.get_layout() == Layout::TILE) {
                formatted_output = ttnn::untilize(formatted_output, mem_config);
                return formatted_output;
            }

        } else if (unpad_output && !convert_layout) {
            // Output can be unpadded and layout supports the shape
            if ((formatted_output.get_layout() == Layout::TILE && AutoFormat::legal_tile_shape(shape)) ||
                (formatted_output.get_layout() == Layout::ROW_MAJOR && AutoFormat::legal_rm_shape(shape))) {
                formatted_output = ttnn::slice(
                    0,
                    formatted_output,
                    std::vector<uint32_t>({0, 0, 0, 0}),
                    std::vector<uint32_t>({shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}),
                    mem_config);
                return formatted_output;
                // Output is tile but shape cannot be tile. We leave in RM
            } else if (formatted_output.get_layout() == Layout::TILE && AutoFormat::legal_rm_shape(shape)) {
                formatted_output = ttnn::untilize_with_unpadding(
                    formatted_output,
                    std::vector<uint32_t>({shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}),
                    mem_config);
                return formatted_output;
            }
        } else if (unpad_output && convert_layout) {
            if (formatted_output.get_layout() == Layout::TILE && target_layout == Layout::ROW_MAJOR &&
                AutoFormat::legal_rm_shape(shape)) {
                formatted_output = ttnn::untilize_with_unpadding(
                    formatted_output,
                    std::vector<uint32_t>({shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}),
                    mem_config);
                return formatted_output;
            } else if (
                formatted_output.get_layout() == Layout::ROW_MAJOR && target_layout == Layout::TILE &&
                AutoFormat::legal_tile_shape(shape)) {
                formatted_output = ttnn::slice(
                    0,
                    formatted_output,
                    std::vector<uint32_t>({0, 0, 0, 0}),
                    std::vector<uint32_t>({shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}),
                    mem_config);
                formatted_output = ttnn::tilize(formatted_output, mem_config);
                return formatted_output;
            }
        }
        // Fall back to host conversions
        formatted_output = ttnn::data_transfer_to_host(formatted_output);
    }

    // Host side conversions
    if (unpad_output) {
        // Requires RM for unpad
        if (formatted_output.get_layout() != Layout::ROW_MAJOR) {
            formatted_output = formatted_output.to(Layout::ROW_MAJOR);
            convert_layout = formatted_output.get_layout() != target_layout;
        }
        formatted_output =
            ttnn::slice(formatted_output, tt::tt_metal::Array4D({0, 0, 0, 0}), tt::tt_metal::Array4D({shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}));
    }

    if (convert_layout) {
        // Default to RM layout if we can't match the formatted_input layout
        if (target_layout == Layout::TILE && !AutoFormat::legal_tile_shape(formatted_output.get_legacy_shape())) {
            if (formatted_output.get_layout() != Layout::ROW_MAJOR) {
                formatted_output = formatted_output.to(Layout::ROW_MAJOR);
            }
        } else {
            formatted_output = formatted_output.to(target_layout);
        }
    }

    // Send formatted_output to device if possible
    // Check that shape is supported on device
    if (formatted_output.storage_type() != StorageType::DEVICE) {
        if (AutoFormat::legal_device_shape(formatted_output.get_legacy_shape(), formatted_output.get_layout())) {
            formatted_output = AutoFormat::move_tensor_to_device(formatted_output, device, mem_config);
        }
    }

    return formatted_output;
}

}  //namespace ttnn::operations::auto_format
