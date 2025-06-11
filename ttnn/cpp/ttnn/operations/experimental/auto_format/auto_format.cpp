// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/auto_format/auto_format.hpp"

#include <utility>

#include <tt-metalium/constants.hpp>
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::auto_format {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

/**
 * Checks if a shape is legal for tile layout
 * @param shape Shape to check
 * @return True if shape dimensions are properly aligned for tile layout
 */
bool legal_tile_shape(const ttnn::Shape& shape) {
    return (shape[2] % tt::constants::TILE_HEIGHT == 0 && shape[3] % tt::constants::TILE_WIDTH == 0);
}

/**
 * Checks if a shape is legal for a specific device layout
 * @param shape Shape to check
 * @param layout Target layout
 * @return True if shape is legal for the specified layout
 */
bool legal_device_shape(const ttnn::Shape& shape, tt::tt_metal::Layout layout) {
    switch (layout) {
        case tt::tt_metal::Layout::ROW_MAJOR: return true;
        case tt::tt_metal::Layout::TILE: return legal_tile_shape(shape);
        default: return true;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // anonymous namespace

Tensor AutoFormat::move_tensor_to_device(const Tensor& input, IDevice* device, const MemoryConfig& mem_config) {
    if (input.storage_type() != StorageType::DEVICE) {
        return input.to_device(device, mem_config);
    } else {
        return input;
    }
}

Tensor AutoFormat::move_tensor_to_mem_config(const Tensor& input, const MemoryConfig& mem_config) {
    if (input.storage_type() != StorageType::DEVICE) {
        return input.to_device(AutoFormat::GetDefaultDevice(), mem_config);
    } else if (input.memory_config() != mem_config) {
        return ttnn::clone(input, std::nullopt, mem_config, std::nullopt);
    } else {
        return input;
    }
}

// This code is a workaround for cases where we need to remove autoformat but other dependent ops
// are not quite ready. So here we basically just put the tensor back on device.
// Used in backward_ops.cpp
// See: Remove auto format within permute_op.cpp #9404
Tensor AutoFormat::move_tensor_to_device_and_pad(
    const Tensor& input, IDevice* device, Layout target_layout, std::optional<MemoryConfig> target_mem_config) {
    using namespace tt::constants;
    const auto device_shape = input.padded_shape();
    const Shape new_device_shape(
        {device_shape[0],
         device_shape[1],
         (device_shape[-2] % TILE_HEIGHT != 0 ? (device_shape[-2] / TILE_HEIGHT + 1) * TILE_HEIGHT : device_shape[-2]),
         (device_shape[-1] % TILE_WIDTH != 0 ? (device_shape[-1] / TILE_WIDTH + 1) * TILE_WIDTH : device_shape[-1])});
    return AutoFormat::format_input_tensor(
        input, device, new_device_shape, 0.0, target_layout, std::move(target_mem_config));
}

Tensor AutoFormat::format_input_tensor(
    const Tensor& input,
    IDevice* device,
    const ttnn::Shape& padded_shape,
    float pad_value,
    Layout target_layout,
    std::optional<MemoryConfig> target_mem_config) {
    bool pad_input = input.padded_shape() != padded_shape;
    bool convert_layout = input.layout() != target_layout;

    if (!pad_input && !convert_layout) {
        return AutoFormat::move_tensor_to_device(input, device);
    }

    MemoryConfig mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    if (target_mem_config.has_value()) {
        mem_config = target_mem_config.value();
    } else if (input.storage_type() == StorageType::DEVICE) {
        mem_config = input.memory_config();
    }

    Tensor formatted_input = input;
    auto shape = formatted_input.padded_shape();

    // TODO: Profile if it is faster to put host tensor to device and then pad/convert if possible
    // Device side conversions
    if (formatted_input.storage_type() == StorageType::DEVICE) {
        if (convert_layout && !pad_input) {
            if (target_layout == Layout::TILE && formatted_input.layout() == Layout::ROW_MAJOR) {
                return ttnn::tilize(formatted_input, mem_config);
            } else if (target_layout == Layout::ROW_MAJOR && formatted_input.layout() == Layout::TILE) {
                return ttnn::untilize(formatted_input, mem_config);
            }
        } else if (!convert_layout && pad_input) {
            if (formatted_input.layout() == Layout::ROW_MAJOR || formatted_input.layout() == Layout::TILE) {
                return ttnn::pad(
                    DefaultQueueId,
                    (const ttnn::Tensor)formatted_input,
                    padded_shape.to_array_4D(),
                    tt::tt_metal::Array4D({0, 0, 0, 0}),
                    pad_value,
                    /* multicore */ false,
                    mem_config);
            }
        } else if (convert_layout && pad_input) {
            if (formatted_input.layout() == Layout::ROW_MAJOR && target_layout == Layout::TILE) {
                PadValue pad_value_variant;
                if (formatted_input.dtype() == ttnn::DataType::BFLOAT16 or
                    formatted_input.dtype() == ttnn::DataType::FLOAT32) {
                    pad_value_variant = (float)pad_value;
                } else {
                    pad_value_variant = (uint32_t)pad_value;
                }
                return ttnn::tilize_with_val_padding(formatted_input, padded_shape, pad_value_variant, mem_config);
            } else if (formatted_input.layout() == Layout::TILE && target_layout == Layout::ROW_MAJOR) {
                formatted_input = ttnn::untilize(formatted_input, mem_config);
                return ttnn::pad(
                    DefaultQueueId,
                    (const ttnn::Tensor)formatted_input,
                    padded_shape.to_array_4D(),
                    tt::tt_metal::Array4D({0, 0, 0, 0}),
                    pad_value,
                    /* multicore */ false,
                    mem_config);
            }
        }
        // Fall back to host conversions
        formatted_input = formatted_input.cpu();
    }

    // Host side conversions
    if (pad_input) {
        if (formatted_input.layout() != Layout::ROW_MAJOR) {
            formatted_input = formatted_input.to_layout(Layout::ROW_MAJOR);
            convert_layout = formatted_input.layout() != target_layout;
        }
        formatted_input = ttnn::pad(
            (const ttnn::Tensor)formatted_input,
            padded_shape.to_array_4D(),
            tt::tt_metal::Array4D({0, 0, 0, 0}),
            pad_value);
    }

    if (convert_layout) {
        formatted_input = formatted_input.to_layout(target_layout);
    }

    return AutoFormat::move_tensor_to_device(formatted_input, device, mem_config);
}

Tensor AutoFormat::format_output_tensor(
    const Tensor& output,
    const ttnn::Shape& shape,
    IDevice* device,
    Layout target_layout,
    std::optional<MemoryConfig> target_mem_config) {
    bool unpad_output = output.padded_shape() != shape;
    bool convert_layout = output.layout() != target_layout;

    if (!unpad_output && !convert_layout) {
        return output;
    }
    MemoryConfig mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
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
            if (target_layout == Layout::TILE && formatted_output.layout() == Layout::ROW_MAJOR) {
                if (CMAKE_UNIQUE_NAMESPACE::legal_tile_shape(formatted_output.padded_shape())) {
                    formatted_output = ttnn::tilize(formatted_output, mem_config);
                }
                return formatted_output;
            } else if (target_layout == Layout::ROW_MAJOR && formatted_output.layout() == Layout::TILE) {
                formatted_output = ttnn::untilize(formatted_output, mem_config);
                return formatted_output;
            }

        } else if (unpad_output && !convert_layout) {
            // Output can be unpadded and layout supports the shape
            if ((formatted_output.layout() == Layout::TILE && CMAKE_UNIQUE_NAMESPACE::legal_tile_shape(shape)) ||
                (formatted_output.layout() == Layout::ROW_MAJOR)) {
                auto begins = std::array<uint32_t, 4>({0, 0, 0, 0});
                auto ends = std::array<uint32_t, 4>({shape[0], shape[1], shape[2], shape[3]});
                auto step = std::array<uint32_t, 4>({1, 1, 1, 1});

                formatted_output = ttnn::slice(formatted_output, begins, ends, step, mem_config);
                return formatted_output;
                // Output is tile but shape cannot be tile. We leave in RM
            } else if (formatted_output.layout() == Layout::TILE) {
                formatted_output = ttnn::untilize_with_unpadding(
                    formatted_output,
                    ttnn::Shape({shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}),
                    mem_config);
                return formatted_output;
            }
        } else if (unpad_output && convert_layout) {
            if (formatted_output.layout() == Layout::TILE && target_layout == Layout::ROW_MAJOR) {
                formatted_output = ttnn::untilize_with_unpadding(
                    formatted_output,
                    ttnn::Shape({shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1}),
                    mem_config);
                return formatted_output;
            } else if (
                formatted_output.layout() == Layout::ROW_MAJOR && target_layout == Layout::TILE &&
                CMAKE_UNIQUE_NAMESPACE::legal_tile_shape(shape)) {
                auto begins = std::array<uint32_t, 4>({0, 0, 0, 0});
                auto ends = std::array<uint32_t, 4>({shape[0], shape[1], shape[2], shape[3]});
                auto step = std::array<uint32_t, 4>({1, 1, 1, 1});
                formatted_output = ttnn::slice(formatted_output, begins, ends, step, mem_config);
                formatted_output = ttnn::tilize(formatted_output, mem_config);
                return formatted_output;
            }
        }
        // Fall back to host conversions
        formatted_output = formatted_output.cpu();
    }

    // Host side conversions
    if (unpad_output) {
        // Requires RM for unpad
        if (formatted_output.layout() != Layout::ROW_MAJOR) {
            formatted_output = formatted_output.to_layout(Layout::ROW_MAJOR);
            convert_layout = formatted_output.layout() != target_layout;
        }
        auto begins = std::array<uint32_t, 4>({0, 0, 0, 0});
        auto ends = std::array<uint32_t, 4>({shape[0], shape[1], shape[2], shape[3]});
        auto step = std::array<uint32_t, 4>({1, 1, 1, 1});
        formatted_output = ttnn::slice(formatted_output, begins, ends, step, std::nullopt);
    }

    if (convert_layout) {
        // Default to RM layout if we can't match the formatted_input layout
        if (target_layout == Layout::TILE &&
            !CMAKE_UNIQUE_NAMESPACE::legal_tile_shape(formatted_output.padded_shape())) {
            if (formatted_output.layout() != Layout::ROW_MAJOR) {
                formatted_output = formatted_output.to_layout(Layout::ROW_MAJOR);
            }
        } else {
            formatted_output = formatted_output.to_layout(target_layout);
        }
    }

    // Send formatted_output to device if possible
    // Check that shape is supported on device
    if (formatted_output.storage_type() != StorageType::DEVICE) {
        if (CMAKE_UNIQUE_NAMESPACE::legal_device_shape(formatted_output.padded_shape(), formatted_output.layout())) {
            formatted_output = AutoFormat::move_tensor_to_device(formatted_output, device, mem_config);
        }
    }

    return formatted_output;
}

void AutoFormat::SetDefaultDevice(tt::tt_metal::IDevice* dev) { device = dev; }

tt::tt_metal::IDevice* AutoFormat::GetDefaultDevice() { return device; }

ttnn::Shape AutoFormat::pad_to_tile_shape(const ttnn::Shape& unpadded_shape) {
    using namespace tt::constants;
    auto rank = unpadded_shape.rank();
    TT_ASSERT(rank >= 1, "rank of shape to pad to tile shape must be at least 1.");
    SmallVector<uint32_t> padded_shape_vec(rank);

    for (auto i = 0; i < rank; ++i) {
        padded_shape_vec[i] = unpadded_shape[i];
    }
    if (rank >= 1) {
        auto w = tt::round_up(unpadded_shape[rank - 1], TILE_WIDTH);
        padded_shape_vec[rank - 1] = w;
    }
    if (rank >= 2) {
        auto h = tt::round_up(unpadded_shape[rank - 2], TILE_HEIGHT);
        padded_shape_vec[rank - 2] = h;
    }
    return Shape(padded_shape_vec);
}

bool AutoFormat::check_input_tensor_format(
    const Tensor& a, const ttnn::Shape& shape, tt::tt_metal::Layout target_layout) {
    return a.layout() == target_layout && a.padded_shape() == shape &&
           a.storage_type() == tt::tt_metal::StorageType::DEVICE;
}

}  // namespace ttnn::operations::experimental::auto_format
