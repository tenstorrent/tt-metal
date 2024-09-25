// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/operation.hpp"

#include <optional>

#include "tt_metal/common/math.hpp"

namespace ttnn::operations::experimental::auto_format{

struct FormatParams {
    ttnn::Shape pad_shape;
    float pad_value;
    Layout target_layout;
};

class AutoFormat {
    private:
        inline static Device* device = nullptr;

        AutoFormat() {}
    public:
        static void SetDefaultDevice(Device * dev) { device = dev; }
        static Device * GetDefaultDevice() { return device; }


        static ttnn::Shape pad_to_tile_shape(const ttnn::Shape& unpadded_shape, bool pad_c=false, bool pad_n=false, bool pad_h=true, bool pad_w=true) {
            using namespace tt::constants;
            auto n = pad_n ? tt::round_up(unpadded_shape.rank() >= 4 ? unpadded_shape[-4] : 1, TILE_HEIGHT) : unpadded_shape.rank() >= 4 ? unpadded_shape[-4] : 1;
            auto c = pad_c ? tt::round_up(unpadded_shape.rank() >= 3 ? unpadded_shape[-3] : 1, TILE_WIDTH) : unpadded_shape.rank() >= 3 ? unpadded_shape[-3] : 1;
            auto h = pad_h ? tt::round_up(unpadded_shape[-2], TILE_HEIGHT) : unpadded_shape[-2];
            auto w = pad_w ? tt::round_up(unpadded_shape[-1], TILE_WIDTH) : unpadded_shape[-1];
            ttnn::Shape padded_shape = ttnn::Shape(std::array<uint32_t, 4>{n, c, h, w});
            return padded_shape;
        }

        static ttnn::Shape pad_to_rm_shape(const ttnn::Shape& unpadded_shape) {
            ttnn::Shape padded_shape = unpadded_shape;
            padded_shape[3] = tt::round_up(unpadded_shape[3], 2);
            return padded_shape;
        }

        static ttnn::Shape pad_to_legal_shape(const ttnn::Shape& unpadded_shape, Layout layout) {
            ttnn::Shape padded_shape = unpadded_shape;
            switch (layout) {
                case Layout::ROW_MAJOR: padded_shape = pad_to_rm_shape(unpadded_shape); break;
                case Layout::TILE: padded_shape = pad_to_tile_shape(unpadded_shape);
                default: break;
            }
            return padded_shape;
        }

        // TODO: These legal checks should probably be somewhere else like tensor class, since it is common logic not just for autoformat
        static bool legal_tile_shape(const ttnn::Shape& shape) {
            return (shape[2] % tt::constants::TILE_HEIGHT == 0 && shape[3] % tt::constants::TILE_WIDTH == 0);
        }

        static bool legal_rm_shape(const ttnn::Shape& shape) {
            return (shape[3] % 2 == 0);
        }

        static bool legal_device_shape(const ttnn::Shape& shape, Layout layout) {
            switch (layout) {
                case Layout::ROW_MAJOR: return legal_rm_shape(shape);
                case Layout::TILE: return legal_tile_shape(shape);
                default: return true;
            }
        }


        static bool check_input_tensor_format(const Tensor &a, const ttnn::Shape& shape, Layout target_layout = Layout::TILE) {
            if (a.get_layout() == target_layout && a.get_shape() == shape && a.storage_type() == StorageType::DEVICE) {
                return true;
            }
            return false;
        }

        // This code is a workaround for cases where we need to remove autoformat but other dependent ops
        // are not quite ready. So here we basically just put the tensor back on device.
        // Used in backward_ops.cpp
        // See: Remove auto format within permute_op.cpp #9404
        static Tensor move_tensor_to_device_and_pad(const Tensor& input, Device *device, Layout target_layout, std::optional<MemoryConfig> target_mem_config);

        static Tensor move_tensor_to_device(const Tensor &input, Device * device, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

        static Tensor move_tensor_to_mem_config(const Tensor &input, const MemoryConfig& mem_config);

        static Tensor format_input_tensor(const Tensor &input, Device * device, const ttnn::Shape& padded_shape, float pad_value, Layout target_layout, std::optional<MemoryConfig> target_mem_config = std::nullopt);

        static Tensor format_output_tensor(const Tensor &output, const ttnn::Shape& shape, Device* device, Layout target_layout, std::optional<MemoryConfig> target_mem_config = std::nullopt);
};


} //ttnn::operations::experimental::auto_format
