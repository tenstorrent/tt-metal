// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm.hpp"
#include "device/fill_rm_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"

namespace ttnn::operations::data_movement{

ttnn::Tensor FillRMOperation::invoke(uint8_t queue_id, uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, const ttnn::Tensor& any, float val_hi, float val_lo, const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(any.memory_config());
    return operation::run_without_autoformat(FillRM{N, C, H, W, hFill, wFill, val_hi, val_lo, output_memory_config}, {any}, {}, {}, queue_id).at(0);
}

ttnn::Tensor FillRMOperation::invoke(uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, const ttnn::Tensor& any, float val_hi, float val_lo, const std::optional<ttnn::MemoryConfig>& memory_config_arg) {
    return invoke(DefaultQueueId, N, C, H, W, hFill, wFill, any, val_hi, val_lo, memory_config_arg);
}

ttnn::Tensor FillOnesRMOperation::invoke(uint8_t queue_id, uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, const ttnn::Tensor& any, const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(any.memory_config());
    return operation::run_without_autoformat(FillRM{N, C, H, W, hFill, wFill, 1.0f, 0.0f, output_memory_config},  {any}, {}, {}, queue_id).at(0);
}

ttnn::Tensor FillOnesRMOperation::invoke(uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, const ttnn::Tensor& any, const std::optional<ttnn::MemoryConfig>& memory_config_arg) {
    return invoke(DefaultQueueId, N, C, H, W, hFill, wFill, any, memory_config_arg);
}

ttnn::Tensor FillImplementation::invoke(uint8_t queue_id, uint32_t N, uint32_t C, uint32_t H, uint32_t W, float fill_value, const ttnn::Tensor& any, const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(any.memory_config());
    if(any.get_layout() == ttnn::TILE_LAYOUT) return operation::run_without_autoformat(FillRM{N, C, H, W, 0, 0, fill_value, fill_value, output_memory_config}, {any}, {}, {}, queue_id).at(0);

    ttnn::Tensor padded_tensor = any;
    Tensor output_tensor;

    // Check if the tensor is in ROW_MAJOR_LAYOUT and if padding is required
    if (any.get_layout() == ttnn::ROW_MAJOR_LAYOUT) {
        // Padding dimensions for H and W must be multiples of 32
        uint32_t padded_H = (H % 32 == 0) ? H : ((H / 32) + 1) * 32;
        uint32_t padded_W = (W % 32 == 0) ? W : ((W / 32) + 1) * 32;

        if (padded_H != H || padded_W != W) {
            // Create padding vector: (before, after) pairs for each dimension
            std::vector<std::pair<uint32_t, uint32_t>> padding_vec = {
                {0, 0},  // No padding for batch dimension N
                {0, 0},  // No padding for channel dimension C
                {0, padded_H - H},  // Padding for height (H)
                {0, padded_W - W}   // Padding for width (W)
            };

            // Apply padding
            padded_tensor = ttnn::pad(
                queue_id,
                any,
                padding_vec,
                fill_value,
                false,
                memory_config
            );
        }
        else return operation::run_without_autoformat(FillRM{N, C, H, W, 0, 0, fill_value, fill_value, output_memory_config}, {padded_tensor}, {}, {}, queue_id).at(0);
        output_tensor = operation::run_without_autoformat(FillRM{N, C, H, W, 0, 0, fill_value, fill_value, output_memory_config}, {padded_tensor}, {}, {}, queue_id).at(0);
        // If the tensor was padded, slice it back to the original dimensions
        // Define the slicing boundaries (keeping the original H and W)
        std::array<uint32_t, 4> begins = {0, 0, 0, 0};  // No slicing on N, C dimensions
        std::array<uint32_t, 4> ends = {N, C, H, W};    // Slice to the original H and W

        // Slice the output tensor back to the original dimensions
        output_tensor = ttnn::operations::data_movement::SliceOperation::invoke(
            queue_id,
            output_tensor,
            begins,
            ends,
            {1, 1, 1, 1},      // No stride
            memory_config
        );
    }
    return output_tensor;
}

ttnn::Tensor FullOperation::invoke(uint8_t queue_id, const std::vector<uint32_t>& shape, float fill_value, ttnn::Device* device, const std::optional<ttnn::MemoryConfig>& memory_config_arg) {
    TT_FATAL(shape.size() <= 4, "Shape must be a vector of <= 4 integers [N, C, H, W]");

    std::vector<uint32_t> padded_shape = shape;
    while (padded_shape.size() < 4) {
        padded_shape.insert(padded_shape.begin(), 1);
    }
    uint32_t N = padded_shape[0], C = padded_shape[1], H = padded_shape[2], W = padded_shape[3];
    auto ttnn_shape = ttnn::SimpleShape(padded_shape);
    const MemoryConfig memory_config = memory_config_arg.value_or(ttnn::DRAM_MEMORY_CONFIG);
    ttnn::Tensor any = create_device_tensor(ttnn_shape, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, device, memory_config);
    ttnn::Tensor filled_tensor = FillImplementation::invoke(queue_id, N, C, H, W, fill_value, any, memory_config_arg);
    ttnn::Tensor squeezed_tensor = filled_tensor;


    //squeeze the tensor back to original shape
    auto new_shape = squeezed_tensor.get_logical_shape();
    while (new_shape != shape) {
        squeezed_tensor = ttnn::operations::data_movement::SqueezeOperation::invoke(squeezed_tensor, 0);
        new_shape = squeezed_tensor.get_logical_shape();
    }
    return squeezed_tensor;
}

ttnn::Tensor FullOperation::invoke(const std::vector<uint32_t>& shape, float fill_value, Device* device, const std::optional<ttnn::MemoryConfig>& memory_config_arg) {
    return invoke(DefaultQueueId, shape, fill_value, device, memory_config_arg);
}

ttnn::Tensor FillOperation::invoke(uint8_t queue_id, float fill_value, const ttnn::Tensor& any, const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto shape = any.get_logical_shape();
    TT_FATAL(shape.rank() <= 4, "Shape must be a vector of less than 4 integers [N, C, H, W]");

    auto padded_shape = shape;
    ttnn::Tensor mutable_tensor = any;

    //unsqueeze the any tensor if padded shape length is different from original shape length
    if (padded_shape.rank() < 4) {

        //unsqueeze to 4D tensor
        while (padded_shape.rank() < 4) {
            mutable_tensor = ttnn::unsqueeze(mutable_tensor, 0);
            padded_shape = mutable_tensor.get_logical_shape();
        }

        //fill operation
        uint32_t N = padded_shape[0], C = padded_shape[1], H = padded_shape[2], W = padded_shape[3];
        mutable_tensor = FillImplementation::invoke(queue_id, N, C, H, W, fill_value, mutable_tensor, memory_config);

        //squeeze the tensor back to original shape
        while(padded_shape != shape) {
            mutable_tensor = ttnn::squeeze(mutable_tensor, 0);
            padded_shape = mutable_tensor.get_logical_shape();
        }

        //return modified tensor
        return mutable_tensor;
    }

    //or just perform operation and return tensor
    uint32_t N = padded_shape[0], C = padded_shape[1], H = padded_shape[2], W = padded_shape[3];
    return FillImplementation::invoke(queue_id, N, C, H, W, fill_value, any, memory_config);
}

ttnn::Tensor FillOperation::invoke(float fill_value, const ttnn::Tensor& any, const std::optional<ttnn::MemoryConfig>& memory_config_arg) {
    return invoke(DefaultQueueId, fill_value, any, memory_config_arg);
}

}  // namespace ttnn::operations::data_movement
