// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "reshape.hpp"
#include "tt_metal/common/constants.hpp"
#include <ttnn/operations/numpy/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "device/reshape_op.hpp"

namespace ttnn::operations::data_movement {


namespace detail {

    static Tensor manual_insertion(
        const Tensor& input_tensor,
        const ttnn::Shape& shape,
        Device* device,
        const MemoryConfig& output_mem_config
        ) {
        TT_ASSERT(input_tensor.get_layout() == Layout::ROW_MAJOR);
        TT_ASSERT(shape.logical_shape().volume() == input_tensor.get_logical_volume(),
            "Required shape volume ({}) must match old shape volume ({})", shape.logical_shape().volume(), input_tensor.get_logical_volume());
        auto device_buffer = input_tensor.device_buffer();
        uint32_t size_in_bytes = device_buffer->size();
        std::vector<uint16_t> data_vec;
        const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
            data_vec.resize(size_in_bytes / sizeof(uint16_t));
            tt::tt_metal::tensor_impl::read_data_from_device_buffer<uint16_t>(
                input_tensor.device()->command_queue(), device_buffer, data_vec.data(), true);
        } else {
            tt::tt_metal::tensor_impl::read_data_from_device_buffer<uint16_t>(device_buffer, data_vec);
        }
        auto owned_buffer = owned_buffer::create<uint16_t>(std::move(data_vec));
        auto output = Tensor(OwnedStorage{owned_buffer}, shape, DataType::BFLOAT16, Layout::ROW_MAJOR).to(Layout::ROW_MAJOR);
        if (device != nullptr) {
            output = output.to(device, output_mem_config);
        }
        return output;
    }
}


ttnn::Tensor ReshapeOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    ttnn::Shape output_shape,
    const std::optional<MemoryConfig>& memory_config_arg) {
    using namespace tt::constants;
    auto output_mem_config = memory_config_arg.value_or(input_tensor.memory_config());
    auto padded_output_shape = output_shape.padded_shape();
    // No-op (Will do a tensor copy)
    if (
        ((input_tensor.get_layout() == Layout::TILE or input_tensor.get_layout() == Layout::ROW_MAJOR) && padded_output_shape[3] == input_tensor.get_padded_shape()[3])
    ) {
        // Don't need to do a check here to see the H and W both divisible by 32
        // since handled within the tensor reshape method
        return input_tensor.reshape(output_shape);
    }
    if (input_tensor.get_padded_shape() == padded_output_shape) {
        return ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_mem_config(input_tensor, output_mem_config);
    }
    uint32_t ROW_MAJOR_WIDTH = 8;
    if (input_tensor.get_layout() == Layout::ROW_MAJOR &&
        (input_tensor.get_legacy_shape()[3] % ROW_MAJOR_WIDTH != 0 ||
        padded_output_shape[3] % ROW_MAJOR_WIDTH != 0) &&
        ((padded_output_shape.volume() / padded_output_shape[-1]) % TILE_HEIGHT != 0
        || padded_output_shape[-1] % TILE_WIDTH != 0
        || input_tensor.get_legacy_shape()[-1] % TILE_WIDTH != 0
        || (input_tensor.volume() / input_tensor.get_legacy_shape()[-1]) % TILE_HEIGHT != 0)) {
        TT_FATAL(input_tensor.get_dtype()==DataType::BFLOAT16, "Error");

        return detail::manual_insertion((tt::tt_metal::Tensor)input_tensor, output_shape, input_tensor.device(), output_mem_config);
    }
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    return operation::run(ReshapeDeviceOperation{output_shape, output_mem_config}, {input_tensor}).at(0);

}

ttnn::Tensor ReshapeOperation::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Shape shape,
    const std::optional<MemoryConfig>& memory_config) {
    return invoke(DefaultQueueId, input_tensor, shape, memory_config);
}

ttnn::Tensor ReshapeOperation::invoke(const ttnn::Tensor& input_tensor, const ttnn::Shape& shape) {
    return invoke(DefaultQueueId, input_tensor, shape, std::nullopt);
}

ttnn::Tensor ReshapeOperation::invoke(uint8_t queue_id, const ttnn::Tensor& input_tensor, const std::vector<int32_t> & shape_vector, const std::optional<MemoryConfig>& memory_config_arg) {
    return invoke(queue_id, input_tensor, ttnn::Shape(infer_dims_for_reshape(input_tensor, shape_vector).as_vector()), memory_config_arg);
}

ttnn::Tensor ReshapeOperation::invoke(const ttnn::Tensor& input_tensor, const std::vector<int32_t>& shape_vector, const std::optional<MemoryConfig>& memory_config_arg) {
    return invoke(DefaultQueueId, input_tensor, shape_vector, memory_config_arg);
}

ttnn::Tensor ReshapeOperation::invoke(const ttnn::Tensor& input_tensor, const std::vector<int32_t>& shape_vector) {
    return invoke(input_tensor, shape_vector, std::nullopt);
}

} // ttnn::operations::data_movement namespace
