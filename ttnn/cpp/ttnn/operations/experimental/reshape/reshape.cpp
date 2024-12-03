
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "reshape.hpp"
#include "tt_metal/common/constants.hpp"
#include <functional>
#include <ttnn/operations/numpy/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/data_transfer/data_transfer.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/core/core.hpp"


#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <memory>

#include "common/bfloat16.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/graph/graph_tracking.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/core.hpp"

namespace ttnn::operations::experimental::reshape {
ttnn::Tensor tensor_reshape(const ttnn::Tensor& input_tensor, const ttnn::Shape& new_shape) {
    ZoneScoped;
    GraphTracker::instance().track_function_start("ttnn::experimental::unsafe_view", input_tensor, new_shape);
    const auto& new_padded_shape = new_shape.padded_shape();

    TT_ASSERT(
        input_tensor.volume() == new_padded_shape.volume(),
        "{} != {}",
        input_tensor.volume(),
        new_padded_shape.volume());
    if (input_tensor.get_layout() == Layout::TILE) {
        TT_ASSERT(
            new_padded_shape[-2] % tile.get_tile_shape()[0] == 0 &&
            new_padded_shape[-1] % tile.get_tile_shape()[1] == 0 &&
            "Expected a multiple of 32 for H, W (or -1 evaluating to such) in ttnn::experimental::unsafe_view()!");
    }
    auto output = std::visit(
        [&input_tensor, &new_shape, &tile](auto&& storage) -> Tensor {
            using T = std::decay_t<decltype(storage)>;
            const auto& tensor = input_tensor;
            if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                auto updated_storage = std::get<T>(tensor.get_storage());
                for (int i = 0; i < updated_storage.shapes.size(); i++) {
                    updated_storage.shapes[i] = new_shape;
                }
                return Tensor(updated_storage, new_shape, tensor.get_dtype(), tensor.get_layout(), tile);
            }
            if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                MultiDeviceStorage updated_storage = std::get<T>(tensor.get_storage());
                std::unordered_map<int, ttnn::Shape> new_shapes;

                for (auto device_id : updated_storage.ordered_device_ids) {
                    new_shapes.insert({device_id, new_shape});
                }
                updated_storage.shapes = new_shapes;
                return Tensor(updated_storage, new_shape, tensor.get_dtype(), tensor.get_layout(), tile);
            }
            if constexpr (std::is_same_v<T, DeviceStorage>) {
                if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
                    if (tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
                        DeviceStorage device_storage = std::get<T>(tensor.get_storage());
                        DeviceBuffer device_buffer = device_storage.get_buffer();
                        device_buffer->set_page_size(new_shape[-1] * tensor.element_size());
                        device_storage.insert_buffer(device_buffer);
                        return Tensor(device_storage, new_shape, tensor.get_dtype(), tensor.get_layout(), std::nullopt);
                    } else {
                        DeviceStorage device_storage = std::get<T>(tensor.get_storage());
                        DeviceBuffer device_buffer = device_storage.get_buffer();
                        ShardSpecBuffer shard_spec_buffer = device_buffer->shard_spec();

                        auto shard_spec = shard_spec_buffer.tensor_shard_spec;
                        auto shard_shape = shard_spec.shape;

                        uint32_t mul_div = new_shape[-1] > shard_shape[1] ? (new_shape[-1] / shard_shape[1])
                                                                          : (shard_shape[1] / new_shape[-1]);
                        shard_spec.shape[0] =
                            new_shape[-1] > shard_shape[1] ? shard_shape[0] / mul_div : shard_shape[0] * mul_div;
                        shard_spec.shape[1] = new_shape[-1];

                        shard_spec_buffer.page_shape = {1, new_shape[-1]};
                        shard_spec_buffer.tensor2d_shape = {shard_spec.shape[0], 1};
                        shard_spec_buffer.set_shard_spec(shard_spec);

                        device_buffer->set_shard_spec(shard_spec_buffer);
                        device_storage.insert_buffer(device_buffer);

                        return Tensor(device_storage, new_shape, tensor.get_dtype(), tensor.get_layout(), std::nullopt);
                    }
                } else {
                    const auto tile = input_tensor.get_tensor_spec().tile();
                    return Tensor(tensor.get_storage(), new_shape, tensor.get_dtype(), tensor.get_layout(), tile);
                }
            } else {
                if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
                    return Tensor(
                        tensor.get_storage(), new_shape, tensor.get_dtype(), tensor.get_layout(), std::nullopt);
                } else {
                    const auto tile = input_tensor.get_tensor_spec().tile();
                    return Tensor(tensor.get_storage(), new_shape, tensor.get_dtype(), tensor.get_layout(), tile);
                }
            }
        },
        input_tensor.get_storage());
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}



ttnn::Tensor ReshapeOperation::invoke(const ttnn::Tensor& tensor, const ttnn::SimpleShape& shape) {
    return tensor_reshape(tensor, ttnn::Shape(shape.view()));
}

ttnn::Tensor ReshapeOperation::invoke(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    return tensor_reshape(tensor, shape);
}

}  // namespace ttnn::operations::experimental::reshape
