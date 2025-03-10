// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "view.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <ttnn/operations/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::reshape {

static MemoryConfig infer_output_memory_config(
    const MemoryConfig& input_memory_config, const ttnn::Shape& output_padded_shape) {
    if (input_memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        auto shard_spec = input_memory_config.shard_spec.value();
        shard_spec.shape[1] = output_padded_shape[-1];  // update output shard to match new shard width
        return MemoryConfig{input_memory_config.memory_layout, input_memory_config.buffer_type, shard_spec};
    } else {
        return input_memory_config;
    }
}

Tensor tensor_reshape(
    const Tensor& input_tensor, const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) {
    ZoneScoped;
    tt::tt_metal::GraphTracker::instance().track_function_start(
        "Tensor::reshape", input_tensor, new_logical_shape, new_padded_shape);

    const auto output_memory_config = infer_output_memory_config(input_tensor.memory_config(), new_padded_shape);
    auto new_spec = ttnn::TensorSpec(
        new_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.get_dtype(),
            input_tensor.get_tensor_spec().page_config(),
            output_memory_config,
            new_logical_shape,
            new_padded_shape));

    auto output = std::visit(
        [&input_tensor, &new_spec, &new_logical_shape, &new_padded_shape](auto&& storage) -> Tensor {
            using T = std::decay_t<decltype(storage)>;
            const auto& tensor = input_tensor;
            if constexpr (std::is_same_v<T, tt::tt_metal::MultiDeviceHostStorage>) {
                auto updated_storage = std::get<T>(tensor.get_storage());
                for (int i = 0; i < updated_storage.specs.size(); i++) {
                    const auto& prev_spec = updated_storage.specs[i];
                    TensorSpec spec(
                        new_logical_shape,
                        TensorLayout::fromPaddedShape(
                            prev_spec.data_type(),
                            prev_spec.page_config(),
                            prev_spec.memory_config(),
                            new_logical_shape,
                            new_padded_shape));
                    updated_storage.specs[i] = spec;
                }
                return Tensor(updated_storage, new_spec);
            }
            /*
            if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                MultiDeviceStorage updated_storage = std::get<T>(tensor.get_storage());
                std::unordered_map<int, ttnn::TensorSpec> new_specs;
                for (auto device_id : updated_storage.ordered_device_ids) {
                    const auto& prev_spec = updated_storage.specs.at(device_id);
                    TensorSpec spec(
                        new_logical_shape,
                        TensorLayout::fromPaddedShape(
                            prev_spec.data_type(),
                            prev_spec.page_config(),
                            prev_spec.memory_config(),
                            new_logical_shape,
                            new_padded_shape));
                    new_specs.insert({device_id, spec});
                }
                updated_storage.specs = new_specs;
                return Tensor(updated_storage, new_spec);
            }
            */
            if constexpr (std::is_same_v<T, tt::tt_metal::DeviceStorage>) {
                if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
                    if (tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
                        tt::tt_metal::DeviceStorage device_storage = std::get<T>(tensor.get_storage());
                        auto device_buffer = device_storage.get_buffer();
                        const auto& tensor_spec = tensor.tensor_spec();
                        auto page_size_bytes = tensor_spec.compute_page_size_bytes();
                        device_buffer->set_page_size(page_size_bytes);
                        return Tensor(device_storage, new_spec);
                    } else {
                        tt::tt_metal::DeviceStorage device_storage = std::get<T>(tensor.get_storage());
                        auto device_buffer = device_storage.get_buffer();
                        tt::tt_metal::ShardSpecBuffer shard_spec_buffer = device_buffer->shard_spec();

                        auto shard_spec = shard_spec_buffer.tensor_shard_spec;
                        auto shard_shape = shard_spec.shape;

                        uint32_t mul_div;
                        if (new_logical_shape[-1] == 0 || shard_shape[1] == 0) {
                            mul_div = 0;
                        } else {
                            mul_div = new_logical_shape[-1] > shard_shape[1] ? (new_logical_shape[-1] / shard_shape[1])
                                                                             : (shard_shape[1] / new_logical_shape[-1]);
                        }

                        shard_spec.shape[0] = new_logical_shape[-1] > shard_shape[1] ? shard_shape[0] / mul_div
                                                                                     : shard_shape[0] * mul_div;
                        shard_spec.shape[1] = new_logical_shape[-1];

                        shard_spec_buffer.page_shape = {1, new_logical_shape[-1]};
                        shard_spec_buffer.tensor2d_shape_in_pages = {shard_spec.shape[0], 1};
                        shard_spec_buffer.set_shard_spec(shard_spec);

                        device_buffer->set_shard_spec(shard_spec_buffer);

                        MemoryConfig mem_config = input_tensor.memory_config();
                        mem_config.shard_spec = shard_spec;

                        auto upd_spec = ttnn::TensorSpec(
                            new_logical_shape,
                            TensorLayout::fromPaddedShape(
                                input_tensor.get_dtype(),
                                input_tensor.get_tensor_spec().page_config(),
                                mem_config,
                                new_logical_shape,
                                new_padded_shape));

                        return Tensor(device_storage, upd_spec);
                    }
                } else {
                    return Tensor(tensor.get_storage(), new_spec);
                }
            } else {
                return Tensor(tensor.get_storage(), new_spec);
            }
        },
        input_tensor.get_storage());
    output = tt::tt_metal::set_tensor_id(output);
    tt::tt_metal::GraphTracker::instance().track_function_end(output);
    return output;
}

ttnn::Tensor ViewOperation::invoke(
    const ttnn::Tensor& tensor, const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
    return tensor_reshape(tensor, logical_shape, padded_shape);
}

ttnn::Tensor ViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    return tensor_reshape(tensor, shape, shape);
}

}  // namespace ttnn::operations::experimental::reshape
