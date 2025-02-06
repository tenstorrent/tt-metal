// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "adaptive_avg_pool.hpp"
#include "ttnn/operations/core/core.hpp"
#include "cpp/ttnn/operations/data_movement/squeeze/squeeze.hpp"
#include "cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"

namespace ttnn::operations::pool {

inline int64_t start_index(int64_t out_idx, int64_t out_size, int64_t in_size) {
    return (out_idx * in_size) / out_size;
}

inline int64_t end_index(int64_t out_idx, int64_t out_size, int64_t in_size) {
    return ((out_idx + 1) * in_size + out_size - 1) / out_size;
}

template <typename T>
static ttnn::Tensor compute_adaptive_avg_pool(
    const ttnn::Tensor& input, const ttnn::Shape& output_size, const std::optional<MemoryConfig>& mem_config) {
    auto input_mem_config = input.memory_config();
    auto input_shape = input.get_logical_shape();
    auto input_height = input_shape[-3];
    auto input_width = input_shape[-2];
    auto input_channels = input_shape[-1];

    auto output_height = output_size[0];
    auto output_width = output_size[1];
    auto channels = input_channels;

    auto output_shape = input_shape;
    output_shape[-3] = output_height;
    output_shape[-2] = output_width;

    auto output_mem_config = mem_config.value_or(input_mem_config);
    auto physical_volume = input.volume();
    // create output tensor
    auto output_tensor =
        create_device_tensor(output_shape, input.get_dtype(), input.get_layout(), input.device(), output_mem_config);
    auto owned_buffer = tt::tt_metal::owned_buffer::create<T>(output_tensor.volume());
    auto device_buffer = input.device_buffer();
    uint32_t size_in_bytes = device_buffer->size();
    std::vector<T> data_vec(size_in_bytes / sizeof(T));
    tt::tt_metal::tensor_impl::read_data_from_device_buffer<T>(
        input.device()->command_queue(), device_buffer, data_vec.data(), true);

    auto input_buffer = tt::tt_metal::owned_buffer::create<T>(std::move(data_vec));

    auto input_strides = input.strides();
    auto output_strides = output_tensor.strides();
    for (uint32_t n = 0; n < input_shape[0]; ++n) {
        for (uint32_t oh = 0; oh < output_height; ++oh) {
            int64_t ih0 = start_index(oh, output_height, input_height);
            int64_t ih1 = end_index(oh, output_height, input_height);
            int64_t kh = ih1 - ih0;

            for (uint32_t ow = 0; ow < output_width; ++ow) {
                int64_t iw0 = start_index(ow, output_width, input_width);
                int64_t iw1 = end_index(ow, output_width, input_width);
                int64_t kw = iw1 - iw0;
                for (uint32_t c = 0; c < channels; ++c) {
                    auto sum = static_cast<T>(0.0f);
                    for (int64_t ih = ih0; ih < ih1; ++ih) {
                        for (int64_t iw = iw0; iw < iw1; ++iw) {
                            size_t input_idx = n * input_strides[0] + ih * input_strides[1] + iw * input_strides[2] + c;
                            sum = sum + (input_buffer[input_idx]);
                        }
                    }
                    size_t output_idx = n * output_strides[0] + oh * output_strides[1] + ow * output_strides[2] + c;
                    owned_buffer[output_idx] = sum / kh / kw;
                }
            }
        }
    }

    auto output = Tensor(OwnedStorage{owned_buffer}, output_shape, input.get_dtype(), input.get_layout());
    if (input.device() != nullptr) {
        output = output.to_device(input.device(), output_mem_config);
    }

    return output;
}
ttnn::Tensor AdaptiveAvgPool2DOperation::invoke(
    const ttnn::Tensor& input, const ttnn::Shape& output_size, const std::optional<MemoryConfig>& mem_config) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor must be on device");

    ttnn::Tensor output;
    switch (input.get_dtype()) {
        case DataType::FLOAT32: output = compute_adaptive_avg_pool<float>(input, output_size, mem_config); break;
        case DataType::BFLOAT16: output = compute_adaptive_avg_pool<bfloat16>(input, output_size, mem_config); break;
        default: TT_FATAL(false, "Unsupported data type for adaptive_avg_pool2d");
    }
    return output;
}

ttnn::Tensor AdaptiveAvgPool1DOperation::invoke(
    const ttnn::Tensor& input, const ttnn::Shape& output_size, const std::optional<MemoryConfig>& mem_config) {
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    auto input_dims = input.get_logical_shape().size();
    TT_FATAL(input_dims >= 2 && input_dims < 4, "Input tensor must have dimensions between 2 and 4");
    TT_FATAL(output_size.size() == 1, "Output size must be a 1D dimension");
    auto input_tensor = ttnn::unsqueeze(input, 1);
    ttnn::Shape out_size{1, output_size[0]};
    auto output_tensor = ttnn::adaptive_avg_pool2d(input_tensor, out_size, mem_config);
    output_tensor = ttnn::squeeze(output_tensor, 1);
    return output_tensor;
}

}  // namespace ttnn::operations::pool
