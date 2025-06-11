// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_utils.hpp"

#include <tt_stl/overloaded.hpp>

#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/types.hpp"

#include <tracy/Tracy.hpp>

namespace tt {
namespace tt_metal {

ttnn::Shape infer_dims_for_reshape(const Tensor& tensor, tt::stl::Span<const int32_t> shape) {
    int64_t old_volume = tensor.logical_volume();
    int64_t new_volume = 1;
    int64_t index_of_negative_1 = -1;
    bool has_zero = false;
    for (auto index = 0; index < shape.size(); ++index) {
        if (shape[index] == -1) {
            if (index_of_negative_1 != -1) {
                std::string error_msg = "Shape cannot have more than 1 elements that is set to -1! Shape used: (";
                for (auto& s : shape) {
                    error_msg += std::to_string(s) + ",";
                }
                error_msg += ")";
                TT_THROW("{}", error_msg);
            }
            index_of_negative_1 = index;
        } else {
            if (shape[index] == 0) {
                has_zero = true;
            }
            new_volume *= shape[index];
        }
    }
    if (has_zero && index_of_negative_1 != -1) {
        std::string error_msg = "cannot reshape tensor of 0 elements into shape (";
        for (auto& s : shape) {
            error_msg += std::to_string(s) + ",";
        }
        error_msg += ") because the unspecified dimension size -1 can be any value and is ambiguous";
        TT_THROW("{}", error_msg);
    }

    ttnn::SmallVector<uint32_t> new_shape(shape.size());
    std::copy(shape.begin(), shape.end(), new_shape.begin());
    if (index_of_negative_1 == -1) {
        TT_FATAL(new_volume == old_volume, "Invalid arguments to reshape");
    } else {
        TT_FATAL(old_volume % new_volume == 0, "Invalid arguments to reshape");
        new_shape[index_of_negative_1] = old_volume / new_volume;
    }

    return ttnn::Shape(std::move(new_shape));
}

int compute_flat_indices(tt::stl::Span<const int> indices, tt::stl::Span<const uint32_t> strides) {
    int flat_index = 0;
    for (auto i = 0; i < indices.size(); i++) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
};

std::size_t compute_buffer_size(const ttnn::Shape& shape, DataType data_type, const Tile& tile) {
    const size_t volume = shape.volume();
    auto tile_hw = tile.get_tile_hw();
    if (data_type == DataType::BFLOAT8_B) {
        auto tile_size_bytes = tile.get_tile_size(DataFormat::Bfp8_b);
        TT_ASSERT(volume % tile_hw == 0);
        const auto bfloat8_b_volume = volume / tile_hw * tile_size_bytes;
        TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
        return bfloat8_b_volume / sizeof(std::uint32_t);
    }
    if (data_type == DataType::BFLOAT4_B) {
        auto tile_size_bytes = tile.get_tile_size(DataFormat::Bfp4_b);
        TT_ASSERT(volume % tile_hw == 0);
        const auto bfloat4_b_volume = volume / tile_hw * tile_size_bytes;
        TT_ASSERT(volume % sizeof(std::uint32_t) == 0);
        return bfloat4_b_volume / sizeof(std::uint32_t);
    }
    return volume;
}

bool is_arch_gs(const tt::ARCH& arch) { return arch == tt::ARCH::GRAYSKULL; }

bool is_arch_whb0(const tt::ARCH& arch) { return arch == tt::ARCH::WORMHOLE_B0; }

bool is_cpu_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::HOST; }

bool is_multi_device_host_tensor(const Tensor& tensor) {
    return tensor.storage_type() == StorageType::MULTI_DEVICE_HOST;
}

bool is_device_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

Tensor transform(const Tensor& tensor, const std::function<Tensor(const Tensor&)>& transform_func) {
    TT_FATAL(is_multi_device_host_tensor(tensor), "transform only supports multi-device host tensors");
    const auto& storage = std::get<MultiDeviceHostStorage>(tensor.storage());

    std::vector<TensorSpec> transformed_specs;
    std::vector<HostBuffer> transformed_buffers;
    transformed_buffers.reserve(storage.num_buffers());
    transformed_specs.reserve(storage.num_buffers());
    for (size_t i = 0; i < storage.num_buffers(); i++) {
        Tensor transformed_tensor_shard = transform_func(Tensor(storage.get_buffer(i), tensor.tensor_spec()));
        transformed_specs.push_back(transformed_tensor_shard.tensor_spec());
        auto* host_storage = std::get_if<HostStorage>(&transformed_tensor_shard.storage());
        TT_FATAL(host_storage != nullptr, "transform function must return a host tensor");
        transformed_buffers.push_back(std::move(host_storage->buffer));
    }
    TensorSpec reference_spec = transformed_specs.front();
    MultiDeviceHostStorage transformed_storage(std::move(transformed_buffers));
    return Tensor(std::move(transformed_storage), reference_spec, tensor.distributed_tensor_config());
}

void apply(const Tensor& tensor, const std::function<void(const Tensor&)>& callable) {
    TT_FATAL(is_multi_device_host_tensor(tensor), "apply only supports multi-device host tensors");
    const auto& storage = std::get<MultiDeviceHostStorage>(tensor.storage());
    for (size_t i = 0; i < storage.num_buffers(); i++) {
        callable(Tensor(storage.get_buffer(i), tensor.tensor_spec()));
    }
}

ShardDivisionSpec compute_shard_division_spec(const Shape2D& shape, const Shape2D& shard_shape) {
    const auto num_shards_height = tt::div_up(shape.height(), shard_shape.height());
    const auto last_shard_height =
        shape.height() % shard_shape.height() > 0 ? shape.height() % shard_shape.height() : shard_shape.height();
    const auto num_shards_width = tt::div_up(shape.width(), shard_shape.width());
    const auto last_shard_width =
        shape.width() % shard_shape.width() > 0 ? shape.width() % shard_shape.width() : shard_shape.width();

    return ShardDivisionSpec{num_shards_height, last_shard_height, num_shards_width, last_shard_width};
};

}  // namespace tt_metal

}  // namespace tt
