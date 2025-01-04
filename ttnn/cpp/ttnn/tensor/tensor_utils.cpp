// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/types.hpp"

namespace tt {

namespace tt_metal {

const ttnn::SimpleShape infer_dims_for_reshape(const Tensor& tensor, tt::stl::Span<const int32_t> shape) {
    int64_t old_volume = tensor.get_logical_volume();
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
        for(auto & s: shape) {
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

    return ttnn::SimpleShape(std::move(new_shape));
}

bool is_arch_gs(const tt::ARCH& arch) { return arch == tt::ARCH::GRAYSKULL; }

bool is_arch_whb0(const tt::ARCH& arch) { return arch == tt::ARCH::WORMHOLE_B0; }

bool is_cpu_tensor(const Tensor& tensor) {
    return tensor.storage_type() == StorageType::OWNED || tensor.storage_type() == StorageType::BORROWED;
}

bool is_device_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::DEVICE; }

Tensor transform(const Tensor& tensor, std::function<Tensor(const Tensor&)> transform_func) {
    auto input_tensors = ttnn::distributed::get_tensors_from_multi_device_storage(tensor);
    std::vector<Tensor> output_tensors(input_tensors.size());
    std::transform(input_tensors.begin(), input_tensors.end(), output_tensors.begin(), [&](const auto& device_tensor) {
        return transform_func(device_tensor);
    });
    return ttnn::distributed::create_multi_device_tensor(
        output_tensors, tensor.storage_type(), ttnn::distributed::get_distributed_tensor_config_from_tensor(tensor));
}

void apply(const Tensor& tensor, const std::function<void(const Tensor&)>& callable) {
    auto input_tensors = ttnn::distributed::get_tensors_from_multi_device_storage(tensor);
    for (const auto& device_tensor : input_tensors) {
        callable(device_tensor);
    }
}

std::vector<Device*> get_devices(const Tensor& tensor) {
    std::vector<Device*> devices;
    if (tensor.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE) {
        TT_ASSERT(
            std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(tensor.get_storage()));
        const auto& tensor_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
        for (int i = 0; i < tensor_storage.ordered_device_ids.size(); ++i) {
            auto device_id = tensor_storage.ordered_device_ids[i];
            devices.push_back(tensor_storage.get_buffer_for_device_id(device_id)->device());
        }
        return devices;
    } else {
        TT_THROW("Tensor is not a multi-device tensor");
    }
}

uint32_t num_buffers_in_tensor(const Tensor& tensor) {
    if (std::holds_alternative<MultiDeviceStorage>(tensor.get_storage())) {
        auto device_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
        return device_storage.num_buffers();
    } else if (std::holds_alternative<MultiDeviceHostStorage>(tensor.get_storage())) {
        auto host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());
        return host_storage.num_buffers();
    } else if (
        std::holds_alternative<DeviceStorage>(tensor.get_storage()) ||
        std::holds_alternative<OwnedStorage>(tensor.get_storage()) ||
        std::holds_alternative<BorrowedStorage>(tensor.get_storage())) {
        return 1;
    } else {
        TT_THROW("num_buffers_in_tensor only supports multi-device or device tensors");
    }
}

Tensor get_shard_for_device(const Tensor& tensor, Device* target_device, std::optional<int> buffer_index) {
    ZoneScopedN("GetShardForDevice");
    Tensor shard = Tensor();
    auto& storage = tensor.tensor_attributes->storage;
    std::visit(
        [target_device, buffer_index, &tensor, &shard](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            // Stalling reads for tensor data-type and layout are needed here
            // since some worker might have raced ahead to these lookups, while
            // another worker is populating this metadata.
            const Tile tile = tensor.get_tensor_spec().tile();
            if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                shard = Tensor{
                    DeviceStorage{s.get_buffer_for_device(target_device)},
                    s.get_tensor_shape_for_device(target_device),
                    tensor.get_dtype(),
                    tensor.get_layout(),
                    tile};
            } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                shard = Tensor{
                    OwnedStorage{s.get_buffer(buffer_index.value())},
                    s.get_tensor_shape(buffer_index.value()),
                    tensor.get_dtype(),
                    tensor.get_layout(),
                    tile};
            } else if constexpr (
                std::is_same_v<T, OwnedStorage> || std::is_same_v<T, BorrowedStorage> ||
                std::is_same_v<T, DeviceStorage>) {
                shard = tensor;
            } else {
                TT_THROW("get_shard_for_device only supports multi-device or device tensors");
            }
        },
        storage);
    return shard;
}

void insert_buffer_and_shape_for_device(
    Device* target_device, const Tensor& shard, Tensor& tensor_to_modify, std::optional<int> buffer_index) {
    ZoneScopedN("InsertBufferAndShapeForDevice");
    std::visit(
        [target_device, &shard, buffer_index](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                s.insert_buffer_and_shape_for_device(
                    buffer_index.value(),
                    std::get<OwnedStorage>(shard.tensor_attributes->storage).get_buffer(),
                    shard.tensor_attributes->tensor_spec.shape());
            } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                s.insert_buffer_and_shape_for_device(
                    target_device,
                    std::get<DeviceStorage>(shard.tensor_attributes->storage).get_buffer(),
                    shard.tensor_attributes->tensor_spec.shape());
            } else if constexpr (std::is_same_v<T, OwnedStorage>) {
                s.insert_buffer(std::get<OwnedStorage>(shard.tensor_attributes->storage).get_buffer());
            } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                s.insert_buffer(std::get<DeviceStorage>(shard.tensor_attributes->storage).get_buffer());
            } else {
                TT_THROW("Unsupported storage in insert_buffer_and_shape_for_device");
            }
        },
        tensor_to_modify.tensor_attributes->storage);
}

Tensor copy_borrowed_tensor_in_async_mode(Device* worker, const Tensor& tensor) {
    // When using async mode, tensors with borrowed storage cannot be passed to workers.
    // They need to be copied to owned storage before being passed to the worker.
    ZoneScopedN("ConvertBorrowedToOwned");
    // Tensor has workers (on device) or runtime mode is synchronous or tensor has multiple buffers.
    // No need to check for borrowed storage.
    if (worker->get_worker_mode() == WorkExecutorMode::SYNCHRONOUS or
        tensor.tensor_attributes->num_shards_to_be_populated > 1) {
        return tensor;
    }

    if (tensor.storage_type() == StorageType::BORROWED) {
        ZoneScopedN("CopyBorrowedStorage");
        auto borrowed_buffer = std::get<BorrowedStorage>(tensor.get_storage()).buffer;
        Tensor owned_tensor;
        std::visit(
            [&owned_tensor, &tensor](auto&& buffer) {
                using BorrowedStorageType = std::vector<std::decay_t<decltype(*(buffer.begin()))>>;
                auto owned_buf = owned_buffer::create(BorrowedStorageType(buffer.begin(), buffer.end()));
                owned_tensor = Tensor(OwnedStorage{owned_buf}, tensor.get_tensor_spec());
            },
            borrowed_buffer);
        return owned_tensor;
    }
    return tensor;
}

Size get_2d_shape(const ttnn::SimpleShape& shape) {
    size_t width = shape[-1];
    size_t height = shape.volume() / width;
    return Size{height, width};
}

ShardDivisionSpec compute_shard_division_spec(const Size& shape, const Size& shard_shape) {
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
