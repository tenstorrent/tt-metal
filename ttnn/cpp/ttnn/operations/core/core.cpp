// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/core/core.hpp"

#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape/reshape.hpp"

namespace ttnn::operations::core {

ttnn::Tensor reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {
    auto tensor_shape = tensor.get_shape();
    if (tensor_shape == shape) {
        return tensor;
    }

    const auto layout = tensor.get_layout();

    if (layout == ttnn::Layout::ROW_MAJOR) {
        if (tensor.is_contiguous()) {
            if (ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
                // Page size depends on the width, so only modify the shape if the width is the same
                if (tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1]) {
                    return tensor.reshape(shape.value);
                }
                //Different page width, need to remake tensor
                else {
                    auto tensor_4d = unsqueeze_to_4D(tensor);
                    const auto shape_4d = tensor_shape.to_rank<4>();
                    return ttnn::reshape_on_device(tensor_4d, shape[0], shape[1], shape[2], shape[3], tensor.memory_config());
                }
            } else {
                return tensor.reshape(shape.value);
            }
        } else if (tensor_shape.rank() >= 2 and shape.rank() >= 2) {
            // Handle the case when the tensor is not contiguous but the last two dimensions are the same and so reshape
            // is possible
            if (tensor_shape[-1] == shape[-1] and tensor_shape[-2] == shape[-2] and
                tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1] and
                tensor_shape.with_tile_padding()[-2] == shape.with_tile_padding()[-2]) {
                return tensor.reshape(shape.value);
            }
        }
    } else if (layout == ttnn::Layout::TILE) {
        const auto new_shape_with_tile_padding = shape.with_tile_padding();
        const auto new_height = new_shape_with_tile_padding[-2];
        const auto new_width = new_shape_with_tile_padding[-1];

        const auto is_tile_multiple = (new_height % ttnn::TILE_SIZE == 0 && new_width % ttnn::TILE_SIZE == 0);
        if (not is_tile_multiple) {
            TT_THROW(
                "Unable to reshape a tensor in TILE_LAYOUT to non-tile height and width! Please convert the tensor to "
                "ROW_MAJOR_LAYOUT first.");
        }

        if (ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
            if (tensor_shape.with_tile_padding()[-1] == new_width) {
                return tensor.reshape(shape.value);
            }
        } else {
            return tensor.reshape(shape.value);
        }
    }
    TT_THROW("Unable to reshape given tensor!");
}

ttnn::Tensor unsqueeze_to_4D(const ttnn::Tensor& tensor) {
    if (is_multi_device_tensor(tensor)) {
        return transform(tensor, [&](const Tensor& device_tensor) { return unsqueeze_to_4D(device_tensor); });
    }

    const auto tensor_shape = tensor.get_shape();
    const auto rank = tensor_shape.rank();
    if (rank == 4) {
        return tensor;
    }
    if (rank > 4) {
        TT_THROW("Tensor rank is greater than 4");
    }

    const auto tensor_shape_4D = tensor_shape.to_rank<4>();
    return ttnn::operations::core::reshape(tensor, tensor_shape_4D);
}

ttnn::Tensor squeeze_from_4D(const ttnn::Tensor& tensor, const int rank) {
    auto shape = tensor.get_shape();
    if (shape.rank() != 4) {
        TT_THROW("Tensor has to be of rank 4!");
    }
    if (rank < 1 or rank > 4) {
        TT_THROW("Cannot use squeeze_from_4D to set the tensor to the rank of {}!", rank);
    }

    for (auto index = 0; index < 4 - rank; ++index) {
        if (shape[index] != 1) {
            TT_THROW("Cannot use squeeze_from_4D to set the tensor to the rank of {}!", rank);
        }
    }

    switch (rank) {
        case 1: return ttnn::operations::core::reshape(tensor, shape.to_rank<1>());
        case 2: return ttnn::operations::core::reshape(tensor, shape.to_rank<2>());
        case 3: return ttnn::operations::core::reshape(tensor, shape.to_rank<3>());
        case 4: return tensor;
        default: TT_THROW("Invalid choice!");
    }
}

ttnn::Tensor to_device(const ttnn::Tensor& tensor, Device* device, const std::optional<MemoryConfig>& memory_config) {
    return tensor.to(device, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
}

ttnn::Tensor to_device(
    const ttnn::Tensor& tensor, MeshDevice* mesh_device, const std::optional<MemoryConfig>& memory_config) {
    return tensor.to(mesh_device, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
}

ttnn::Tensor allocate_tensor_on_device(
    const Shape& shape,
    DataType data_type,
    Layout layout,
    Device* device,
    const std::optional<MemoryConfig>& memory_config) {
    return tt::tt_metal::allocate_tensor_on_device(
        shape, data_type, layout, device, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
}

ttnn::Tensor allocate_tensor_on_device(
    const Shape& shape,
    DataType data_type,
    Layout layout,
    MeshDevice* mesh_device,
    const std::optional<MemoryConfig>& memory_config) {
    return tt::tt_metal::allocate_tensor_on_device(
        shape, data_type, layout, mesh_device, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
}

void copy_host_to_device_tensor(ttnn::Tensor host_tensor, ttnn::Tensor device_tensor, uint8_t cq_id) {
    tt::tt_metal::write_tensor(host_tensor, device_tensor, cq_id);
}

ttnn::Tensor from_device(const ttnn::Tensor& tensor, bool blocking, uint8_t cq_id) { return tensor.cpu(blocking, cq_id); }

void deallocate(Tensor& tensor, bool force) { tensor.deallocate(force); }

Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::move(input_tensor, memory_config);
}

// Trace APIs - Single Device
uint32_t begin_trace_capture(Device* device, const uint8_t cq_id) {
    uint32_t tid = Trace::next_id();
    device->push_work([device, cq_id, tid]() mutable { device->begin_trace(cq_id, tid); });
    return tid;
}

void end_trace_capture(Device* device, const uint32_t tid, const uint8_t cq_id) {
    device->push_work([device, cq_id, tid]() mutable { device->end_trace(cq_id, tid); });
}

void execute_trace(Device* device, const uint32_t tid, const uint8_t cq_id, bool blocking) {
    // If blocking, ensure that worker thread blocks until trace is completed
    device->push_work([device, cq_id, tid, blocking]() mutable { device->replay_trace(cq_id, tid, blocking); });
    // If blocking, wait until worker threads have completed
    if (blocking) {
        device->synchronize();
    }
}

void release_trace(Device* device, const uint32_t tid) {
    device->push_work([device, tid]() mutable { device->release_trace(tid); });
}

// Trace APIs - Multi Device
uint32_t begin_trace_capture(MeshDevice* device, const uint8_t cq_id) {
    auto workers = device->get_devices();
    uint32_t tid = Trace::next_id();
    for (auto& worker : workers) {
        worker->push_work([worker, cq_id, tid]() mutable { worker->begin_trace(cq_id, tid); });
    }
    return tid;
}

void end_trace_capture(MeshDevice* device, const uint32_t tid, const uint8_t cq_id) {
    auto workers = device->get_devices();
    for (auto& worker : workers) {
        worker->push_work([worker, cq_id, tid]() mutable { worker->end_trace(cq_id, tid); });
    }
}

void execute_trace(MeshDevice* device, const uint32_t tid, const uint8_t cq_id, bool blocking) {
    auto workers = device->get_devices();
    // If blocking, ensure that each worker thread blocks until device-local trace is completed
    for (auto& worker : workers) {
        worker->push_work([worker, cq_id, tid, blocking]() mutable { worker->replay_trace(cq_id, tid, blocking); });
    }
    // If blocking, wait until worker threads have completed
    if (blocking) {
        for (auto& worker : workers) {
            worker->synchronize();
        }
    }
}

void release_trace(MeshDevice* device, const uint32_t tid) {
    auto workers = device->get_devices();
    for (auto& worker : workers) {
        worker->push_work([worker, tid]() mutable { worker->release_trace(tid); });
    }
}

}  // namespace ttnn::operations::core
