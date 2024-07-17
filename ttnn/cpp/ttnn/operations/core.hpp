// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/tensor/tensor.hpp"
#include "ttnn/experimental/tensor/tensor_utils.hpp"
#include "ttnn/experimental/tensor/types.hpp"
#include "ttnn/experimental/tt_dnn/op_library/copy/copy_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/move/move_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/trace/trace.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/op_library/to_layout/to_layout_op.hpp"
#include "ttnn/op_library/to_dtype/to_dtype_op.hpp"
#include "ttnn/op_library/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {
namespace core {

inline ttnn::Tensor reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape) {

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
                    return tensor.reshape(shape.value());
                }
            } else {
                return tensor.reshape(shape.value());
            }
        } else if (tensor_shape.rank() >= 2 and shape.rank() >= 2) {
            // Handle the case when the tensor is not contiguous but the last two dimensions are the same and so reshape
            // is possible
            if (tensor_shape[-1] == shape[-1] and tensor_shape[-2] == shape[-2] and
                tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1] and
                tensor_shape.with_tile_padding()[-2] == shape.with_tile_padding()[-2]) {
                return tensor.reshape(shape.value());
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
                return tensor.reshape(shape.value());
            }
        } else {
            return tensor.reshape(shape.value());
        }
    }
    TT_THROW("Unable to reshape given tensor!");
}

template <std::size_t Rank>
inline ttnn::Tensor reshape(const ttnn::Tensor& tensor, const std::array<int32_t, Rank>& shape) {
    std::int64_t new_volume = 1;
    std::int64_t index_of_negative_1 = -1;
    for (auto index = 0; index < Rank; ++index) {
        if (shape[index] == -1) {
            if (index_of_negative_1 != -1) {
                TT_THROW("Shape cannot have more than 1 elements that is set to -1!");
            }
            index_of_negative_1 = index;
        }
        new_volume *= shape[index];
    }

    std::array<std::uint32_t, Rank> new_shape{};
    std::copy(shape.begin(), shape.end(), new_shape.begin());
    if (new_volume < 0) {
        const auto volume = tensor.get_shape().with_tile_padding().volume();
        new_shape[index_of_negative_1] = volume / (-new_volume);
    }
    return reshape(tensor, ttnn::Shape(new_shape));
}

inline ttnn::Tensor unsqueeze_to_4D(const ttnn::Tensor& tensor) {
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

inline ttnn::Tensor squeeze_from_4D(const ttnn::Tensor& tensor, const int rank) {
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

inline ttnn::Tensor to_device(
    const ttnn::Tensor& tensor, Device* device, const std::optional<MemoryConfig>& memory_config) {
    return tensor.to(device, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
}

inline ttnn::Tensor to_device(
    const ttnn::Tensor& tensor, DeviceMesh* device_mesh, const std::optional<MemoryConfig>& memory_config) {
    return tensor.to(device_mesh, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
}

inline ttnn::Tensor allocate_tensor_on_device(
    const Shape& shape, DataType data_type, Layout layout, Device *device, const std::optional<MemoryConfig>& memory_config) {
    return tt::tt_metal::allocate_tensor_on_device(shape, data_type, layout, device, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
}

inline ttnn::Tensor allocate_tensor_on_device(
    const Shape& shape, DataType data_type, Layout layout, DeviceMesh *device_mesh, const std::optional<MemoryConfig>& memory_config) {
    return tt::tt_metal::allocate_tensor_on_device(shape, data_type, layout, device_mesh, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
}

inline void copy_host_to_device_tensor(ttnn::Tensor host_tensor, ttnn::Tensor device_tensor, uint8_t cq_id = 0) {
    tt::tt_metal::write_tensor(host_tensor, device_tensor, cq_id);
}

inline ttnn::Tensor from_device(const ttnn::Tensor& tensor, bool blocking = true) { return tensor.cpu(blocking); }

inline void deallocate(Tensor& tensor, bool force = true) { tensor.deallocate(force); }

inline Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    if (input_tensor.is_sharded()) {
        return move_sharded(input_tensor, memory_config);
    } else {
        return move(input_tensor, memory_config);
    }
}

// Trace APIs - Single Device
inline uint32_t begin_trace_capture(Device* device, const uint8_t cq_id) {
    uint32_t tid = Trace::next_id();
    device->push_work(
        [device, cq_id, tid] () mutable {
            device->begin_trace(cq_id, tid);
        });
    return tid;
}

inline void end_trace_capture(Device* device, const uint32_t tid, const uint8_t cq_id) {
    device->push_work(
        [device, cq_id, tid] () mutable {
            device->end_trace(cq_id, tid);
        }
    );
}

inline void execute_trace(Device* device, const uint32_t tid, const uint8_t cq_id, bool blocking) {
    // If blocking, ensure that worker thread blocks until trace is completed
    device->push_work(
        [device, cq_id, tid, blocking] () mutable {
            device->replay_trace(cq_id, tid, blocking);
        }
    );
    // If blocking, wait until worker threads have completed
    if (blocking) {
        device->synchronize();
    }
}

inline void release_trace(Device* device, const uint32_t tid) {
    device->push_work(
        [device, tid] () mutable {
            device->release_trace(tid);
        }
    );
}

// Trace APIs - Multi Device
inline uint32_t begin_trace_capture(DeviceMesh* device, const uint8_t cq_id = 0) {
    auto workers = device->get_devices();
    uint32_t tid = Trace::next_id();
    for (auto& worker : workers) {
        worker->push_work(
            [worker, cq_id, tid] () mutable {
                worker->begin_trace(cq_id, tid);
            });
    }
    return tid;
}

inline void end_trace_capture(DeviceMesh* device, const uint32_t tid, const uint8_t cq_id = 0) {
    auto workers = device->get_devices();
    for (auto& worker : workers) {
        worker->push_work(
            [worker, cq_id, tid] () mutable {
                worker->end_trace(cq_id, tid);
            });
    }
}

inline void execute_trace(DeviceMesh* device, const uint32_t tid, const uint8_t cq_id = 0, bool blocking = true) {
    auto workers = device->get_devices();
    // If blocking, ensure that each worker thread blocks until device-local trace is completed
    for (auto& worker : workers) {
        worker->push_work(
            [worker, cq_id, tid, blocking] () mutable {
                worker->replay_trace(cq_id, tid, blocking);
            });
    }
    // If blocking, wait until worker threads have completed
    if (blocking) {
        for (auto& worker : workers) {
            worker->synchronize();
        }
    }
}

inline void release_trace(DeviceMesh* device, const uint32_t tid) {
    auto workers = device->get_devices();
    for (auto& worker : workers) {
        worker->push_work(
            [worker, tid] () mutable {
                worker->release_trace(tid);
            });
    }
}

}  // namespace core
}  // namespace operations

using operations::core::deallocate;
using operations::core::from_device;
using operations::core::reallocate;
using operations::core::reshape;
using operations::core::squeeze_from_4D;
using operations::core::to_device;
using operations::core::unsqueeze_to_4D;

constexpr auto to_dtype = ttnn::register_operation<ttnn::operations::core::ToDtype>("ttnn::to_dtype");
constexpr auto to_memory_config = ttnn::register_operation<ttnn::operations::core::ToMemoryConfig>("ttnn::to_memory_config");
constexpr auto to_layout = ttnn::register_operation<ttnn::operations::core::ToLayout>("ttnn::to_layout");

}  // namespace ttnn
