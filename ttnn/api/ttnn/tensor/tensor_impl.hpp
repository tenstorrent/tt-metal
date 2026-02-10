// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>

#include <tt-metalium/mesh_device.hpp>

// NOLINTBEGIN(misc-include-cleaner)
// Needs shape.hpp to export ttnn::Shape alias to tt_metal::Shape.
#include "ttnn/tensor/shape/shape.hpp"
// Forward include - re-exports tt-metalium tensor_impl APIs for TTNN users.
#include <tt-metalium/experimental/tensor/details/tensor_impl.hpp>
// NOLINTEND(misc-include-cleaner)

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

namespace tt::tt_metal::tensor_impl {

// Empty structs to facilitate Tensor template logic.
struct bfloat4_b {};
struct bfloat8_b {};

// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================

std::shared_ptr<distributed::MeshBuffer> allocate_device_buffer(
    distributed::MeshDevice* mesh_device, const TensorSpec& tensor_spec);

HostBuffer allocate_host_buffer(const TensorSpec& tensor_spec);

// ======================================================================================
//                                         .to_host() and .to_device()
// ======================================================================================

Tensor to_host(const Tensor& tensor, bool blocking = true, std::optional<QueueId> cq_id = std::nullopt);

// TODO: Move this to tt_metal
HostTensor to_host(distributed::MeshCommandQueue& queue, const DeviceTensor& tensor, bool blocking = true);

void copy_to_host(
    const Tensor& device_tensor,
    Tensor& host_tensor,
    bool blocking = true,
    std::optional<QueueId> cq_id = std::nullopt);

// TODO: Move this to tt_metal
void copy_to_host(
    distributed::MeshCommandQueue& queue,
    const DeviceTensor& device_tensor,
    HostTensor& host_tensor,
    bool blocking = true);

void copy_to_host(
    distributed::MeshCommandQueue& queue,
    const Tensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);

Tensor to_device(
    const Tensor& tensor,
    distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config = std::nullopt,
    std::optional<QueueId> cq_id = std::nullopt);

// TODO: Move this to tt_metal
DeviceTensor to_device(
    distributed::MeshCommandQueue& queue,
    const HostTensor& tensor,
    ttsl::optional_reference<const MemoryConfig> memory_config = std::nullopt);

void copy_to_device(const Tensor& host_tensor, Tensor& device_tensor, std::optional<QueueId> cq_id = std::nullopt);

// TODO: Move this to tt_metal
void copy_to_device(distributed::MeshCommandQueue& queue, const HostTensor& host_tensor, DeviceTensor& device_tensor);

void copy_to_device(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    Tensor& device_tensor,
    const std::optional<BufferRegion>& region = std::nullopt);

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

Tensor to_layout(const Tensor& tensor, Layout target_layout);

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================
Tensor pad(
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value);

Tensor unpad(
    const Tensor& tensor, const tt::tt_metal::Shape& output_tensor_start, const tt::tt_metal::Shape& output_tensor_end);

// ======================================================================================
//                                         Print
// ======================================================================================

std::ostream& operator<<(std::ostream& os, const DataType& dtype);

enum class TensorPrintProfile {
    Empty,
    Short,
    Full,
};

enum class SciMode {
    Enable,
    Disable,
    Default,
};

struct PrintOptions {
    TensorPrintProfile profile = TensorPrintProfile::Short;
    SciMode sci_mode = SciMode::Default;
    int precision = 4;
};

extern PrintOptions TTNN_PRINT_OPTIONS;

std::string to_string(const Tensor& tensor);

Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id);

Tensor to_dtype(const Tensor& input_tensor, DataType dtype);

// Utility to convert runtime DataType to compile-time constant and dispatch the function call
template <typename Func, typename... Args>
auto dispatch(DataType dtype, Func&& func, Args&&... args) {
    switch (dtype) {
        case DataType::BFLOAT16:
            return (std::forward<Func>(func)).template operator()<bfloat16>(std::forward<Args>(args)...);
        case DataType::FLOAT32:
            return (std::forward<Func>(func)).template operator()<float>(std::forward<Args>(args)...);
        case DataType::INT32:
            return (std::forward<Func>(func)).template operator()<int32_t>(std::forward<Args>(args)...);
        case DataType::UINT32:
            return (std::forward<Func>(func)).template operator()<uint32_t>(std::forward<Args>(args)...);
        case DataType::UINT16:
            return (std::forward<Func>(func)).template operator()<uint16_t>(std::forward<Args>(args)...);
        case DataType::UINT8:
            return (std::forward<Func>(func)).template operator()<uint8_t>(std::forward<Args>(args)...);
        case DataType::BFLOAT8_B:
            return (std::forward<Func>(func)).template operator()<bfloat8_b>(std::forward<Args>(args)...);
        case DataType::BFLOAT4_B:
            return (std::forward<Func>(func)).template operator()<bfloat4_b>(std::forward<Args>(args)...);
        default: TT_THROW("Unsupported data type");
    }
}

}  // namespace tt::tt_metal::tensor_impl
