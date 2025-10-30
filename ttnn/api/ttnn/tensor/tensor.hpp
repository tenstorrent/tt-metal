// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <random>
#include <tuple>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/tensor_topology.hpp"
#include <tt-metalium/host_buffer.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/device.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/optional_reference.hpp>
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include "types.hpp"
#include "ttnn/tensor/metal_tensor.hpp"
#include "ttnn/experimental/jit/lazy_tensor.hpp"

namespace tt {

namespace tt_metal {

class Tensor {
public:
    std::optional<std::int64_t> tensor_id = std::nullopt;

    [[nodiscard]] explicit Tensor() = default;
    [[nodiscard]] Tensor(const Tensor& other) = default;
    [[nodiscard]] Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    ~Tensor() = default;

    [[nodiscard]] Tensor(
        tt::tt_metal::Storage storage, TensorSpec tensor_spec, tt::tt_metal::TensorTopology tensor_topology);

    [[nodiscard]] Tensor(
        tt::tt_metal::HostBuffer buffer,
        const ttnn::Shape& shape,
        tt::tt_metal::DataType dtype,
        tt::tt_metal::Layout layout,
        const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
    [[nodiscard]] Tensor(
        tt::tt_metal::HostBuffer buffer,
        const ttnn::Shape& logical_shape,
        const ttnn::Shape& padded_shape,
        tt::tt_metal::DataType dtype,
        tt::tt_metal::Layout layout,
        const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
    [[nodiscard]] Tensor(tt::tt_metal::HostBuffer buffer, TensorSpec tensor_spec);

    template <typename T>
    [[nodiscard]] static Tensor from_span(
        tt::stl::Span<const T> buffer,
        const TensorSpec& spec,
        tt::tt_metal::distributed::MeshDevice* device = nullptr,
        std::optional<ttnn::QueueId> cq_id = std::nullopt,
        T pad_value = 0);

    template <typename T>
    [[nodiscard]] static Tensor from_borrowed_data(
        tt::stl::Span<T> buffer,
        const ttnn::Shape& shape,
        tt::tt_metal::MemoryPin buffer_pin,
        const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

    template <typename T>
    [[nodiscard]] static Tensor from_borrowed_data(
        tt::stl::Span<T> buffer,
        const ttnn::Shape& shape,
        const std::function<void()>& on_creation_callback,
        const std::function<void()>& on_destruction_callback,
        const std::optional<tt::tt_metal::Tile>& tile = std::nullopt) {
        return from_borrowed_data(buffer, shape, MemoryPin(on_creation_callback, on_destruction_callback), tile);
    }

    template <typename T>
    [[nodiscard]] static Tensor from_vector(
        const std::vector<T>& buffer,
        const TensorSpec& spec,
        tt::tt_metal::distributed::MeshDevice* device = nullptr,
        std::optional<ttnn::QueueId> cq_id = std::nullopt,
        T pad_value = 0) {
        return from_span(tt::stl::Span<const T>(buffer), spec, device, cq_id, pad_value);
    }

    template <typename T>
    [[nodiscard]] static Tensor from_vector(
        std::vector<T>&& buffer,
        const TensorSpec& spec,
        tt::tt_metal::distributed::MeshDevice* device = nullptr,
        std::optional<ttnn::QueueId> cq_id = std::nullopt,
        T pad_value = 0);

    template <typename T>
    [[nodiscard]] std::vector<T> to_vector(std::optional<ttnn::QueueId> cq_id = std::nullopt) const;

    template <typename T>
    [[nodiscard]] T item(std::optional<ttnn::QueueId> cq_id = std::nullopt) const;

    [[nodiscard]] Tensor to_device(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        ttsl::optional_reference<const tt::tt_metal::MemoryConfig> mem_config = std::nullopt,
        std::optional<ttnn::QueueId> cq_id = std::nullopt) const;

    [[nodiscard]] Tensor to_layout(tt::tt_metal::Layout target_layout) const;

    [[nodiscard]] Tensor pad(
        const ttnn::Shape& output_padded_shape, const ttnn::Shape& input_tensor_start, float pad_value) const;

    [[nodiscard]] Tensor cpu(bool blocking = true, std::optional<ttnn::QueueId> cq_id = std::nullopt) const;

    [[nodiscard]] Tensor unpad(const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) const;

    [[nodiscard]] Tensor pad_to_tile(float pad_value) const;

    [[nodiscard]] Tensor unpad_from_tile(const ttnn::Shape& output_tensor_shape) const;

    [[nodiscard]] std::string write_to_string() const;
    void print() const;

    void deallocate(bool force = false);

    [[nodiscard]] Tensor extract_shard(const tt::tt_metal::CoreCoord& core) const;
    [[nodiscard]] Tensor extract_shard(const uint32_t& core_id) const;

    [[nodiscard]] Tensor reshape(const ttnn::Shape& new_shape) const;
    [[nodiscard]] Tensor reshape(const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) const;

    Tensor with_tensor_topology(tt::tt_metal::TensorTopology tensor_topology) const;

    const tt::tt_metal::Storage& storage() const;
    tt::tt_metal::Storage& storage();
    tt::tt_metal::DataType dtype() const;
    tt::tt_metal::Layout layout() const;
    const ttnn::Shape& logical_shape() const;
    const ttnn::Shape& padded_shape() const;
    const TensorSpec& tensor_spec() const;
    uint64_t logical_volume() const;
    uint64_t physical_volume() const;
    const tt::tt_metal::MemoryConfig& memory_config() const;

    const tt::tt_metal::TensorTopology& tensor_topology() const;

    const std::optional<tt::tt_metal::ShardSpec>& shard_spec() const;
    const std::optional<tt::tt_metal::NdShardSpec>& nd_shard_spec() const;

    tt::tt_metal::StorageType storage_type() const;
    ttnn::Shape strides() const;

    bool is_scalar() const;
    bool is_allocated() const;

    tt::tt_metal::Buffer* buffer() const;

    const tt::tt_metal::DeviceStorage& device_storage() const&;
    const tt::tt_metal::DeviceStorage& device_storage() const&& = delete;

    const tt::tt_metal::HostStorage& host_storage() const&;
    const tt::tt_metal::HostStorage& host_storage() const&& = delete;

    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> mesh_buffer() const;

    tt::tt_metal::distributed::MeshDevice* device() const;

    bool is_sharded() const;

    uint32_t element_size() const;

    static constexpr auto attribute_names = std::forward_as_tuple("lazy_tensor_id");
    auto attribute_values() const { return std::forward_as_tuple(lazy_tensor_.id()); }

    // ttnn Tensor-only methods / constructors
    tt::tt_metal::metal_tensor::Tensor& get_materialized_tensor();
    const tt::tt_metal::metal_tensor::Tensor& get_materialized_tensor() const;
    [[nodiscard]] Tensor(const tt::tt_metal::metal_tensor::Tensor& metal_tensor);
    [[nodiscard]] Tensor(tt::tt_metal::metal_tensor::Tensor&& metal_tensor);
    [[nodiscard]] const ttnn::experimental::jit::LazyTensor& lazy() const;
    std::shared_ptr<TensorAttributes> tensor_attributes() const;

    static Tensor make_lazy_tensor(
        const std::vector<Tensor>& op_inputs,
        const std::shared_ptr<ttnn::experimental::jit::LazyOperation>& op,
        TensorSpec tensor_spec);
    static std::vector<Tensor> make_lazy_tensors(
        const std::vector<Tensor>& op_inputs,
        const std::shared_ptr<ttnn::experimental::jit::LazyOperation>& op,
        const std::vector<TensorSpec>& tensor_specs);

    void materialize();

private:
    Tensor(ttnn::experimental::jit::LazyTensor lazy_tensor);
    ttnn::experimental::jit::LazyTensor lazy_tensor_;
};

Tensor create_device_tensor(const TensorSpec& tensor_spec, IDevice* device);

[[deprecated]] Tensor create_device_tensor(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    IDevice* device,
    const MemoryConfig& memory_config = MemoryConfig{},
    const std::optional<Tile>& tile = std::nullopt);

// The set of memcpy functions below are used to copy data between host buffers/tensors and single-device tensors
void memcpy(
    distributed::MeshCommandQueue& queue,
    void* dst,
    const Tensor& src,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);

void memcpy(
    distributed::MeshCommandQueue& queue,
    Tensor& dst,
    const void* src,
    const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(
    distributed::MeshCommandQueue& queue,
    Tensor& dst,
    const Tensor& src,
    const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(
    void* dst, const Tensor& src, const std::optional<BufferRegion>& region = std::nullopt, bool blocking = true);

void memcpy(Tensor& dst, const void* src, const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(Tensor& dst, const Tensor& src, const std::optional<BufferRegion>& region = std::nullopt);

// Allocates a tensor on device.
Tensor allocate_tensor_on_device(const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device);

// Allocates a tensor on host. Uses `mesh_device` to allocate sufficient number of host buffers for each multi-device
// shard.
Tensor allocate_tensor_on_host(const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device);

// Writes tensor from `src` to `dst`; supports only host-to-device and device-to-host transfers.
void write_tensor(
    const Tensor& src, Tensor& dst, bool blocking = true, std::optional<ttnn::QueueId> cq_id = std::nullopt);

Tensor set_tensor_id(const Tensor& tensor);

}  // namespace tt_metal

}  // namespace tt

namespace ttnn {

using Tensor = tt::tt_metal::Tensor;
using TensorSpec = tt::tt_metal::TensorSpec;

}  // namespace ttnn
