// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/experimental/jit/lazy_tensor.hpp"

namespace ttnn {

using TensorSpec = tt::tt_metal::TensorSpec;

class TTNNTensor {
public:
    std::optional<std::int64_t> tensor_id = std::nullopt;

    [[nodiscard]] explicit TTNNTensor() = default;
    [[nodiscard]] TTNNTensor(const TTNNTensor& other) = default;
    [[nodiscard]] TTNNTensor(TTNNTensor&& other) noexcept = default;
    TTNNTensor& operator=(const TTNNTensor& other) = default;
    TTNNTensor& operator=(TTNNTensor&& other) noexcept = default;
    ~TTNNTensor() = default;

    [[nodiscard]] TTNNTensor(
        tt::tt_metal::Storage storage, TensorSpec tensor_spec, tt::tt_metal::TensorTopology tensor_topology);

    [[nodiscard]] TTNNTensor(
        tt::tt_metal::HostBuffer buffer,
        const ttnn::Shape& shape,
        tt::tt_metal::DataType dtype,
        tt::tt_metal::Layout layout,
        const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
    [[nodiscard]] TTNNTensor(
        tt::tt_metal::HostBuffer buffer,
        const ttnn::Shape& logical_shape,
        const ttnn::Shape& padded_shape,
        tt::tt_metal::DataType dtype,
        tt::tt_metal::Layout layout,
        const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
    [[nodiscard]] TTNNTensor(tt::tt_metal::HostBuffer buffer, TensorSpec tensor_spec);

    template <typename T>
    [[nodiscard]] static TTNNTensor from_span(
        tt::stl::Span<const T> buffer,
        const TensorSpec& spec,
        tt::tt_metal::distributed::MeshDevice* device = nullptr,
        std::optional<ttnn::QueueId> cq_id = std::nullopt,
        T pad_value = 0);

    template <typename T>
    [[nodiscard]] static TTNNTensor from_borrowed_data(
        tt::stl::Span<T> buffer,
        const ttnn::Shape& shape,
        tt::tt_metal::MemoryPin buffer_pin,
        const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

    template <typename T>
    [[nodiscard]] static TTNNTensor from_borrowed_data(
        tt::stl::Span<T> buffer,
        const ttnn::Shape& shape,
        const std::function<void()>& on_creation_callback,
        const std::function<void()>& on_destruction_callback,
        const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

    template <typename T>
    [[nodiscard]] static TTNNTensor from_vector(
        const std::vector<T>& buffer,
        const TensorSpec& spec,
        tt::tt_metal::distributed::MeshDevice* device = nullptr,
        std::optional<ttnn::QueueId> cq_id = std::nullopt,
        T pad_value = 0);

    template <typename T>
    [[nodiscard]] static TTNNTensor from_vector(
        std::vector<T>&& buffer,
        const TensorSpec& spec,
        tt::tt_metal::distributed::MeshDevice* device = nullptr,
        std::optional<ttnn::QueueId> cq_id = std::nullopt,
        T pad_value = 0);

    template <typename T>
    [[nodiscard]] std::vector<T> to_vector(std::optional<ttnn::QueueId> cq_id = std::nullopt) const;

    template <typename T>
    [[nodiscard]] T item(std::optional<ttnn::QueueId> cq_id = std::nullopt) const;

    [[nodiscard]] TTNNTensor to_device(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        ttsl::optional_reference<const tt::tt_metal::MemoryConfig> mem_config = std::nullopt,
        std::optional<ttnn::QueueId> cq_id = std::nullopt) const;

    [[nodiscard]] TTNNTensor to_layout(tt::tt_metal::Layout target_layout) const;

    [[nodiscard]] TTNNTensor pad(
        const ttnn::Shape& output_padded_shape, const ttnn::Shape& input_tensor_start, float pad_value) const;

    [[nodiscard]] TTNNTensor cpu(bool blocking = true, std::optional<ttnn::QueueId> cq_id = std::nullopt) const;

    [[nodiscard]] TTNNTensor unpad(const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) const;

    [[nodiscard]] TTNNTensor pad_to_tile(float pad_value) const;

    [[nodiscard]] TTNNTensor unpad_from_tile(const ttnn::Shape& output_tensor_shape) const;

    [[nodiscard]] std::string write_to_string() const;
    void print() const;

    void deallocate(bool force = false);

    [[nodiscard]] TTNNTensor extract_shard(const tt::tt_metal::CoreCoord& core) const;
    [[nodiscard]] TTNNTensor extract_shard(const uint32_t& core_id) const;

    [[nodiscard]] TTNNTensor reshape(const ttnn::Shape& new_shape) const;
    [[nodiscard]] TTNNTensor reshape(const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) const;

    TTNNTensor with_tensor_topology(tt::tt_metal::TensorTopology tensor_topology) const;

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
    tt::tt_metal::Tensor& get_materialized_tensor();
    const tt::tt_metal::Tensor& get_materialized_tensor() const;
    [[nodiscard]] TTNNTensor(const tt::tt_metal::Tensor& metal_tensor);
    [[nodiscard]] TTNNTensor(tt::tt_metal::Tensor&& metal_tensor);
    [[nodiscard]] const experimental::jit::LazyTensor& lazy() const;
    // This is temporary operation, I don't like an idea of implicit casting to metarialized tensor
    // TODO: remove this once we have a proper lazy tensor support
    operator tt::tt_metal::Tensor&() { return get_materialized_tensor(); }
    operator const tt::tt_metal::Tensor&() const { return get_materialized_tensor(); }

private:
    TTNNTensor(experimental::jit::LazyTensor lazy_tensor);
    experimental::jit::LazyTensor lazy_tensor_;
};

TTNNTensor set_tensor_id(const TTNNTensor& tensor);

}  // namespace ttnn
