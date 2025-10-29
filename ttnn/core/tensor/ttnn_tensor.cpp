#include "ttnn/tensor/ttnn_tensor.hpp"
#include <tt_stl/assert.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include "ttnn/core.hpp"

namespace ttnn {

TTNNTensor::TTNNTensor(experimental::jit::LazyTensor lazy_tensor) : lazy_tensor_(std::move(lazy_tensor)) {}

TTNNTensor::TTNNTensor(const tt::tt_metal::Tensor& metal_tensor) :
    lazy_tensor_(experimental::jit::LazyTensor::make_materialized_tensor(metal_tensor)) {}

TTNNTensor::TTNNTensor(tt::tt_metal::Tensor&& metal_tensor) :
    lazy_tensor_(experimental::jit::LazyTensor::make_materialized_tensor(metal_tensor)) {}

TTNNTensor::TTNNTensor(
    tt::tt_metal::Storage storage, TensorSpec tensor_spec, tt::tt_metal::TensorTopology tensor_topology) :
    lazy_tensor_(experimental::jit::LazyTensor::make_materialized_tensor(
        tt::tt_metal::Tensor(std::move(storage), std::move(tensor_spec), std::move(tensor_topology)))) {}

TTNNTensor::TTNNTensor(
    tt::tt_metal::HostBuffer buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::DataType dtype,
    tt::tt_metal::Layout layout,
    const std::optional<tt::tt_metal::Tile>& tile) :
    lazy_tensor_(experimental::jit::LazyTensor::make_materialized_tensor(
        tt::tt_metal::Tensor(std::move(buffer), shape, dtype, layout, tile))) {}

TTNNTensor::TTNNTensor(
    tt::tt_metal::HostBuffer buffer,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    tt::tt_metal::DataType dtype,
    tt::tt_metal::Layout layout,
    const std::optional<tt::tt_metal::Tile>& tile) :
    lazy_tensor_(experimental::jit::LazyTensor::make_materialized_tensor(
        tt::tt_metal::Tensor(std::move(buffer), logical_shape, padded_shape, dtype, layout, tile))) {}

TTNNTensor::TTNNTensor(tt::tt_metal::HostBuffer buffer, TensorSpec tensor_spec) :
    lazy_tensor_(experimental::jit::LazyTensor::make_materialized_tensor(
        tt::tt_metal::Tensor(std::move(buffer), std::move(tensor_spec)))) {}

template <typename T>
TTNNTensor TTNNTensor::from_span(
    tt::stl::Span<const T> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    T pad_value) {
    return TTNNTensor(tt::tt_metal::Tensor::from_span(buffer, spec, device, cq_id, pad_value));
}

template <typename T>
TTNNTensor TTNNTensor::from_borrowed_data(
    tt::stl::Span<T> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile) {
    return TTNNTensor(tt::tt_metal::Tensor::from_borrowed_data(buffer, shape, std::move(buffer_pin), tile));
}

template <typename T>
TTNNTensor TTNNTensor::from_borrowed_data(
    tt::stl::Span<T> buffer,
    const ttnn::Shape& shape,
    const std::function<void()>& on_creation_callback,
    const std::function<void()>& on_destruction_callback,
    const std::optional<tt::tt_metal::Tile>& tile) {
    return TTNNTensor(
        tt::tt_metal::Tensor::from_borrowed_data(buffer, shape, on_creation_callback, on_destruction_callback, tile));
}

template <typename T>
TTNNTensor TTNNTensor::from_vector(
    const std::vector<T>& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    T pad_value) {
    return TTNNTensor(tt::tt_metal::Tensor::from_vector(buffer, spec, device, cq_id, pad_value));
}

template <typename T>
TTNNTensor TTNNTensor::from_vector(
    std::vector<T>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    T pad_value) {
    return TTNNTensor(tt::tt_metal::Tensor::from_vector(std::move(buffer), spec, device, cq_id, pad_value));
}

template <typename T>
std::vector<T> TTNNTensor::to_vector(std::optional<ttnn::QueueId> cq_id) const {
    return get_materialized_tensor().to_vector<T>(cq_id);
}

template <typename T>
T TTNNTensor::item(std::optional<ttnn::QueueId> cq_id) const {
    return get_materialized_tensor().item<T>(cq_id);
}

TTNNTensor TTNNTensor::to_device(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const tt::tt_metal::MemoryConfig> mem_config,
    std::optional<ttnn::QueueId> cq_id) const {
    return TTNNTensor(get_materialized_tensor().to_device(mesh_device, mem_config, cq_id));
}

TTNNTensor TTNNTensor::to_layout(tt::tt_metal::Layout target_layout) const {
    return TTNNTensor(get_materialized_tensor().to_layout(target_layout));
}

TTNNTensor TTNNTensor::pad(
    const ttnn::Shape& output_padded_shape, const ttnn::Shape& input_tensor_start, float pad_value) const {
    return TTNNTensor(get_materialized_tensor().pad(output_padded_shape, input_tensor_start, pad_value));
}

TTNNTensor TTNNTensor::cpu(bool blocking, std::optional<ttnn::QueueId> cq_id) const {
    return TTNNTensor(get_materialized_tensor().cpu(blocking, cq_id));
}

TTNNTensor TTNNTensor::unpad(const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) const {
    return TTNNTensor(get_materialized_tensor().unpad(output_tensor_start, output_tensor_end));
}

TTNNTensor TTNNTensor::pad_to_tile(float pad_value) const {
    return TTNNTensor(get_materialized_tensor().pad_to_tile(pad_value));
}

TTNNTensor TTNNTensor::unpad_from_tile(const ttnn::Shape& output_tensor_shape) const {
    return TTNNTensor(get_materialized_tensor().unpad_from_tile(output_tensor_shape));
}

std::string TTNNTensor::write_to_string() const { return get_materialized_tensor().write_to_string(); }

void TTNNTensor::print() const { get_materialized_tensor().print(); }

void TTNNTensor::deallocate(bool force) { get_materialized_tensor().deallocate(force); }

TTNNTensor TTNNTensor::extract_shard(const tt::tt_metal::CoreCoord& core) const {
    return TTNNTensor(get_materialized_tensor().extract_shard(core));
}

TTNNTensor TTNNTensor::extract_shard(const uint32_t& core_id) const {
    return TTNNTensor(get_materialized_tensor().extract_shard(core_id));
}

TTNNTensor TTNNTensor::reshape(const ttnn::Shape& new_shape) const {
    return TTNNTensor(get_materialized_tensor().reshape(new_shape));
}

TTNNTensor TTNNTensor::reshape(const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape) const {
    return TTNNTensor(get_materialized_tensor().reshape(new_logical_shape, new_padded_shape));
}

TTNNTensor TTNNTensor::with_tensor_topology(tt::tt_metal::TensorTopology tensor_topology) const {
    return TTNNTensor(get_materialized_tensor().with_tensor_topology(std::move(tensor_topology)));
}

const tt::tt_metal::Storage& TTNNTensor::storage() const { return get_materialized_tensor().storage(); }

tt::tt_metal::Storage& TTNNTensor::storage() { return get_materialized_tensor().storage(); }

tt::tt_metal::DataType TTNNTensor::dtype() const { return lazy_tensor_.tensor_spec().tensor_layout().get_data_type(); }

tt::tt_metal::Layout TTNNTensor::layout() const { return lazy_tensor_.tensor_spec().tensor_layout().get_layout(); }

const ttnn::Shape& TTNNTensor::logical_shape() const { return lazy_tensor_.tensor_spec().logical_shape(); }

const ttnn::Shape& TTNNTensor::padded_shape() const { return lazy_tensor_.tensor_spec().padded_shape(); }

const TensorSpec& TTNNTensor::tensor_spec() const { return lazy_tensor_.tensor_spec(); }

uint64_t TTNNTensor::logical_volume() const { return lazy_tensor_.tensor_spec().logical_shape().volume(); }

uint64_t TTNNTensor::physical_volume() const { return lazy_tensor_.tensor_spec().padded_shape().volume(); }

const tt::tt_metal::MemoryConfig& TTNNTensor::memory_config() const {
    return lazy_tensor_.tensor_spec().tensor_layout().get_memory_config();
}

const tt::tt_metal::TensorTopology& TTNNTensor::tensor_topology() const {
    // TODO: Should be available without materialization
    return get_materialized_tensor().tensor_topology();
}

const std::optional<tt::tt_metal::ShardSpec>& TTNNTensor::shard_spec() const {
    return lazy_tensor_.tensor_spec().tensor_layout().get_memory_config().shard_spec();
}

const std::optional<tt::tt_metal::NdShardSpec>& TTNNTensor::nd_shard_spec() const {
    return lazy_tensor_.tensor_spec().tensor_layout().get_memory_config().nd_shard_spec();
}

tt::tt_metal::StorageType TTNNTensor::storage_type() const {
    // TODO: check if this information is required for non-materialized tensors
    return get_materialized_tensor().storage_type();
}

ttnn::Shape TTNNTensor::strides() const {
    // TODO: remove duplication with tensor.cpp
    auto s = tt::tt_metal::compute_strides(this->padded_shape());
    return ttnn::Shape(tt::stl::SmallVector<uint32_t>(s.begin(), s.end()));
}

bool TTNNTensor::is_scalar() const {
    const ttnn::Shape logical_shape = this->logical_shape();
    return logical_shape.rank() == 0 || logical_shape.volume() == 1;
}

bool TTNNTensor::is_allocated() const {
    return lazy_tensor_.is_materialized() ? get_materialized_tensor().is_allocated() : false;
}

tt::tt_metal::Buffer* TTNNTensor::buffer() const { return get_materialized_tensor().buffer(); }

const tt::tt_metal::DeviceStorage& TTNNTensor::device_storage() const& {
    return get_materialized_tensor().device_storage();
}

const tt::tt_metal::HostStorage& TTNNTensor::host_storage() const& { return get_materialized_tensor().host_storage(); }

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> TTNNTensor::mesh_buffer() const {
    return get_materialized_tensor().mesh_buffer();
}

tt::tt_metal::distributed::MeshDevice* TTNNTensor::device() const { return get_materialized_tensor().device(); }

bool TTNNTensor::is_sharded() const { return get_materialized_tensor().is_sharded(); }

uint32_t TTNNTensor::element_size() const { return get_materialized_tensor().element_size(); }

// ttnn Tensor-only methods / constructors
tt::tt_metal::Tensor& TTNNTensor::get_materialized_tensor() {
    TT_FATAL(lazy_tensor_.is_materialized(), "TTNNTensor is not materialized");
    return lazy_tensor_.materialized_tensor();
}

const tt::tt_metal::Tensor& TTNNTensor::get_materialized_tensor() const {
    TT_FATAL(lazy_tensor_.is_materialized(), "TTNNTensor is not materialized");
    return lazy_tensor_.materialized_tensor();
}

const experimental::jit::LazyTensor& TTNNTensor::lazy() const { return lazy_tensor_; }

template TTNNTensor TTNNTensor::from_span<bfloat16>(
    tt::stl::Span<const bfloat16> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    bfloat16 pad_value);
template TTNNTensor TTNNTensor::from_span<float>(
    tt::stl::Span<const float> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value);
template TTNNTensor TTNNTensor::from_span<int32_t>(
    tt::stl::Span<const int32_t> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    int32_t pad_value);
template TTNNTensor TTNNTensor::from_span<uint8_t>(
    tt::stl::Span<const uint8_t> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint8_t pad_value);
template TTNNTensor TTNNTensor::from_span<uint16_t>(
    tt::stl::Span<const uint16_t> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint16_t pad_value);
template TTNNTensor TTNNTensor::from_span<uint32_t>(
    tt::stl::Span<const uint32_t> buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint32_t pad_value);

template TTNNTensor TTNNTensor::from_borrowed_data<float>(
    tt::stl::Span<float> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);
template TTNNTensor TTNNTensor::from_borrowed_data<bfloat16>(
    tt::stl::Span<bfloat16> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);
template TTNNTensor TTNNTensor::from_borrowed_data<int32_t>(
    tt::stl::Span<int32_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);
template TTNNTensor TTNNTensor::from_borrowed_data<uint8_t>(
    tt::stl::Span<uint8_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);
template TTNNTensor TTNNTensor::from_borrowed_data<uint16_t>(
    tt::stl::Span<uint16_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);
template TTNNTensor TTNNTensor::from_borrowed_data<uint32_t>(
    tt::stl::Span<uint32_t> buffer,
    const ttnn::Shape& shape,
    tt::tt_metal::MemoryPin buffer_pin,
    const std::optional<tt::tt_metal::Tile>& tile);

template TTNNTensor TTNNTensor::from_vector<bfloat16>(
    std::vector<bfloat16>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    bfloat16 pad_value);
template TTNNTensor TTNNTensor::from_vector<float>(
    std::vector<float>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value);
template TTNNTensor TTNNTensor::from_vector<int32_t>(
    std::vector<int32_t>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    int32_t pad_value);
template TTNNTensor TTNNTensor::from_vector<uint8_t>(
    std::vector<uint8_t>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint8_t pad_value);
template TTNNTensor TTNNTensor::from_vector<uint16_t>(
    std::vector<uint16_t>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint16_t pad_value);
template TTNNTensor TTNNTensor::from_vector<uint32_t>(
    std::vector<uint32_t>&& buffer,
    const TensorSpec& spec,
    tt::tt_metal::distributed::MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    uint32_t pad_value);

template std::vector<bfloat16> TTNNTensor::to_vector<bfloat16>(std::optional<ttnn::QueueId> cq_id) const;
template std::vector<float> TTNNTensor::to_vector<float>(std::optional<ttnn::QueueId> cq_id) const;
template std::vector<int32_t> TTNNTensor::to_vector<int32_t>(std::optional<ttnn::QueueId> cq_id) const;
template std::vector<uint8_t> TTNNTensor::to_vector<uint8_t>(std::optional<ttnn::QueueId> cq_id) const;
template std::vector<uint16_t> TTNNTensor::to_vector<uint16_t>(std::optional<ttnn::QueueId> cq_id) const;
template std::vector<uint32_t> TTNNTensor::to_vector<uint32_t>(std::optional<ttnn::QueueId> cq_id) const;

template float TTNNTensor::item<float>(std::optional<ttnn::QueueId> cq_id) const;
template bfloat16 TTNNTensor::item<bfloat16>(std::optional<ttnn::QueueId> cq_id) const;
template int32_t TTNNTensor::item<int32_t>(std::optional<ttnn::QueueId> cq_id) const;
template uint8_t TTNNTensor::item<uint8_t>(std::optional<ttnn::QueueId> cq_id) const;
template uint16_t TTNNTensor::item<uint16_t>(std::optional<ttnn::QueueId> cq_id) const;
template uint32_t TTNNTensor::item<uint32_t>(std::optional<ttnn::QueueId> cq_id) const;

}  // namespace ttnn
