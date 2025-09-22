
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_conversion.hpp>
#include <tracy/Tracy.hpp>

#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/operations/core/core.hpp>
#include <fmt/format.h>

using namespace tt::tt_metal;
namespace {

// Host buffer does not have an API to get the original number of elements,
// but in context of the conversion from python it is possible to use
// the type ID and the set of expected types.
DataType map_hostbuffer_type_to_datatype(const HostBuffer& buffer) {
    const auto& type_info = *buffer.type_info();

    if (type_info == typeid(bfloat16)) {
        return DataType::BFLOAT16;
    } else if (type_info == typeid(float)) {
        return DataType::FLOAT32;
    } else if (type_info == typeid(uint32_t)) {
        return DataType::UINT32;
    } else if (type_info == typeid(uint8_t)) {
        return DataType::UINT8;
    } else if (type_info == typeid(uint16_t)) {
        return DataType::UINT16;
    } else if (type_info == typeid(int32_t)) {
        return DataType::INT32;
    } else {
        TT_THROW("Unsupported type in HostBuffer: {}", buffer.type_info()->name());
    }
}

std::size_t get_element_count(const HostBuffer& buffer) {
    auto data_type = map_hostbuffer_type_to_datatype(buffer);
    auto byte_span = buffer.view_bytes();
    switch (data_type) {
        case DataType::BFLOAT16: return byte_span.size() / sizeof(bfloat16);
        case DataType::FLOAT32: return byte_span.size() / sizeof(float);
        case DataType::UINT32: return byte_span.size() / sizeof(uint32_t);
        case DataType::UINT8: return byte_span.size() / sizeof(uint8_t);
        case DataType::UINT16: return byte_span.size() / sizeof(uint16_t);
        case DataType::INT32: return byte_span.size() / sizeof(int32_t);
        default: TT_FATAL(false, "Unhandled DataType in get_element_count");
    }
}

struct TensorPreparedConversion {
    /// Use this layout to construct the initial tensor -- extra conversion might be done
    /// after the tensor has been moved to device.
    Layout construct_with_layout = Layout::TILE;
    DataType host_convert_data_type = DataType::INVALID;
};

template <typename T>
Tensor create_typed_tt_tensor_from_host_data(
    const HostBuffer& host_data,
    const TensorSpec& tensor_spec,
    ttnn::distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    TT_FATAL(
        !tensor_spec.tensor_layout().get_memory_config().is_sharded() ||
            tensor_spec.tensor_layout().get_memory_config().shard_spec().has_value() ||
            tensor_spec.tensor_layout().get_memory_config().nd_shard_spec().has_value(),
        "Sharded tensors must have a shard spec when converting to tt tensors!");

    TT_FATAL(
        *host_data.type_info() == typeid(T),
        "Mismatch between the host buffer data type and the target tensor data: host buffer is {} and the target is {}",
        host_data.type_info()->name(),
        typeid(T).name());

    tt::stl::Span<T> pydata_span(
        const_cast<T*>(reinterpret_cast<const T*>(host_data.view_bytes().data())),
        tensor_spec.logical_shape().volume());

    // Shard pydata across mesh and apply `tensor_layout` at each shard.
    // Shapes of multi device shards will be derived automatically.
    if (mesh_mapper != nullptr) {
        return ttnn::distributed::create_distributed_tensor(
            pydata_span,
            tensor_spec.logical_shape(),
            host_data.pin(),
            tensor_spec.tensor_layout(),
            *mesh_mapper,
            device != nullptr ? std::make_optional(std::ref(*device)) : std::nullopt,
            cq_id,
            static_cast<T>(pad_value));
    }

    // Otherwise, create a single tt tensor from the pydata.
    if (const bool pydata_borrowable = tensor_spec.layout() == Layout::ROW_MAJOR &&
                                       tensor_spec.physical_shape() == tensor_spec.logical_2d_shape() &&
                                       tensor_spec.data_type() == convert_to_data_type<T>();
        pydata_borrowable) {
        auto output =
            Tensor::from_borrowed_data(pydata_span, tensor_spec.logical_shape(), host_data.pin(), tensor_spec.tile());
        if (device != nullptr) {
            output = output.to_device(device, tensor_spec.memory_config(), cq_id);
        }
        return output;
    } else {
        return Tensor::from_span(
            tt::stl::make_const_span(pydata_span), tensor_spec, device, cq_id, static_cast<T>(pad_value));
    }
}

Tensor create_tt_tensor_from_host_data(
    const HostBuffer& host_data,
    const TensorSpec& tensor_spec,
    ttnn::distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    auto create_concrete = [&]<typename T>() {
        return create_typed_tt_tensor_from_host_data<T>(host_data, tensor_spec, device, cq_id, pad_value, mesh_mapper);
    };
    switch (tensor_spec.tensor_layout().get_data_type()) {
        case DataType::UINT8: return create_concrete.operator()<uint8_t>();
        case DataType::UINT16: return create_concrete.operator()<uint16_t>();
        case DataType::INT32: return create_concrete.operator()<int32_t>();
        case DataType::UINT32: return create_concrete.operator()<uint32_t>();
        case DataType::FLOAT32: return create_concrete.operator()<float>();
        case DataType::BFLOAT16: return create_concrete.operator()<bfloat16>();
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            return create_concrete.operator()<float>();
        }
        case DataType::INVALID: {
            TT_THROW("Unsupported DataType: {}", tensor_spec.tensor_layout().get_data_type());
        }
    }

    TT_THROW("Unsupported DataType: {}", tensor_spec.tensor_layout().get_data_type());
}

Tensor convert_host_buffer_to_tt_tensor_on_device(
    const HostBuffer& host_data,
    const TensorSpec& tensor_spec,
    ttnn::distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper,
    const TensorPreparedConversion& strategy) {
    ZoneScoped;

    auto output = create_tt_tensor_from_host_data(
        host_data,
        TensorSpec(
            tensor_spec.logical_shape(),
            TensorLayout(
                strategy.host_convert_data_type,
                PageConfig(strategy.construct_with_layout, tensor_spec.tile()),
                tensor_spec.memory_config())),
        device,
        cq_id,
        pad_value,
        mesh_mapper);

    output = tt::tt_metal::set_tensor_id(output);

    auto set_layout = [&](Layout target) {
        if (output.layout() != target) {
            output = ttnn::to_layout(output, target, std::nullopt, tensor_spec.memory_config());
        }
    };

    if (output.dtype() != tensor_spec.data_type()) {
        // Need to perform final data conversion on device, typecast requires TILE layout.
        set_layout(Layout::TILE);
        output = ttnn::typecast(output, tensor_spec.data_type());
    }

    set_layout(tensor_spec.layout());

    return output;
}

Tensor convert_host_buffer_to_tt_tensor_on_host(
    const HostBuffer& host_data,
    const TensorSpec& tensor_spec,
    ttnn::distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    ZoneScoped;
    if (tensor_spec.data_type() == DataType::BFLOAT8_B || tensor_spec.data_type() == DataType::BFLOAT4_B) {
        TT_FATAL(
            tensor_spec.layout() == Layout::TILE,
            "Tile layout is required for tensor of type bfloat8_b or bfloat4_b; got {}.",
            tensor_spec.layout());
    }

    auto output = create_tt_tensor_from_host_data(host_data, tensor_spec, device, cq_id, pad_value, mesh_mapper);

    return tt::tt_metal::set_tensor_id(output);
}

struct HostBufferConversionInput {
    host_buffer_data_type host_type;
    DataType target_type;
    Layout layout;

    bool operator==(const HostBufferConversionInput& other) const {
        return host_type == other.host_type && target_type == other.target_type && layout == other.layout;
    }
};

struct HostBufferConversionInputHash {
    std::size_t operator()(const HostBufferConversionInput& input) const {
        std::size_t h1 = std::hash<int>{}(static_cast<int>(input.host_type));
        std::size_t h2 = std::hash<int>{}(static_cast<int>(input.target_type));
        std::size_t h3 = std::hash<int>{}(static_cast<int>(input.layout));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

std::optional<TensorPreparedConversion> prepare_tensor_conversion(
    const host_buffer_data_type& host_data_type, const TensorSpec& tensor_spec, bool has_device) {
    // Early exit conditions -- on-device strategy is not supported

    if (!has_device ||
        // Device is required
        tensor_spec.memory_config().is_sharded() ||
        // Sharded tensor handling and on-device type-casting cannot be done with the regular strategy
        (((tensor_spec.tile().get_tile_shape()[0] % tt::constants::TILE_WIDTH) != 0) ||
         ((tensor_spec.tile().get_tile_shape()[1] % tt::constants::TILE_HEIGHT) != 0))
        // on-device tiling operation expects 32x32 row
    ) {
        return std::nullopt;
    }

    // High-level overview of the conversion strategy logic.
    //
    // Not all mappings improve performance if they are done on device: the type conversion itself is not the most
    // expensive part of the conversion, it is ROW->TILE conversion. If done on host, it might be ~10 times slower than
    // device. But due to existing issues with some on-device operators, only the mappings below can be safely done on
    // device, without the loss of precision.
    //
    // Edge cases that require host-side conversion due to known bugs:
    //    - int32 tensors with retiling can lose precision https://github.com/tenstorrent/tt-metal/issues/23407,
    //      although the size is not stable. `(32, 32, 64, 64)` Can trigger the bug as well.
    //    - uint8 typecast missing device support https://github.com/tenstorrent/tt-metal/issues/21682
    //    - float32 precision loss when changing layout https://github.com/tenstorrent/tt-metal/issues/23405
    //    - bfloat16 to bfloat4b/bfloat8b conversions can zero half the tensor in some conditions.
    //      The test triggering this bug is test_matmul.py::test_tiny_tiles_bfloat
    //
    // Based on the benchmark data, not all conversion pairings have performance improvements
    // when done on host. Additionally, some types cannot be stored in ROW-MAJOR form, like bfloat8, meaning that
    // on-host conversion to TILE is mandatory for the TTNN tensor creation.
    //
    // To extend the conversion map once the aforementioned bugs are resolved:
    //
    // - `construct_with_layout` constrols which layout should be used for the host-side tensor construction. For
    //   performance reasons the ROW-MAJOR is the most optimal one.
    // - `host_side_conversion` to show whether on-device type casting is necessary or not.
    //   If not, the tensor will be created using torch (or on-host converted torch data) and optionally changed to the
    //   right layout.

    // Mapping
    // `{input_torch_type, expected_ttnn_type, expected_layout}` -> `{on-host_tensor_layout, on-host_tensor_data_type,
    // torch_data_conversion}`

    static std::unordered_map<HostBufferConversionInput, TensorPreparedConversion, HostBufferConversionInputHash>
        conversion_map = {
    // clang-format off

            // At the moment there are no cases that can be safely implemented with on-device
            // conversion, and bfloat16 cases are to be implemented in a follow-up PR to avoid
            // breaking too many tests in a scope of a single PR. The conversion mappings below
            // can be enabled and updated as related bugs with type/layout conversion are fixed
            // in the other parts of the library

            // The mapping structure is
            // {<Input-Type>, <Target-Type>, <Target-Layout>} -> {<Layout-To-Construct-On-Host>, <Type-To-Cast-On-Host>}

#if false
            {{host_buffer_data_type::BFLOAT16,     DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::BFLOAT16, Layout::ROW_MAJOR},  {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::BFLOAT16,     DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::UINT32 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT16,      DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::UINT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT32,      DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::FLOAT64,      DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::FLOAT32 }},
            {{host_buffer_data_type::INT16,        DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT16,        DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT32,        DataType::UINT8,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::INT64,        DataType::BFLOAT16,  Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::BFLOAT4_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::BFLOAT8_B, Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::FLOAT32,   Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::INT32,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::UINT16,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::INT64,        DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::UINT8 }},
            {{host_buffer_data_type::INT64,        DataType::UINT8,     Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::BFLOAT16,  Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::BFLOAT16 }},
            {{host_buffer_data_type::UINT8,        DataType::BFLOAT16,  Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::BFLOAT4_B, Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::BFLOAT8_B, Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::FLOAT32,   Layout::ROW_MAJOR}, {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::FLOAT32,   Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::INT32,     Layout::ROW_MAJOR}, {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::INT32,     Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT16,    Layout::ROW_MAJOR}, {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT16,    Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT32,    Layout::ROW_MAJOR}, {Layout::TILE,      DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT32,    Layout::TILE},      {Layout::ROW_MAJOR, DataType::INT32 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT8,     Layout::ROW_MAJOR}, {Layout::ROW_MAJOR, DataType::UINT8 }},
            {{host_buffer_data_type::UINT8,        DataType::UINT8,     Layout::TILE},      {Layout::TILE,      DataType::UINT8 }},
#endif

            // clang-format on
        };

    HostBufferConversionInput input{
        .host_type = host_data_type,
        .target_type = tensor_spec.data_type(),
        .layout = tensor_spec.layout(),
    };

    auto it = conversion_map.find(input);
    if (it == conversion_map.end()) {
        return std::nullopt;
    } else {
        return it->second;
    }
}
}  // namespace

Tensor tt::tt_metal::create_device_tensor_from_host_data(
    const TensorSpec& tensor_spec,
    const host_buffer_data_type& host_data_type,
    std::function<HostBuffer(DataType)> get_host_data,
    ttnn::distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    auto strategy = prepare_tensor_conversion(host_data_type, tensor_spec, device != nullptr);
    Tensor output;

    DataType on_device_conversion_target;
    if (strategy) {
        on_device_conversion_target = strategy->host_convert_data_type;
    } else {
        if (tensor_spec.data_type() == DataType::BFLOAT4_B || tensor_spec.data_type() == DataType::BFLOAT8_B) {
            on_device_conversion_target = DataType::FLOAT32;
        } else {
            on_device_conversion_target = tensor_spec.data_type();
        }
    }

    HostBuffer host_data = get_host_data(on_device_conversion_target);

    TT_FATAL(
        get_element_count(host_data) == tensor_spec.logical_shape().volume(),
        "Number of elements from python tensor {} must match volume of shape {}!",
        get_element_count(host_data),
        tensor_spec.logical_shape().volume());

    if (strategy) {
        if (host_data.view_bytes().empty()) {
            // to tile the tensor it must have non-zero volume or a sufficient rank -- if this fails
            // the tensor must be constructed on host.
            output =
                convert_host_buffer_to_tt_tensor_on_host(host_data, tensor_spec, device, cq_id, pad_value, mesh_mapper);
        } else {
            output = convert_host_buffer_to_tt_tensor_on_device(
                host_data, tensor_spec, device, cq_id, pad_value, mesh_mapper, strategy.value());
        }
    } else {
        output =
            convert_host_buffer_to_tt_tensor_on_host(host_data, tensor_spec, device, cq_id, pad_value, mesh_mapper);
    }
    return output;
}
