
#include <ttnn/tensor/tensor.hpp>
#include <tracy/Tracy.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

using namespace tt::tt_metal;

namespace {
template <typename T>
Tensor create_typed_tt_tensor_from_py_data(
    std::size_t py_data_ptr,
    const Shape& py_data_shape,
    const TensorLayout& tensor_layout,
    ttnn::distributed::MeshDevice* device,
    const tt::tt_metal::MemoryPin& pydata_pin,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    TT_FATAL(
        !tensor_layout.get_memory_config().is_sharded() || tensor_layout.get_memory_config().shard_spec().has_value() ||
            tensor_layout.get_memory_config().nd_shard_spec().has_value(),
        "Sharded tensors must have a shard spec when converting to tt tensors!");

    tt::stl::Span<T> pydata_span(reinterpret_cast<T*>(py_data_ptr), py_data_shape.volume());

    // Shard pydata across mesh and apply `tensor_layout` at each shard.
    // Shapes of multi device shards will be derived automatically.
    if (mesh_mapper != nullptr) {
        return ttnn::distributed::create_distributed_tensor(
            pydata_span,
            py_data_shape,
            pydata_pin,
            tensor_layout,
            *mesh_mapper,
            device != nullptr ? std::make_optional(std::ref(*device)) : std::nullopt,
            cq_id,
            static_cast<T>(pad_value));
    }

    // Otherwise, create a single tt tensor from the pydata.
    const TensorSpec tensor_spec(py_data_shape, tensor_layout);
    if (const bool pydata_borrowable = tensor_spec.layout() == Layout::ROW_MAJOR &&
                                       tensor_spec.physical_shape() == tensor_spec.logical_2d_shape() &&
                                       tensor_spec.data_type() == convert_to_data_type<T>();
        pydata_borrowable) {
        auto output =
            Tensor::from_borrowed_data(pydata_span, tensor_spec.logical_shape(), pydata_pin, tensor_spec.tile());
        if (device != nullptr) {
            output = output.to_device(device, tensor_spec.memory_config(), cq_id);
        }
        return output;
    } else {
        return Tensor::from_span(
            tt::stl::make_const_span(pydata_span), tensor_spec, device, cq_id, static_cast<T>(pad_value));
    }
}

Tensor create_tt_tensor_from_py_data(
    std::size_t py_data_ptr,
    const Shape& py_data_shape,
    const TensorLayout& tensor_layout,
    ttnn::distributed::MeshDevice* device,
    const tt::tt_metal::MemoryPin& pydata_pin,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    auto create_concrete = [&]<typename T>() {
        return create_typed_tt_tensor_from_py_data<T>(
            py_data_ptr, py_data_shape, tensor_layout, device, pydata_pin, cq_id, pad_value, mesh_mapper);
    };
    switch (tensor_layout.get_data_type()) {
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
            TT_THROW("Unsupported DataType: {}", tensor_layout.get_data_type());
        }
    }

    TT_THROW("Unsupported DataType: {}", tensor_layout.get_data_type());
}

Tensor convert_python_tensor_to_tt_tensor_on_device(
    const py::handle& py_tensor,
    std::optional<DataType> optional_data_type,
    std::optional<Layout> optional_layout,
    const std::optional<Tile>& optional_tile,
    const MemoryConfig& memory_config,
    ttnn::distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper,
    const PyTensorPreparedConversion& strategy) {
    ZoneScoped;
    py::object contiguous_py_tensor = py_tensor.attr("contiguous")();
    py::object torch = py::module_::import("torch");

    if (strategy.torch_convert_dtype) {
        ZoneScopedN("Convert type on host");
        contiguous_py_tensor =
            contiguous_py_tensor.attr("to")(torch.attr(strategy.torch_convert_dtype.value().c_str()));
    }

    auto py_data_ptr = py::cast<std::size_t>(contiguous_py_tensor.attr("data_ptr")());
    auto num_elements = py::cast<std::size_t>(contiguous_py_tensor.attr("numel")());

    TT_FATAL(
        num_elements == shape.volume(),
        "Number of elements from python tensor {} must match volume of shape {}!",
        num_elements,
        shape.volume());

    tt::tt_metal::MemoryPin pydata_pin(std::make_shared<py::object>(contiguous_py_tensor));

    auto output = create_tt_tensor_from_py_data(
        py_data_ptr,
        shape,
        TensorLayout(
            strategy.construct_with_data_type,
            PageConfig(strategy.construct_with_layout, optional_tile),
            memory_config),
        device,
        pydata_pin,
        cq_id,
        pad_value,
        mesh_mapper);

    output = tt::tt_metal::set_tensor_id(output);

    auto set_layout = [&](Layout target) {
        if (output.layout() != target) {
            output = ttnn::to_layout(output, target, std::nullopt, memory_config);
        }
    };

    if (optional_data_type.has_value() && output.dtype() != optional_data_type.value()) {
        // Need to perform final data conversion on device, typecast requires TILE layout.
        set_layout(Layout::TILE);
        output = ttnn::typecast(output, optional_data_type.value());
    }

    if (optional_layout.has_value()) {
        set_layout(optional_layout.value());
    }

    return output;
}

Tensor convert_python_tensor_to_tt_tensor_on_host(
    const py::handle& py_tensor,
    std::optional<DataType> optional_data_type,
    std::optional<Layout> optional_layout,
    const std::optional<Tile>& optional_tile,
    const MemoryConfig& memory_config,
    ttnn::distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    ZoneScoped;
    auto preprocessed_py_tensor = parse_py_tensor(py_tensor, optional_data_type);
    const auto shape = ttnn::Shape(py::cast<ttnn::SmallVector<uint32_t>>(py_tensor.attr("shape")));

    TT_FATAL(
        preprocessed_py_tensor.num_elements == shape.volume(),
        "Number of elements from python tensor {} must match volume of shape {}!",
        preprocessed_py_tensor.num_elements,
        shape.volume());

    const Layout layout = [&]() {
        // Block float types require tile layout.
        // Choose tile by default and disallow overriding to anything else.
        if (preprocessed_py_tensor.data_type == DataType::BFLOAT8_B ||
            preprocessed_py_tensor.data_type == DataType::BFLOAT4_B) {
            TT_FATAL(
                !optional_layout.has_value() or *optional_layout == Layout::TILE,
                "Tile layout is required for tensor of type bfloat8_b or bfloat4_b; got {}.",
                *optional_layout);
            return Layout::TILE;
        } else {
            return optional_layout.value_or(Layout::ROW_MAJOR);
        }
    }();

    // Important: `py::object` copying and destruction must be done while holding GIL, which pybind ensures for a thread
    // that calls the C++ APIs. We wrap `py::object` in `MemoryPin` so that multi-threaded C++ code only increments /
    // decrements the reference count on the memory pin; the last decrement to the pin should be triggered from the
    // pybind caller thread, which will correctly decrement the `py::object` reference count while hodling GIL.
    tt::tt_metal::MemoryPin pydata_pin(std::make_shared<py::object>(preprocessed_py_tensor.contiguous_py_tensor));

    auto output = create_tt_tensor_from_py_data(
        preprocessed_py_tensor.py_data_ptr,
        shape,
        TensorLayout(preprocessed_py_tensor.data_type, PageConfig(layout, optional_tile), memory_config),
        device,
        pydata_pin,
        cq_id,
        pad_value,
        mesh_mapper);

    return tt::tt_metal::set_tensor_id(output);
}

struct HostBufferConversionInput {
    std::type_info host_type;
    DataType data_type;
    Layout layout;

    bool operator==(const HostBufferConversionInput& other) const {
        return host_type == other.host_type && data_type == other.data_type && layout == other.layout;
    }
};

struct HostBufferConversionInputHash {
    std::size_t operator()(const HostBufferConversionInput& input) const {
        std::size_t h1 = std::hash<std::string>{}(input.host_type);
        std::size_t h2 = std::hash<int>{}(static_cast<int>(input.data_type));
        std::size_t h3 = std::hash<int>{}(static_cast<int>(input.layout));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct TensorPreparedConversion {
    /// Use this layout to construct the initial tensor -- extra conversion might be done
    /// after the tensor has been moved to device.
    Layout construct_with_layout = Layout::TILE;
    DataType construct_with_data_type = DataType::INVALID;
    std::optional<std::string> torch_convert_dtype = std::nullopt;
};

std::optional<TensorPreparedConversion> prepare_tensor_conversion(
    const HostBuffer& host_data,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    bool has_device,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& optional_tile) {
    // Early exit conditions -- on-device strategy is not supported

    if (!has_device ||
        // Device is required
        host_data.view_as().empty() ||
        // to tile the tensor it must have non-zero volume or a sufficient rank -- if this fails
        // the tensor must be constructed on host.
        memory_config.is_sharded() ||
        // Sharded tensor handling and on-device type-casting cannot be done with the regular strategy
        (optional_tile.has_value() && (((optional_tile->get_tile_shape()[0] % tt::constants::TILE_WIDTH) != 0) ||
                                       ((optional_tile->get_tile_shape()[1] % tt::constants::TILE_HEIGHT) != 0))) ||
        // on-device tiling operation expects 32x32 row
        !map_torch_data_type_to_ttnn(py_tensor.attr("dtype"), torch).has_value()) {
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
            // breaking too many tests in a scope of a single PR.

            // clang-format on
        };

    DataType expected_dtype = dtype.value_or(map_torch_data_type_to_ttnn(py_tensor.attr("dtype"), torch).value());
    HostBufferConversionInput input{
        .torch_dtype = host_data.type_info(),
        .data_type = expected_dtype,
        .layout = layout.value_or(Layout::ROW_MAJOR),
    };

    auto it = conversion_map.find(input);
    if (it == conversion_map.end()) {
        return std::nullopt;
    } else {
        return it->second;
    }
}
}  // namespace

Tensor create_device_tensor_from_host_data(
    const TensorSpec& tensor_spec,
    IDevice* device,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper,
    const HostBuffer& host_buffer) {
    auto strategy = prepare_tensor_conversion(
        host_buffer, optional_data_type, optional_layout, device != nullptr, memory_config, optional_tile);
    Tensor output;
    if (strategy) {
        output = convert_python_tensor_to_tt_tensor_on_device(
            py_tensor,
            optional_data_type,
            optional_layout,
            optional_tile,
            memory_config,
            device,
            cq_id,
            pad_value,
            mesh_mapper,
            strategy.value());
    } else {
        output = convert_python_tensor_to_tt_tensor_on_host(
            py_tensor,
            optional_data_type,
            optional_layout,
            optional_tile,
            memory_config,
            device,
            cq_id,
            pad_value,
            mesh_mapper);
    }
}
