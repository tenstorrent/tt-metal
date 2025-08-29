// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_conversion.hpp"

#include "ttnn/operations/core/core.hpp"
#include <tracy/Tracy.hpp>

using namespace tt::tt_metal;

namespace {

struct PyFromTorchConversionInput {
    std::string torch_dtype;
    DataType data_type;
    Layout layout;

    bool operator==(const PyFromTorchConversionInput& other) const {
        return torch_dtype == other.torch_dtype && data_type == other.data_type && layout == other.layout;
    }
};

struct PyFromTorchConversionInputHash {
    std::size_t operator()(const PyFromTorchConversionInput& input) const {
        std::size_t h1 = std::hash<std::string>{}(input.torch_dtype);
        std::size_t h2 = std::hash<int>{}(static_cast<int>(input.data_type));
        std::size_t h3 = std::hash<int>{}(static_cast<int>(input.layout));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

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
}  // namespace

Tensor tt::tt_metal::create_tt_tensor_from_py_data(
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

std::optional<PyTensorPreparedConversion> tt::tt_metal::prepare_torch_tensor_conversion(
    const std::string& torch_dtype,
    bool is_tensor_empty,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    bool has_device,
    const MemoryConfig& memory_config,
    const std::optional<Tile>& optional_tile) {
    // Early exit conditions -- on-device strategy is not supported
    if (!has_device ||
        // Device is required
        is_tensor_empty ||
        // to tile the tensor it must have non-zero volume or a sufficient rank -- if this fails
        // the tensor must be constructed on host.
        memory_config.is_sharded() ||
        // Sharded tensor handling and on-device type-casting cannot be done with the regular strategy
        (optional_tile.has_value() && (((optional_tile->get_tile_shape()[0] % tt::constants::TILE_WIDTH) != 0) ||
                                       ((optional_tile->get_tile_shape()[1] % tt::constants::TILE_HEIGHT) != 0))) ||
        // on-device tiling operation expects 32x32 row
        !map_torch_data_type_to_ttnn(torch_dtype).has_value()) {
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
    static std::unordered_map<PyFromTorchConversionInput, PyTensorPreparedConversion, PyFromTorchConversionInputHash>
        conversion_map = {
            // clang-format off

            // At the moment there are no cases that can be safely implemented with on-device
            // conversion, and bfloat16 cases are to be implemented in a follow-up PR to avoid
            // breaking too many tests in a scope of a single PR.

            // clang-format on
        };

    DataType expected_dtype = dtype.value_or(map_torch_data_type_to_ttnn(torch_dtype).value());
    PyFromTorchConversionInput input{
        .torch_dtype = torch_dtype,
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

std::optional<DataType> tt::tt_metal::map_torch_data_type_to_ttnn(const std::string& py_dtype) {
    if (py_dtype == "float32") {
        return DataType::FLOAT32;
    } else if (py_dtype == "float16") {
        return DataType::BFLOAT16;
    } else if (py_dtype == "bfloat16") {
        return DataType::BFLOAT16;
    } else if (py_dtype == "int64") {
        return DataType::UINT32;
    } else if (py_dtype == "int32") {
        return DataType::INT32;
    } else if (py_dtype == "int16") {
        return DataType::UINT16;
    } else if (py_dtype == "uint8") {
        return DataType::UINT8;
    } else {
        return std::nullopt;
    }
}
