// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pytensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "tools/profiler/op_profiler.hpp"
#include "ttnn-nanobind/bfloat_dtype_traits.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/ttnn_dtype_traits.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt_stl/overloaded.hpp>
#include <tt_stl/span.hpp>
#include <ttnn/tensor/to_string.hpp>

#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>

#include <tracy/Tracy.hpp>

using namespace tt::tt_metal;

namespace ttnn::tensor {
namespace CMAKE_UNIQUE_NAMESPACE {
namespace {

#ifdef DEBUG

void log_external_operation(const operation::ExternalOperation& operation, const std::vector<Tensor>& input_tensors) {
    log_debug(tt::LogOp, "Launching External Operation: \"{}\"", operation.get_type_name());

    auto attributes = operation.attributes();
    if (not attributes.empty()) {
        log_debug(tt::LogOp, "Attributes:");
        for ([[maybe_unused]] auto&& [name, value] : attributes) {
            log_debug(tt::LogOp, "\t{} = {}", name, value);
        }
    }

    log_debug(tt::LogOp, "Input std::vector<Tensor>:");
    for (auto index = 0; index < input_tensors.size(); index++) {
        [[maybe_unused]] const auto& tensor = input_tensors[index];
        log_debug(tt::LogOp, "\t{}: {}", index, tensor);
    }

    log_debug(tt::LogOp, "");
}
#else

void log_external_operation(
    const operation::ExternalOperation& /*operation*/, const std::vector<Tensor>& /*input_tensors*/) {}

#endif

template <typename... Ts>
Shape ndarray_shape_to_ttnn(const nb::ndarray<Ts...>& arr) {
    // nb shape uses 64 bit whereas ttnn uses 32 bit shape sizes, so convert explicitly
    Shape::Container shp;
    shp.reserve(arr.ndim());
    for (size_t i = 0; i < arr.ndim(); ++i) {
        shp.emplace_back(static_cast<Shape::coord_type>(arr.shape(i)));
    }

    return Shape{std::move(shp)};
}

template <typename T>
std::vector<size_t> ttnn_shape_to_ndarray(const T& arr) {
    // nb shape uses 64 bit whereas ttnn uses 32 bit shape sizes, so convert explicitly
    // int64_t
    std::vector<size_t> shp;
    shp.reserve(arr.size());
    for (auto dim : arr) {
        shp.emplace_back(static_cast<size_t>(dim));
    }

    return shp;
}

template <typename T>
Tensor create_typed_tt_tensor_from_py_data(
    const nb::ndarray<nb::array_api>& py_ndarray,
    const Shape& py_data_shape,
    const TensorLayout& tensor_layout,
    MeshDevice* device,
    const tt::tt_metal::MemoryPin& pydata_pin,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value,
    const distributed::TensorToMesh* mesh_mapper) {
    TT_FATAL(
        !tensor_layout.get_memory_config().is_sharded() || tensor_layout.get_memory_config().shard_spec().has_value() ||
            tensor_layout.get_memory_config().nd_shard_spec().has_value(),
        "Sharded tensors must have a shard spec when converting to tt tensors!");

    tt::stl::Span<T> pydata_span(reinterpret_cast<T*>(py_ndarray.data()), py_data_shape.volume());

    TT_FATAL(py_ndarray.is_valid(), "create_typed_tt_tensor_from_py_data: py_ndarray is invalid!");
    TT_FATAL(
        py_ndarray.size() == py_data_shape.volume(),
        "create_typed_tt_tensor_from_py_data: array size ({}) shape volume ({}) mismatch!",
        py_ndarray.size(),
        py_data_shape.volume());

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
    }
    return Tensor::from_span(
        tt::stl::make_const_span(pydata_span), tensor_spec, device, cq_id, static_cast<T>(pad_value));
}

Tensor create_tt_tensor_from_py_data(
    nb::ndarray<nb::array_api> py_data,
    const Shape& py_data_shape,
    const TensorLayout& tensor_layout,
    MeshDevice* device,
    const tt::tt_metal::MemoryPin& pydata_pin,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value,
    const distributed::TensorToMesh* mesh_mapper) {
    auto create_concrete = [&]<typename T>() {
        return create_typed_tt_tensor_from_py_data<T>(
            py_data, py_data_shape, tensor_layout, device, pydata_pin, cq_id, pad_value, mesh_mapper);
    };

    switch (tensor_layout.get_data_type()) {
        case DataType::UINT8: return create_concrete.template operator()<uint8_t>();
        case DataType::UINT16: return create_concrete.template operator()<uint16_t>();
        case DataType::INT32: return create_concrete.template operator()<int32_t>();
        case DataType::UINT32: return create_concrete.template operator()<uint32_t>();
        case DataType::FLOAT32: return create_concrete.template operator()<float>();
        case DataType::BFLOAT16: return create_concrete.template operator()<bfloat16>();
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            return create_concrete.template operator()<float>();
        }
        case DataType::INVALID: {
            TT_THROW("Unsupported DataType: {}", tensor_layout.get_data_type());
        }
    }

    TT_THROW("Unsupported DataType: {}", tensor_layout.get_data_type());
}

// Preprocess the python tensor, optionally performing dtype conversion.
// May need to create a handle and hold onto it here?
struct PreprocessedPyTensor {
    nb::ndarray<nb::array_api> contiguous_py_tensor;
    DataType data_type = DataType::INVALID;
};

PreprocessedPyTensor parse_py_tensor(nb::ndarray<nb::array_api> py_tensor, std::optional<DataType> optional_data_type) {
    auto py_tensor_dtype = py_tensor.dtype();
    // handle bool types by changing them to uint8
    // TODO: add proper handling for bool types as a DataType
    if (py_tensor_dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bool)) {
        py_tensor_dtype.code = static_cast<uint8_t>(nb::dlpack::dtype_code::UInt);
        py_tensor_dtype.bits = 8;
        py_tensor_dtype.lanes = 1;
    }

    DataType data_type = optional_data_type.value_or(get_ttnn_datatype_from_dtype(py_tensor_dtype));

    nb::detail::ndarray_config config(decltype(py_tensor)::Config{});
    config.dtype = get_dtype_from_ttnn_datatype(data_type);
    config.order = nb::c_contig::value;  // force row-major contiguous
    config.device_type = nb::device::cpu::value;

    TT_FATAL(
        data_type != DataType::INVALID || py_tensor.is_valid(),
        "parse_py_tensor: DataType::INVALID!.\n\t"
        "py_tensor: dtype.code: {}, dtype.bits: {}\n\t"
        "optional_data_type::has_value: {}, value: {}\n\t"
        "data_type: dtype.code: {}, dtype.bits: {}\n\t",
        py_tensor.dtype().code,
        py_tensor.dtype().bits,
        optional_data_type.has_value(),
        optional_data_type.value_or(DataType::INVALID),
        config.dtype.code,
        config.dtype.bits);

    nb::detail::ndarray_handle* converted_tensor_handle = nanobind::detail::ndarray_import(
        py_tensor.cast(nb::rv_policy::automatic).ptr(), &config, true /*convert*/, nullptr /*cleanup*/);

    return {.contiguous_py_tensor = nb::ndarray<nb::array_api>(converted_tensor_handle), .data_type = data_type};
}

Tensor convert_python_tensor_to_tt_tensor(
    nb::ndarray<nb::array_api> py_tensor,
    std::optional<DataType> optional_data_type,
    std::optional<Layout> optional_layout,
    const std::optional<Tile>& optional_tile,
    const MemoryConfig& memory_config,
    MeshDevice* device,
    std::optional<ttnn::QueueId> cq_id,
    float pad_value,
    const distributed::TensorToMesh* mesh_mapper) {
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::detail::convert_python_tensor_to_tt_tensor",
        py_tensor,
        optional_data_type,
        optional_layout,
        optional_tile,
        memory_config,
        device,
        cq_id,
        pad_value,
        mesh_mapper);

    auto preprocessed_py_tensor = parse_py_tensor(py_tensor, optional_data_type);
    const ttnn::Shape shape = ndarray_shape_to_ttnn(py_tensor);

    TT_FATAL(
        preprocessed_py_tensor.contiguous_py_tensor.size() == shape.volume(),
        "Number of elements from python tensor {} must match volume of shape {}!\n\t"
        "is_valid: {}, itemsize: {}, deviceid: {}, nbytes: {}, ndim: {}\n\t"
        "dtype.code: {}, dtype.bits: {}",
        preprocessed_py_tensor.contiguous_py_tensor.size(),
        shape.volume(),
        preprocessed_py_tensor.contiguous_py_tensor.is_valid(),
        preprocessed_py_tensor.contiguous_py_tensor.itemsize(),
        preprocessed_py_tensor.contiguous_py_tensor.device_id(),
        preprocessed_py_tensor.contiguous_py_tensor.nbytes(),
        preprocessed_py_tensor.contiguous_py_tensor.ndim(),
        preprocessed_py_tensor.contiguous_py_tensor.dtype().code,
        preprocessed_py_tensor.contiguous_py_tensor.dtype().bits);

    const Layout layout = [&]() {
        // Block float types require tile layout.
        // Choose tile by default and disallow overriding to anything else.
        if (preprocessed_py_tensor.data_type == DataType::BFLOAT8_B ||
            preprocessed_py_tensor.data_type == DataType::BFLOAT4_B) {
            TT_FATAL(
                !optional_layout.has_value() or optional_layout.value() == Layout::TILE,
                "Tile layout is required for tensor of type bfloat8_b or bfloat4_b; got {}.",
                optional_layout.value());
            return Layout::TILE;
        }
        return optional_layout.value_or(Layout::ROW_MAJOR);
    }();

    // Important: `nb::object` copying and destruction must be done while holding GIL, which nanobind ensures for a
    // thread that calls the C++ APIs. We wrap `nb::object` in `MemoryPin` so that multi-threaded C++ code only
    // increments / decrements the reference count on the memory pin; the last decrement to the pin should be triggered
    // from the nanobind caller thread, which will correctly decrement the `nb::object` reference count while hodling
    // GIL.
    tt::tt_metal::MemoryPin pydata_pin(
        std::make_shared<nb::ndarray<nb::array_api>>(preprocessed_py_tensor.contiguous_py_tensor));

    auto output = create_tt_tensor_from_py_data(
        preprocessed_py_tensor.contiguous_py_tensor,
        shape,
        TensorLayout(preprocessed_py_tensor.data_type, PageConfig(layout, optional_tile), memory_config),
        device,
        pydata_pin,
        cq_id,
        pad_value,
        mesh_mapper);

    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;

}  // Tensor convert_python_tensor_to_tt_tensor

// Wrapper around HostBuffer that provides a row-major view of the data, handles padding / logical view, and provides
// `shape` and `data_type` information.
struct RowMajorHostBuffer {
    static RowMajorHostBuffer create_padded(HostBuffer buffer, const ttnn::TensorSpec& tensor_spec) {
        tt::stl::Span<const uint32_t> shape_view = tensor_spec.padded_shape().view();
        return RowMajorHostBuffer{
            .buffer = std::move(buffer),
            .shape = std::vector<uint32_t>(shape_view.begin(), shape_view.end()),
            .data_type = tensor_spec.data_type(),
        };
    }

    static RowMajorHostBuffer create_logical(HostBuffer buffer, const ttnn::TensorSpec& tensor_spec) {
        tt::stl::Span<const uint32_t> shape_view = tensor_spec.logical_shape().view();
        return RowMajorHostBuffer{
            .buffer = std::move(buffer),
            .shape = std::vector<uint32_t>(shape_view.begin(), shape_view.end()),
            .data_type = tensor_spec.data_type(),
        };
    }

    HostBuffer buffer;
    std::vector<uint32_t> shape;
    ttnn::DataType data_type = ttnn::DataType::INVALID;
};

// Converts a TT tensor to a RowMajorHostBuffer.
//
// If `padded_output` is true, the returned buffer will be padded to the tile size.
// If `padded_output` is false, the returned buffer will be in logical view.
RowMajorHostBuffer convert_to_row_major_host_buffer(const Tensor& tt_tensor, const bool padded_output) {
    // conversion to Host storage after
    // issue #31136: to_torch with mesh_composer=None on device-sharded tensor
    if (std::holds_alternative<DeviceStorage>(tt_tensor.storage())) {
        return convert_to_row_major_host_buffer(tt_tensor.cpu(), padded_output);
    }

    const auto& tensor_spec = tt_tensor.tensor_spec();

    // Performs logical data conversion on the concrete data type.
    auto dispatch_to_concrete = [&tensor_spec, padded_output]<typename T>(HostBuffer host_buffer) {
        if (padded_output) {
            if (tensor_spec.layout() == Layout::TILE) {
                auto row_major_data = tensor_impl::convert_layout_tile_to_row_major(
                    tensor_spec.physical_shape(), tensor_spec.tile(), host_buffer.view_as<const T>());
                return RowMajorHostBuffer::create_padded(HostBuffer(std::move(row_major_data)), tensor_spec);
            }
            return RowMajorHostBuffer::create_padded(std::move(host_buffer), tensor_spec);
        }

        // Previous impl only copied if data needed transformation. Instead *always* copy
        // because the HostBuffer will be returned directly to the other python frameworks
        // wrapped in an ndarray

        auto logical_data = tensor_impl::decode_tensor_data(host_buffer.view_as<const T>(), tensor_spec);
        return RowMajorHostBuffer::create_logical(HostBuffer(std::move(logical_data)), tensor_spec);
    };

    auto convert_to_logical = [&tensor_spec, &dispatch_to_concrete](const HostBuffer& buffer) {
        const auto tt_dtype = tensor_spec.data_type();
        switch (tt_dtype) {
            case DataType::UINT8: return dispatch_to_concrete.template operator()<uint8_t>(buffer);
            case DataType::UINT16: return dispatch_to_concrete.template operator()<uint16_t>(buffer);
            case DataType::INT32: return dispatch_to_concrete.template operator()<int32_t>(buffer);
            case DataType::UINT32: return dispatch_to_concrete.template operator()<uint32_t>(buffer);
            case DataType::FLOAT32: return dispatch_to_concrete.template operator()<float>(buffer);
            case DataType::BFLOAT16: return dispatch_to_concrete.template operator()<bfloat16>(buffer);
            case DataType::BFLOAT8_B:
            case DataType::BFLOAT4_B: {
                const auto& tile = tensor_spec.tile();
                tt::stl::Span<const std::uint32_t> uint32_data = host_buffer::get_as<std::uint32_t>(buffer);
                auto float_unpacked_data = tt_dtype == DataType::BFLOAT8_B
                                               ? unpack_bfp8_tiles_into_float_vec(
                                                     uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile)
                                               : unpack_bfp4_tiles_into_float_vec(
                                                     uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
                auto input_float_buffer = tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
                return dispatch_to_concrete.template operator()<float>(input_float_buffer);
            }
            case DataType::INVALID: TT_THROW("Unsupported DataType: {}", tt_dtype);
        }
        TT_THROW("Unreachable");
    };

    return convert_to_logical(std::visit(
        tt::stl::overloaded{
            [](const HostStorage& storage) {
                std::vector<HostBuffer> buffers;
                storage.buffer().apply([&buffers](const HostBuffer& shard) { buffers.push_back(shard); });
                TT_FATAL(
                    buffers.size() == 1,
                    "Can't convert a tensor distributed on {} mesh to row-major logical tensor. Supply a mesh "
                    "composer "
                    "to concatenate multi-device shards.",
                    storage.buffer().shape());
                return buffers.front();
            },
            [&tt_tensor](auto&&) -> HostBuffer {
                TT_THROW(
                    "Tensor with {} cannot be converted to torch",
                    tt::stl::get_active_type_name_in_variant(tt_tensor.storage()));
            },
        },
        tt_tensor.storage()));
}

// Overload that converts a distributed tensor to a RowMajorHostBuffer.
//
// The returned buffer will be in logical view.
RowMajorHostBuffer convert_to_row_major_host_buffer(
    const Tensor& tt_tensor, const ttnn::distributed::MeshToTensor& mesh_composer) {
    //
    auto dispatch_to_concrete = [&mesh_composer]<typename T>(const Tensor& tt_tensor) {
        auto [data, shape] = mesh_composer.compose<T>(tt_tensor);
        tt::stl::Span<const uint32_t> shape_view = shape.view();
        return RowMajorHostBuffer{
            .buffer = HostBuffer(std::move(data)),
            .shape = std::vector<uint32_t>(shape_view.begin(), shape_view.end()),
            .data_type = tt_tensor.dtype(),
        };
    };

    switch (tt_tensor.dtype()) {
        case DataType::UINT8: return dispatch_to_concrete.template operator()<uint8_t>(tt_tensor);
        case DataType::UINT16: return dispatch_to_concrete.template operator()<uint16_t>(tt_tensor);
        case DataType::INT32: return dispatch_to_concrete.template operator()<int32_t>(tt_tensor);
        case DataType::UINT32: return dispatch_to_concrete.template operator()<uint32_t>(tt_tensor);
        case DataType::BFLOAT16: return dispatch_to_concrete.template operator()<bfloat16>(tt_tensor);
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
        case DataType::FLOAT32: return dispatch_to_concrete.template operator()<float>(tt_tensor);
        case DataType::INVALID: TT_THROW("Unsupported DataType: {}", tt_tensor.dtype());
    }
    TT_THROW("Unreachable");
}

template <typename Framework>
    requires requires {
        { Framework::is_framework == true };
    }
nb::ndarray<Framework> convert_tt_tensor_to_framework_tensor(RowMajorHostBuffer& row_major_host_buffer) {
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::detail::convert_tt_tensor_to_framework_tensor", row_major_host_buffer);

    auto shape_vec = ttnn_shape_to_ndarray(row_major_host_buffer.shape);

    // HostBuffer usage like this is shallow copy
    HostBuffer* buffer = new HostBuffer{row_major_host_buffer.buffer};

    nb::capsule owner(buffer, [](void* p) noexcept { delete static_cast<HostBuffer*>(p); });

    // Fiddling with sign bit to match previous behavior
    nb::dlpack::dtype dt = get_dtype_from_ttnn_datatype(row_major_host_buffer.data_type);
    if (dt.code == static_cast<std::uint8_t>(nb::dlpack::dtype_code::UInt) && dt.bits > 8) {
        dt.code = static_cast<std::uint8_t>(nb::dlpack::dtype_code::Int);
    }

    // ndarray constructor will make a deep copy of shape/stride, so no need to worry about ownership
    // with shape/stride pointers
    auto framework_tensor = nb::ndarray<Framework>(
        buffer->view_bytes().data(),
        shape_vec.size(),
        shape_vec.data(),
        std::move(owner),
        nullptr /*strides*/,
        dt,
        nb::device::cpu::value,
        0 /*device_id*/,
        nb::c_contig::value);

    GraphTracker::instance().track_function_end(framework_tensor);

    return framework_tensor;
}

auto parse_external_operation(
    const nb::callable& external_operation, std::optional<std::string> function_name_override = std::nullopt) {
    std::string function_name;
    if (function_name_override.has_value()) {
        function_name = function_name_override.value();
    } else {
        function_name = nb::cast<std::string>(external_operation.attr("__qualname__"));
    }

    // original impl had a bunch of no-ops. Reduce to minimum functionality.
    std::vector<Tensor> input_tensors;
    tt::stl::reflection::Attributes attributes;

    auto operation = tt::tt_metal::operation::ExternalOperation{function_name, attributes};
    return std::make_tuple(operation, input_tensors);
}

}  // namespace
}  // namespace CMAKE_UNIQUE_NAMESPACE

void pytensor_module_types(nb::module_& mod) {
    // Tensor constructors that accept device and .to_device() function use keep alive call policy to communicate that
    // Device needs to outlive Tensor. This is because when tensors on device are destroyed they need to deallocate
    // their buffers via device. keep_alive increases the ref count of the Device object being passed into the
    // constructor and .to_device() function. For additional info see:
    // https://nanobind.readthedocs.io/en/latest/functions.html#lifetime-annotations
    auto pyTensor = nb::class_<Tensor>(mod, "Tensor", R"doc(

        Class constructor supports tensors of any rank.
        The constructor takes following arguments:

        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        |  Argument  |                 Description                            |       Data type           |           Valid range              | Required |
        +============+========================================================+===========================+====================================+==========+
        | data       | Data to store in TT tensor                             | List[float/int]           |                                    | Yes      |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | shape      | Shape of TT tensor                                     | List[int]                 |                                    | Yes      |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | data_type  | Data type of numbers in TT tensor                      | ttnn.DataType             | ttnn.DataType.BFLOAT16             | Yes      |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | ttnn.DataType.FLOAT32              |          |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | ttnn.DataType.UINT32               |          |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | ttnn.DataType.BFLOAT8_B            |          |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | ttnn.DataType.BFLOAT4_B            |          |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | layout     | Layout of tensor data in memory                        | ttnn.Layout               | ttnn.Layout.ROW_MAJOR              | Yes      |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | ttnn.Layout.TILE                   |          |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | device     | Device on which tensor will be created                 | ttnn.Device               | Host or TT accelerator device      | No       |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | mem_config | Layout of tensor in TT Accelerator device memory banks | ttnn.MemoryConfig         |                                    | No       |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+

    )doc");
}

void pytensor_module(nb::module_& mod) {
    mod.def(
        "decorate_external_operation",
        [](const nb::callable& function, const std::optional<std::string>& function_name) -> nb::object {
            return nb::cpp_function(
                std::function([function, function_name](const nb::args& args, const nb::kwargs& kwargs) {
                    ZoneScopedN("TT_DNN_FALLBACK_OP");
                    auto [operation, input_tensors] =
                        CMAKE_UNIQUE_NAMESPACE::parse_external_operation(function, function_name);
                    GraphTracker::instance().track_function_start(operation.get_type_name(), args, kwargs);
                    CMAKE_UNIQUE_NAMESPACE::log_external_operation(operation, input_tensors);
                    auto output = function(*args, **kwargs);
                    TracyOpTTNNExternal(
                        operation, input_tensors, ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id());
                    GraphTracker::instance().track_function_end(output);
                    return output;
                }));
        },
        // nb::rv_policy::move,
        nb::arg("function").noconvert(),
        nb::arg("function_name").noconvert() = nb::none(),
        R"doc(
        Decorate external operation for purposes of reporting and profiling.

            +----------+----------------------+-----------+-------------+----------+
            | Argument | Description          | Data type | Valid range | Required |
            +==========+======================+===========+=============+==========+
            | function | Fallback Operation   | Function  |             | Yes      |
            +----------+----------------------+-----------+-------------+----------+
            | args     | Packed args          | tuple     |             | No       |
            +----------+----------------------+-----------+-------------+----------+
            | kwargs   | Packed kwargs        | dict      |             | No       |
            +----------+----------------------+-----------+-------------+----------+
    )doc");

    // Helper template to add tensor constructors for different data types
    auto add_typed_constructors = []<typename T>(nb::class_<Tensor>& pyTensor, T default_pad_value) {
        // Constructor 1: Host-only (no device, no mem_config)
        pyTensor.def(
            "__init__",
            [](Tensor* t,
               std::vector<T>&& data,
               const ttnn::SmallVector<uint32_t>& shape,
               DataType data_type,
               Layout layout,
               const std::optional<Tile>& tile,
               T pad_value) {
                new (t) Tensor(Tensor::from_vector(
                    std::move(data),
                    TensorSpec(ttnn::Shape(shape), TensorLayout(data_type, PageConfig(layout, tile), MemoryConfig{})),
                    /*device=*/nullptr,
                    std::nullopt,
                    pad_value));
            },
            nb::arg("data"),
            nb::arg("shape"),
            nb::arg("data_type"),
            nb::arg("layout"),
            nb::arg("tile") = nb::none(),
            nb::arg("pad_value") = default_pad_value,
            nb::rv_policy::move,
            R"doc(
                Create a TT Tensor on host from data vector.

                +---------------+----------------------+
                | Argument      | Description          |
                +===============+======================+
                | data          | Data vector          |
                +---------------+----------------------+
                | shape         | Tensor shape         |
                +---------------+----------------------+
                | data_type     | Data type            |
                +---------------+----------------------+
                | layout        | Memory layout        |
                +---------------+----------------------+
                | tile          | Tile spec (optional) |
                +---------------+----------------------+
                | pad_value     | Pad value (optional) |
                +---------------+----------------------+

                .. code-block:: python

                    # Float tensor
                    ttnn.Tensor([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2],
                                ttnn.DataType.FLOAT32, ttnn.Layout.ROW_MAJOR)

                    # Integer tensor
                    ttnn.Tensor([1, 2, 3, 4], [1, 1, 2, 2],
                                ttnn.DataType.INT32, ttnn.Layout.ROW_MAJOR)
            )doc");

        // Constructor 2: With optional device
        pyTensor.def(
            "__init__",
            [](Tensor* t,
               std::vector<T>&& data,
               const ttnn::SmallVector<uint32_t>& shape,
               DataType data_type,
               Layout layout,
               std::optional<MeshDevice*> device,
               const std::optional<Tile>& tile,
               T pad_value) {
                new (t) Tensor(Tensor::from_vector(
                    std::move(data),
                    TensorSpec(ttnn::Shape(shape), TensorLayout(data_type, PageConfig(layout, tile), MemoryConfig{})),
                    device.value_or(nullptr),
                    std::nullopt,
                    pad_value));
            },
            nb::keep_alive<1, 6>(),
            nb::arg("data"),
            nb::arg("shape"),
            nb::arg("data_type"),
            nb::arg("layout"),
            nb::arg("device") = nb::none(),
            nb::arg("tile") = nb::none(),
            nb::arg("pad_value") = default_pad_value,
            nb::rv_policy::move,
            R"doc(
                Create a TT Tensor on device from data vector.

                +---------------+---------------------------+
                | Argument      | Description               |
                +===============+===========================+
                | data          | Data vector               |
                +---------------+---------------------------+
                | shape         | Tensor shape              |
                +---------------+---------------------------+
                | data_type     | Data type                 |
                +---------------+---------------------------+
                | layout        | Memory layout             |
                +---------------+---------------------------+
                | device        | Device ptr (optional)     |
                +---------------+---------------------------+
                | tile          | Tile spec (optional)      |
                +---------------+---------------------------+
                | pad_value     | Pad value (optional)      |
                +---------------+---------------------------+

                .. code-block:: python

                    device = ttnn.open_device(0)
                    ttnn.Tensor([1, 2, 3, 4], [1, 1, 2, 2],
                                ttnn.DataType.INT32, ttnn.Layout.ROW_MAJOR, device)
            )doc");

        // Constructor 3: With device and mem_config
        pyTensor.def(
            "__init__",
            [](Tensor* t,
               std::vector<T>&& data,
               const ttnn::SmallVector<uint32_t>& shape,
               DataType data_type,
               Layout layout,
               std::optional<MeshDevice*> device,
               const MemoryConfig& memory_config,
               const std::optional<Tile>& tile,
               T pad_value) {
                new (t) Tensor(Tensor::from_vector(
                    std::move(data),
                    TensorSpec(ttnn::Shape(shape), TensorLayout(data_type, PageConfig(layout, tile), memory_config)),
                    device.value_or(nullptr),
                    std::nullopt,
                    pad_value));
            },
            nb::keep_alive<1, 7>(),  // 1,6?
            nb::arg("data"),
            nb::arg("shape"),
            nb::arg("data_type"),
            nb::arg("layout"),
            nb::arg("device") = nb::none(),
            nb::arg("memory_config"),
            nb::arg("tile") = nb::none(),
            nb::arg("pad_value") = default_pad_value,
            nb::rv_policy::move,
            R"doc(
                Create a TT Tensor on device with specified memory config.

                +---------------+---------------------------+
                | Argument      | Description               |
                +===============+===========================+
                | data          | Data vector               |
                +---------------+---------------------------+
                | shape         | Tensor shape              |
                +---------------+---------------------------+
                | data_type     | Data type                 |
                +---------------+---------------------------+
                | layout        | Memory layout             |
                +---------------+---------------------------+
                | device        | Device ptr (optional)     |
                +---------------+---------------------------+
                | mem_config    | Memory config             |
                +---------------+---------------------------+
                | tile          | Tile spec (optional)      |
                +---------------+---------------------------+
                | pad_value     | Pad value (optional)      |
                +---------------+---------------------------+

                .. code-block:: python

                    device = ttnn.open_device(0)
                    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED)
                    ttnn.Tensor([1, 2, 3, 4], [1, 1, 2, 2],
                                ttnn.DataType.INT32, ttnn.Layout.ROW_MAJOR,
                                device, mem_config)
            )doc");
    };

    auto pyTensor = static_cast<nb::class_<Tensor>>(mod.attr("Tensor"));
    pyTensor.def(nb::init<ttnn::Tensor&>());

    // Register constructors for float and int32_t data types
    add_typed_constructors.operator()<float>(pyTensor, 0.0f);
    add_typed_constructors.operator()<int32_t>(pyTensor, 0);

    pyTensor
        .def(
            "__init__",
            [](Tensor* t,
               nb::ndarray<nb::array_api> dlpack_tensor,
               std::optional<DataType> data_type,
               std::optional<MeshDevice*> device,
               std::optional<Layout> layout,
               const std::optional<MemoryConfig>& mem_config,
               const std::optional<Tile>& tile,
               std::optional<ttnn::QueueId> cq_id,
               std::optional<float> pad_value,
               const distributed::TensorToMesh* mesh_mapper) {
                new (t) Tensor(CMAKE_UNIQUE_NAMESPACE::convert_python_tensor_to_tt_tensor(
                    std::move(dlpack_tensor),
                    data_type,
                    layout,
                    tile,
                    mem_config.value_or(MemoryConfig{}),
                    device.value_or(nullptr),
                    cq_id,
                    pad_value.value_or(0.0f),
                    mesh_mapper));
            },
            nb::arg("tensor").noconvert(false),
            nb::arg("data_type") = nb::none(),
            nb::arg("device") = nb::none(),
            nb::arg("layout").noconvert() = nb::none(),
            nb::arg("mem_config").noconvert() = nb::none(),
            nb::arg("tile").noconvert() = nb::none(),
            nb::arg("cq_id") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("mesh_mapper") = nullptr,
            nb::keep_alive<1, 4>(),  // test: matches other k_a
            nb::rv_policy::move,
            R"doc(
                +--------------+--------------------------------+
                | Argument     | Description                    |
                +==============+================================+
                | tensor       | dlpack compliant Tensor        |
                +--------------+--------------------------------+
                | data_type    | TT Tensor data type (optional) |
                +--------------+--------------------------------+
                | device       | TT device ptr (optional)       |
                +--------------+--------------------------------+
                | layout       | TT layout (optional)           |
                +--------------+--------------------------------+
                | mem_config   | TT memory_config (optional)    |
                +--------------+--------------------------------+
                | tile         | TT Tile Spec (optional)        |
                +--------------+--------------------------------+
                | cq_id        | TT Command Queue ID (optional) |
                +--------------+--------------------------------+
                | pad_value    | Padding value (optional)       |
                +--------------+--------------------------------+
                | mesh_mapper  | TT-NN Mesh Mapper (optional)    |
                +--------------+--------------------------------+

                Example of creating a TT Tensor from numpy tensor:

                .. code-block:: python

                    device = ttnn.open_device(device_id=0)
                    py_tensor = np.zeros((1, 1, 32, 32))
                    ttnn.Tensor(py_tensor, ttnn.bfloat16, device, ttnn.TILE_LAYOUT)
            )doc")
        .def_prop_ro("spec", [](const Tensor& self) { return self.tensor_spec(); })
        .def_prop_ro("shape", [](const Tensor& self) { return self.logical_shape(); })
        .def_prop_ro("padded_shape", [](const Tensor& self) { return self.padded_shape(); })
        .def_prop_ro("dtype", [](const Tensor& self) { return self.dtype(); })
        .def_prop_ro("layout", [](const Tensor& self) { return self.layout(); })
        .def_prop_ro("tile", [](const Tensor& self) { return self.tensor_spec().tile(); })
        .def(
            "deallocate",
            [](Tensor& self, bool force) { self.deallocate(force); },
            nb::arg("force") = false,
            R"doc(
                Dellocates all data of a tensor. This either deletes all host data or deallocates tensor data from device memory.
            )doc")
        .def(
            "to",
            [](const Tensor& self,
               MeshDevice* device,
               std::optional<const MemoryConfig>& mem_config,
               std::optional<ttnn::QueueId> cq_id) { return self.to_device(device, mem_config, cq_id); },
            nb::arg("device").noconvert(),
            nb::arg("mem_config").noconvert() = nb::none(),
            nb::arg("cq_id") = nb::none(),
            nb::keep_alive<0, 2>(),
            R"doc(
            Move TT Tensor from host device to TT accelerator device.

            Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

            If ``arg1`` is not supplied, memory config from the source tensor will be used.

            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range           | Required |
            +===========+=================================================+============================+=======================+==========+
            | arg0      | MeshDevice to which tensor will be moved        | ttnn.MeshDevice            | TT accelerator device | Yes      |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | arg1      | MemoryConfig of tensor of TT accelerator device | ttnn.MemoryConfig          |                       | No       |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | arg2      | CQ ID of TT accelerator device to use           | uint8_t                    |                       | No       |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_device)
        )doc")
        .def(
            "extract_shard",
            [](const Tensor& self, CoreCoord core) { return self.extract_shard(core); },
            nb::arg("core").noconvert(),
            nb::keep_alive<0, 2>(),  // orig
            // nb::keep_alive<0, 1>(), // does this work?
            R"doc(
            Move TT Tensor from host device to TT accelerator device.

            Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

            If ``arg1`` is not supplied, default ``MemoryConfig`` with ``interleaved`` set to ``True``.

            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range           | Required |
            +===========+=================================================+============================+=======================+==========+
            | arg0      | Core who's shard we want                        | ttnn.CoreCoord             | TT accelerator device | Yes      |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+


            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_device)
        )doc")
        .def(
            "extract_shard",
            [](const Tensor& self, const uint32_t& core_id) { return self.extract_shard(core_id); },
            nb::arg("core_id").noconvert(),
            nb::keep_alive<0, 2>(),  // orig
            // nb::keep_alive<0, 1>(),
            R"doc(
            Move TT Tensor from host device to TT accelerator device.

            Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

            If ``arg1`` is not supplied, default ``MemoryConfig`` with ``interleaved`` set to ``True``.

            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range           | Required |
            +===========+=================================================+============================+=======================+==========+
            | arg0      | Core who's shard we want                        | uint32_t                   | TT accelerator device | Yes      |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+


            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_device)
        )doc")
        .def(
            "cpu",
            [](const Tensor& self, bool blocking, std::optional<ttnn::QueueId> cq_id) {
                return self.cpu(blocking, cq_id);
            },
            nb::arg("blocking") = true,
            nb::arg("cq_id") = nb::none(),
            R"doc(
            Move TT Tensor from TT accelerator device to host device.

            .. code-block:: python

                tt_tensor = tt_tensor.cpu()
        )doc")
        .def(
            "item",
            [](const Tensor& self) -> nb::object {
                TT_FATAL(
                    self.logical_volume() == 1,
                    "tensor.item() requires tensor to have exactly one element, but got {} elements",
                    self.logical_volume());
                switch (self.dtype()) {
                    case DataType::FLOAT32: return nb::cast(self.to_vector<float>()[0]);
                    case DataType::BFLOAT16: return nb::cast(static_cast<float>(self.to_vector<bfloat16>()[0]));
                    case DataType::BFLOAT8_B:
                    case DataType::BFLOAT4_B: return nb::cast(self.to_vector<float>()[0]);
                    case DataType::INT32: return nb::cast(self.to_vector<int32_t>()[0]);
                    case DataType::UINT32: return nb::cast(self.to_vector<uint32_t>()[0]);
                    case DataType::UINT16: return nb::cast(self.to_vector<uint16_t>()[0]);
                    case DataType::UINT8: return nb::cast(self.to_vector<uint8_t>()[0]);
                    case DataType::INVALID: TT_THROW("Unsupported DataType");
                }
                TT_THROW("Unreachable");
            },
            // nb::rv_policy::copy,
            R"doc(
                 Extract the scalar value from a tensor containing exactly one element.

                 Similar to PyTorch's tensor.item(), this method returns the value of this tensor as a standard Python number.
                 This only works for tensors with one element.

                 Returns:
                     Python scalar: The scalar value contained in the tensor.

                 Raises:
                     RuntimeError: If the tensor doesn't contain exactly one element.

                 .. code-block:: python

                     # Create a tensor with one element
                     scalar_tensor = ttnn.from_torch(torch.tensor([3.14]), device=device)
                     value = scalar_tensor.item()  # Returns 3.14
             )doc")
        .def(
            "to",
            nb::overload_cast<Layout>(&Tensor::to_layout, nb::const_),
            nb::arg("target_layout").noconvert(),
            R"doc(
            Convert TT Tensor to provided memory layout. Available layouts conversions are:

            * ROW_MAJOR to TILE
            * TILE to ROW_MAJOR

            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range                    | Required |
            +===========+=================================================+============================+================================+==========+
            | arg0      | Target memory layout                            | ttnn.Layout                | ROW_MAJOR, TILE                | Yes      |
            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(ttnn.Layout.TILE)
        )doc")
        .def(
            "pad",
            [](const Tensor& self,
               const ttnn::SmallVector<uint32_t>& output_tensor_shape,
               const ttnn::SmallVector<uint32_t>& input_tensor_start,
               float pad_value) {
                return self.pad(ttnn::Shape(output_tensor_shape), ttnn::Shape(input_tensor_start), pad_value);
            },
            R"doc(
            Pad TT Tensor with given pad value ``arg2``.

            The input tensor must be on host and in ROW_MAJOR layout.

            Returns an output tensor that contains the input tensor at the given input tensor start indices ``arg1`` and the padded value everywhere else.

            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | Argument            | Description                                          | Data type    | Valid range                                         | Required |
            +=====================+======================================================+==============+=====================================================+==========+
            | arg0                | Shape of output tensor                               | List[int[4]] |                                                     | Yes      |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | arg1                | Start indices to place input tensor in output tensor | List[int[4]] | Values along each dim must be                       | Yes      |
            |                     |                                                      |              |                                                     |          |
            |                     |                                                      |              | <= (output_tensor_shape[i] - input_tensor_shape[i]) |          |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | arg2                | Value to pad input tensor                            | float        |                                                     | Yes      |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+

            .. code-block:: python

                input_tensor_shape = [1, 1, 3, 3]
                output_tensor_shape = [1, 2, 5, 5]
                input_tensor_start = [0, 1, 1, 1]
                pad_value = 0

                inp = torch.Tensor(
                    [ 1, 2, 3,
                    4, 5, 6,
                    7, 8, 9 ]
                )
                tt_tensor = ttnn.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttnn.DataType.BFLOAT16,
                    ttnn.Layout.ROW_MAJOR,
                )
                tt_tensor_padded = tt_tensor.pad(output_tensor_shape, input_tensor_start, pad_value)

                print("Input tensor:")
                print(tt_tensor)
                print("\nPadded tensor:")
                print(tt_tensor_padded)

            Example output:

            .. code-block::

                Input tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]

                Padded tensor:
                [ [[[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                    [[0, 0, 0, 0, 0],
                    [0, 1, 2, 3, 0],
                    [0, 4, 5, 6, 0],
                    [0, 7, 8, 9, 0],
                    [0, 0, 0, 0, 0]]] dtype=bfloat16 ]
        )doc")
        .def(
            "unpad",
            [](const Tensor& self,
               const ttnn::SmallVector<uint32_t>& output_tensor_start,
               const ttnn::SmallVector<uint32_t>& output_tensor_end) {
                return self.unpad(ttnn::Shape(output_tensor_start), ttnn::Shape(output_tensor_end));
            },
            R"doc(
            Unpad this TT Tensor.

            This tensor must be on host and in ROW_MAJOR layout.

            Returns an output tensor from output tensor start indices ``arg0`` to output tensor end indices ``arg1`` (inclusive) of the input tensor.

            +---------------------+----------------------------------------------+--------------+-----------------------------------------------------+----------+
            | Argument            | Description                                  | Data type    | Valid range                                         | Required |
            +=====================+==============================================+==============+=====================================================+==========+
            | arg0                | Start indices of input tensor                | List[int]    | Values along each dim must be                       | Yes      |
            |                     |                                              |              |                                                     |          |
            |                     |                                              |              | < input_tensor_shape[i] and <= output_tensor_end[i] |          |
            +---------------------+----------------------------------------------+--------------+-----------------------------------------------------+----------+
            | arg1                | End indices of input tensor in output tensor | List[int]    | Values along each dim must be                       | Yes      |
            |                     |                                              |              |                                                     |          |
            |                     |                                              |              | < input_tensor_shape[i]                             |          |
            +---------------------+----------------------------------------------+--------------+-----------------------------------------------------+----------+

            .. code-block:: python

                input_tensor_shape = [1, 1, 5, 5]
                output_tensor_start = [0, 0, 1, 1]
                output_tensor_end = [0, 0, 3, 3]

                inp = torch.Tensor(
                    [ 0, 0, 0, 0, 0,
                    0, 1, 2, 3, 0,
                    0, 4, 5, 6, 0,
                    0, 7, 8, 9, 0,
                    0, 0, 0, 0, 0 ]
                )
                tt_tensor = ttnn.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttnn.DataType.BFLOAT16,
                    ttnn.Layout.ROW_MAJOR,
                )
                tt_tensor_unpadded = tt_tensor.unpad(output_tensor_start, output_tensor_end)

                print("Input tensor:")
                print(tt_tensor)
                print("\nUnpadded tensor:")
                print(tt_tensor_unpadded)

            Example output:

            .. code-block::

                Input tensor:
                [ [[[0, 0, 0, 0, 0],
                    [0, 1, 2, 3, 0],
                    [0, 4, 5, 6, 0],
                    [0, 7, 8, 9, 0],
                    [0, 0, 0, 0, 0]]] dtype=bfloat16 ]

                Unpadded tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]
        )doc")
        .def(
            "pad_to_tile", [](const Tensor& self, float pad_value) { return self.pad_to_tile(pad_value); }, R"doc(
            Pads TT Tensor with given pad value ``arg0``.

            The input tensor must be on host and in ROW_MAJOR layout.

            Returns an output tensor that contains the input tensor padded with the padded value in the last two dims to multiples of 32.

            Padding will be added to the right and bottom of the tensor.

            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | Argument            | Description                                          | Data type    | Valid range                                         | Required |
            +=====================+======================================================+==============+=====================================================+==========+
            | arg0                | Value to pad input tensor                            | float        |                                                     | Yes      |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+

            .. code-block:: python

                input_tensor_shape = [1, 1, 3, 3]
                pad_value = 0

                inp = torch.Tensor(
                    [ 1, 2, 3,
                    4, 5, 6,
                    7, 8, 9 ]
                )
                tt_tensor = ttnn.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttnn.DataType.BFLOAT16,
                    ttnn.Layout.ROW_MAJOR,
                )
                tt_tensor_padded = tt_tensor.pad_to_tile(pad_value)

                print("Input tensor:")
                print(tt_tensor)
                print("\nPadded tensor:")
                print(tt_tensor_padded)

            Example output:

            .. code-block::

                Input tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]

                Padded tensor:
                [ [[[1, 2, 3, 0, ..., 0],
                    [4, 5, 6, 0, ..., 0],
                    [7, 8, 9, 0, ..., 0],
                    [0, 0, 0, 0, ..., 0],
                    ...,
                    [0, 0, 0, 0, ..., 0]]] dtype=bfloat16 ]
        )doc")
        .def(
            "unpad_from_tile",
            [](const Tensor& self, const ttnn::SmallVector<uint32_t>& output_tensor_shape) {
                return self.unpad_from_tile(ttnn::Shape(output_tensor_shape));
            },
            R"doc(
            Unpads TT Tensor from given input tensor ``arg0``.

            The input tensor must be on host and in ROW_MAJOR layout.

            This function expects the real data to aligned on the top left of the tensor.

            Returns an output tensor with padding removed from the right and bottom of the input tensor.

            +---------------------+----------------------------------------------+--------------+------------------------------------------------------------------------------+----------+
            | Argument            | Description                                  | Data type    | Valid range                                                                  | Required |
            +=====================+==============================================+==============+==============================================================================+==========+
            | arg0                | Shape of output tensor                       | List[int[4]] | All dims must match the input tensor dims apart from the last two dims.      | Yes      |
            |                     |                                              |              |                                                                              |          |
            |                     |                                              |              | Last two dims have the following restrictions:                               |          |
            |                     |                                              |              |                                                                              |          |
            |                     |                                              |              | input_tensor_shape[i] must be a multiple of 32                               |          |
            |                     |                                              |              |                                                                              |          |
            |                     |                                              |              | input_tensor_shape[i] - 32 < output_tensor_shape[i] <= input_tensor_shape[i] |          |
            +---------------------+----------------------------------------------+--------------+------------------------------------------------------------------------------+----------+


            .. code-block:: python

                input_tensor_shape = [1, 1, 32, 32]
                output_tensor_shape = [1, 1, 3, 3]

                inp = torch.arange(start=1.0, end=10.0).reshape(1, 1, 3, 3)
                inp = torch.nn.functional.pad(inp, [0, input_tensor_shape[3] - inp.shape[3], 0, input_tensor_shape[2] - inp.shape[2]]).reshape(-1)
                tt_tensor = ttnn.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttnn.DataType.BFLOAT16,
                    ttnn.Layout.ROW_MAJOR,
                )
                tt_tensor_unpadded = tt_tensor.unpad_from_tile(output_tensor_shape)

                print("Input tensor:")
                print(tt_tensor)
                print("\nUnpadded tensor:")
                print(tt_tensor_unpadded)

            Example output:

            .. code-block::

                Input tensor:
                [ [[[1, 2, 3, 0, ..., 0],
                    [4, 5, 6, 0, ..., 0],
                    [7, 8, 9, 0, ..., 0],
                    [0, 0, 0, 0, ..., 0],
                    ...,
                    [0, 0, 0, 0, ..., 0]]] dtype=bfloat16 ]

                Unpadded tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]
        )doc")
        .def(
            "__repr__", [](const Tensor& self) { return ttnn::to_string(self); }, R"doc(
            Prints the tensor as list of nested lists. Number of levels of nesting is equal to tensor rank.

            .. code-block:: python

                print(tt_tensor)

            Example output for a rank 4 TT Tensor with shape (1, 1, 32, 32):

            .. code-block::

                [ [[[0.220703, 0.839844, 0.960938, ..., 0.378906, 0.507812],
                [0.03125, 0.511719, 0.0407715, ..., 0.945312, 0.671875],
                ...
                [0.433594, 0.165039, 0.980469, ..., , 0.349609]]] dtype=bfloat16 ]

        )doc")
        .def(
            // TODO: Rename to physical_volume
            "volume",
            [](const Tensor& self) { return self.physical_volume(); },
            R"doc(
            Get the volume of the tensor.

            .. code-block:: python

                volume = tt_tensor.physical_volume()

        )doc")
        .def(
            "logical_volume",
            [](const Tensor& self) { return self.logical_volume(); },
            R"doc(
            Get the logical volume of the tensor.

            .. code-block:: python

                volume = tt_tensor.logical_volume()

        )doc")
        .def(
            "storage_type",
            [](const Tensor& self) { return self.storage_type(); },
            R"doc(
            Check if the tensor is on host

            .. code-block:: python

                storage_type = tt_tensor.storage_type()

        )doc")
        .def(
            "device",
            [](const Tensor& self) { return dynamic_cast<MeshDevice*>(self.device()); },
            R"doc(
            Get the device of the tensor.

            .. code-block:: python

                device = tt_tensor.device()

        )doc",
            nb::rv_policy::reference)
        .def(
            "devices",
            [](const Tensor& self) { return std::vector<MeshDevice*>{dynamic_cast<MeshDevice*>(self.device())}; },
            R"doc(
            Get devices tensor is mapped on to.

            .. code-block:: python

                devices = tt_tensor.devices()

        )doc",
            nb::rv_policy::reference)
        .def(
            "to_torch_with_padded_shape",
            [](const Tensor& self) -> nb::ndarray<nb::pytorch> {
                using namespace CMAKE_UNIQUE_NAMESPACE;

                auto buffer = convert_to_row_major_host_buffer(self, /*padded_output=*/true);
                return convert_tt_tensor_to_framework_tensor<nb::pytorch>(buffer);
            },
            nb::rv_policy::take_ownership,
            R"doc(
            Convert tensor to torch tensor using legacy padded shape.
            WARNING: Will be deprecated soon!

            The tensor must be on host when calling this function.

            .. code-block:: python

                data = tt_tensor.cpu().to_torch_with_padded_shape() # move TT Tensor to host and convert it to torch tensor

        )doc")
        .def(
            "to_torch",
            [](const Tensor& self, const ttnn::distributed::MeshToTensor* mesh_composer) -> nb::ndarray<nb::pytorch> {
                using namespace CMAKE_UNIQUE_NAMESPACE;

                auto buffer = mesh_composer ? convert_to_row_major_host_buffer(self, *mesh_composer)
                                            : convert_to_row_major_host_buffer(self, /*padded_output=*/false);
                return convert_tt_tensor_to_framework_tensor<nb::pytorch>(buffer);
            },
            nb::rv_policy::take_ownership,
            nb::arg("mesh_composer") = nullptr,
            R"doc(
            Convert tensor to torch tensor.

            The tensor must be on host when calling this function.

            .. code-block:: python

                data = tt_tensor.cpu().to_torch() # move TT Tensor to host and convert it to torch tensor

        )doc")
        .def(
            "to_numpy",
            [](const Tensor& self, const ttnn::distributed::MeshToTensor* mesh_composer) -> nb::ndarray<nb::numpy> {
                using namespace CMAKE_UNIQUE_NAMESPACE;

                auto buffer = mesh_composer ? convert_to_row_major_host_buffer(self, *mesh_composer)
                                            : convert_to_row_major_host_buffer(self, /*padded_output=*/false);
                return convert_tt_tensor_to_framework_tensor<nb::numpy>(buffer);
            },
            nb::rv_policy::take_ownership,
            nb::arg("mesh_composer") = nullptr,
            R"doc(
            Convert tensor to numpy tensor.

            The tensor must be on host when calling this function.

            .. code-block:: python

                data = tt_tensor.cpu().to_numpy() # move TT Tensor to host and convert it to numpy tensor

        )doc")
        .def(
            "host_buffer",
            [](Tensor& self) -> DistributedHostBuffer {
                TT_FATAL(self.storage_type() == StorageType::HOST, "Tensor must be on host to access host_buffer");
                return self.host_storage().buffer();
            },
            nb::rv_policy::reference_internal,
            R"doc(
            Get the underlying host buffer.

            The tensor must be on the cpu when calling this function.

            .. code-block:: python

                buffer = tt_tensor.cpu().host_buffer() # move TT Tensor to host and get the buffer

        )doc")
        .def(
            "buffer_address",
            [](const Tensor& self) -> uint32_t {
                return std::visit(
                    tt::stl::overloaded{
                        [](const DeviceStorage& s) -> uint32_t {
                            TT_FATAL(s.mesh_buffer != nullptr, "Tensor is not allocated.");
                            return s.mesh_buffer->address();
                        },
                        [&](auto&&) -> uint32_t {
                            TT_THROW(
                                "{} doesn't support buffer_address method",
                                tt::stl::get_active_type_name_in_variant(self.storage()));
                        },
                    },
                    self.storage());
            },
            R"doc(
            Get the address of the underlying buffer.

            The tensor must be on the single device when calling this function.

            .. code-block:: python

                address = tt_tensor.buffer_address()

        )doc")
        .def(
            "get_layout", [](const Tensor& self) { return self.layout(); }, R"doc(
            Get memory layout of TT Tensor.

            .. code-block:: python

                layout = tt_tensor.layout()

        )doc")
        .def(
            "get_tile", [](const Tensor& self) { return self.tensor_spec().tile(); }, R"doc(
            Get tile dims of TT Tensor.

            .. code-block:: python

                tile = tt_tensor.get_tile()

        )doc")
        .def(
            "memory_config", [](const Tensor& self) { return self.memory_config(); }, R"doc(
            Get buffer type of TT Tensor.

            .. code-block:: python

                memory_config = tt_tensor.memory_config()

        )doc")
        .def(
            "is_allocated", [](const Tensor& self) { return self.is_allocated(); }, R"doc(
            Check if TT Tensor is allocated.

            .. code-block:: python

                is_sharded = tt_tensor.is_sharded()

        )doc")
        .def(
            "is_sharded", [](const Tensor& self) { return self.is_sharded(); }, R"doc(
            Check if TT Tensor is sharded.

            .. code-block:: python

                is_sharded = tt_tensor.is_sharded()

        )doc")
        .def(
            "get_dtype", [](const Tensor& self) { return self.dtype(); }, R"doc(
            Get dtype of TT Tensor.

            .. code-block:: python

                dtype = tt_tensor.dtype()
        )doc")
        .def(
            "reshape",
            [](Tensor& self, int N, int C, int H, int W) {
                return ttnn::reshape(self, ttnn::SmallVector<int>{N, C, H, W});
            },
            R"doc(
                Reshapes TT tensor

                .. code-block:: python

                    reshaped_tensor = tt_tensor.reshape(N, C, H, W)
            )doc")
        .def(
            "reshape",
            [](Tensor& self, const ttnn::Shape& shape) -> Tensor { return ttnn::reshape(self, shape); },
            R"doc(
                Reshapes TT tensor

                .. code-block:: python

                    reshaped_tensor = tt_tensor.reshape((4, 3, 32))
            )doc")
        .def(
            "reshape",
            [](Tensor& self, const ttnn::SmallVector<int32_t>& shape) -> Tensor { return ttnn::reshape(self, shape); },
            R"doc(
                Reshapes TT tensor

                .. code-block:: python

                    reshaped_tensor = tt_tensor.reshape((4, -1, 32))
            )doc")
        .def(
            "to_list",
            [](Tensor& self) -> nb::object {
                using namespace tt::tt_metal::tensor_impl;
                return dispatch(self.dtype(), [&]<typename T>() -> nb::object {
                    const auto& logical_shape = self.logical_shape();
                    auto shape_vec = CMAKE_UNIQUE_NAMESPACE::ttnn_shape_to_ndarray(logical_shape);

                    auto compat_dtype = get_dtype_from_ttnn_datatype(self.dtype());

                    // convert bfloat to float32
                    if (compat_dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat)) {
                        compat_dtype.code = static_cast<uint8_t>(nb::dlpack::dtype_code::Float);
                        compat_dtype.bits = 32;
                    }

                    // TODO: move to
                    auto data_vec = [&self]() {
                        if constexpr (
                            std::is_same_v<T, bfloat8_b> || std::is_same_v<T, bfloat4_b> ||
                            std::is_same_v<T, bfloat16>) {
                            return self.to_vector<float>();
                        } else {
                            return self.to_vector<T>();
                        }
                    }();

                    auto builtin = nb::module_::import_("builtins");
                    auto arr = nb::ndarray<nb::array_api>(
                        data_vec.data(),
                        shape_vec.size(),
                        shape_vec.data(),
                        nb::find(&self),
                        nullptr,
                        compat_dtype,
                        nb::device::cpu::value,
                        0,
                        nb::c_contig::value);

                    // builtin memoryview has a tolist, so just call that.
                    return builtin.attr("memoryview")(arr.cast()).attr("tolist")();
                });
            },
            R"doc(
                Return TT tensor values as python list

                .. code-block:: python

                    py_list = tt_tensor.to_list()
            )doc")
        .def(
            "tensor_topology",
            [](const Tensor& self) { return self.tensor_topology(); },
            nb::rv_policy::reference_internal,
            R"doc(
                Get the topology of the tensor.

                .. code-block:: python

                    topology = tt_tensor.tensor_topology()
            )doc")
        .def_prop_rw(
            "tensor_id",
            [](const Tensor& self) { return self.tensor_id; },
            [](Tensor& self, std::size_t tensor_id) { self.tensor_id = tensor_id; });
}

}  // namespace ttnn::tensor
