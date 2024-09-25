// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <memory>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "tensor.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "tt_metal/graph/graph_tracking.hpp"
#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

namespace py = pybind11;

namespace ttnn::tensor {

using tt::tt_metal::CoreCoord;

namespace detail {

#ifdef DEBUG

void log_external_operation(
    std::size_t operation_id,
    std::size_t device_operation_id,
    const operation::ExternalOperation& operation,
    const std::vector<Tensor>& input_tensors) {
    tt::log_debug(tt::LogOp, "Launching External Operation: \"{}\"", operation.get_type_name());

    auto attributes = operation.attributes();
    if (not attributes.empty()) {
        tt::log_debug(tt::LogOp, "Attributes:");
        for (auto&& [name, value] : attributes) {
            tt::log_debug(tt::LogOp, "\t{} = {}", name, value);
        }
    }

    tt::log_debug(tt::LogOp, "Input std::vector<Tensor>:");
    for (auto index = 0; index < input_tensors.size(); index++) {
        const auto& tensor = input_tensors[index];
        tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
    }

    tt::log_debug(tt::LogOp, "");
}
#else

void log_external_operation(
    std::size_t operation_id,
    std::size_t device_operation_id,
    const operation::ExternalOperation& operation,
    const std::vector<Tensor>& input_tensors) {}

#endif

template <typename T>
Tensor create_owned_tensor(T* data_ptr, size_t num_elements, std::vector<uint32_t>& shape, DataType data_type, Layout layout, const std::optional<Tile>& optional_tile = std::nullopt)
{
    auto data = std::vector(data_ptr, data_ptr + num_elements);
    auto buffer = owned_buffer::create(std::move(data));
    auto storage = OwnedStorage{std::move(buffer)};
    return Tensor(std::move(storage), shape, data_type, layout, optional_tile);
}

Tensor convert_torch_tensor_to_tt_tensor(
    const py::handle &torch_tensor, std::optional<DataType> optional_data_type = std::nullopt, const std::optional<Tile>& optional_tile = std::nullopt, bool enable_borrow = true) {
    py::object torch = py::module_::import("torch");
    if (not py::isinstance(torch_tensor, torch.attr("Tensor"))) {
        TT_THROW("The argument must be of type torch.Tensor!");
    }

    auto torch_dtype = torch_tensor.attr("dtype");
    auto shape = py::cast<std::vector<uint32_t>>(torch_tensor.attr("shape"));

    auto contiguous_torch_tensor = torch_tensor.attr("contiguous")();

    // Override the data type if there is an user-provided one
    // Otherwise, figure it out from torch dtype
    DataType data_type;
    if (optional_data_type.has_value()) {
        data_type = optional_data_type.value();
    } else if (torch_dtype.equal(torch.attr("float32"))) {
        data_type = DataType::FLOAT32;
    } else if (torch_dtype.equal(torch.attr("float16"))) {
        contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("bfloat16"));
        // TODO(arakhmati): add DataType::FLOAT16?
        data_type = DataType::BFLOAT16;
    } else if (torch_dtype.equal(torch.attr("bfloat16"))) {
        data_type = DataType::BFLOAT16;
    } else if (torch_dtype.equal(torch.attr("int64"))) {
        contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("int32"));
        // TODO(arakhmati): add DataType::INT64?
        data_type = DataType::UINT32;
    } else if (torch_dtype.equal(torch.attr("int32"))) {
        data_type = DataType::INT32;
    }  else if (torch_dtype.equal(torch.attr("int16"))) {
        // TODO(arakhmati): add DataType::INT16?
        data_type = DataType::UINT16;
    }  else if (torch_dtype.equal(torch.attr("uint8"))) {
        data_type = DataType::UINT8;
    } else {
        TT_THROW("Unsupported DataType: {}", std::string(py::repr(torch_dtype)));
    }

    switch (data_type) {
        case DataType::UINT8: {
            if (not torch_dtype.equal(torch.attr("uint8"))) {
                contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("uint8"));
            }
            break;
        }
        case DataType::UINT16: {
            if (not torch_dtype.equal(torch.attr("int16"))) {
                contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("int16"));
            }
            break;
        }
        case DataType::INT32:
        case DataType::UINT32: {
            if (not torch_dtype.equal(torch.attr("int32"))) {
                contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("int32"));
            }
            break;
        }
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
        case DataType::FLOAT32: {
            if (not torch_dtype.equal(torch.attr("float32"))) {
                contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("float32"));
            }
            break;
        }
        case DataType::BFLOAT16: {
            if (not torch_dtype.equal(torch.attr("bfloat16"))) {
                contiguous_torch_tensor = contiguous_torch_tensor.attr("to")(torch.attr("bfloat16"));
            }
            break;
        }
        default: {
            TT_THROW("Unsupported DataType: {}", data_type);
            break;
        }
    }

    auto on_creation_callback = [tensor = contiguous_torch_tensor] { tensor.inc_ref(); };
    auto on_destruction_callback = [tensor = contiguous_torch_tensor] { tensor.dec_ref(); };

    auto num_elements = py::cast<std::size_t>(contiguous_torch_tensor.attr("numel")());
    auto torch_data_ptr = py::cast<std::size_t>(contiguous_torch_tensor.attr("data_ptr")());
    switch (data_type) {
        case DataType::UINT8: {
            auto data_ptr = reinterpret_cast<uint8_t *>(torch_data_ptr);
            if (enable_borrow) {
                auto storage = BorrowedStorage(
                    borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
                return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
            } else {
                return create_owned_tensor(data_ptr, num_elements, shape, data_type, Layout::ROW_MAJOR, optional_tile);
            }
        }
        case DataType::UINT16: {
            auto data_ptr = reinterpret_cast<uint16_t *>(torch_data_ptr);
            if (enable_borrow) {
                auto storage = BorrowedStorage(
                    borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
                return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
            } else {
                return create_owned_tensor(data_ptr, num_elements, shape, data_type, Layout::ROW_MAJOR, optional_tile);
            }
        }
        case DataType::INT32: {
            auto data_ptr = reinterpret_cast<int32_t *>(torch_data_ptr);
            if (enable_borrow) {
                auto storage = BorrowedStorage(
                    borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
                return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
            } else {
                return create_owned_tensor(data_ptr, num_elements, shape, data_type, Layout::ROW_MAJOR, optional_tile);
            }
        }
        case DataType::UINT32: {
            auto data_ptr = reinterpret_cast<uint32_t *>(torch_data_ptr);
            if (enable_borrow) {
                auto storage = BorrowedStorage(
                    borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
                return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
            } else {
                return create_owned_tensor(data_ptr, num_elements, shape, data_type, Layout::ROW_MAJOR, optional_tile);
            }
        }
        case DataType::FLOAT32: {
            auto data_ptr = reinterpret_cast<float *>(torch_data_ptr);
            if (enable_borrow) {
                auto storage = BorrowedStorage(
                    borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
                return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
            } else {
                return create_owned_tensor(data_ptr, num_elements, shape, data_type, Layout::ROW_MAJOR, optional_tile);
            }
        }
        case DataType::BFLOAT16: {
            auto data_ptr = reinterpret_cast<::bfloat16 *>(torch_data_ptr);
            if (enable_borrow) {
                auto storage = BorrowedStorage(
                    borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
                return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
            } else {
                return create_owned_tensor(data_ptr, num_elements, shape, data_type, Layout::ROW_MAJOR, optional_tile);
            }
        }
        case DataType::BFLOAT8_B: {
            auto data_ptr = reinterpret_cast<float *>(torch_data_ptr);
            auto data = std::vector<float>(data_ptr, data_ptr + num_elements);
            auto uint32_vector = pack_fp32_vec_as_bfp8_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            auto buffer = owned_buffer::create<uint32_t>(std::move(uint32_vector));
            auto storage = OwnedStorage{std::move(buffer)};
            // TODO(arakhmati): should it be Layout::TILE?
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
        }
        case DataType::BFLOAT4_B: {
            auto data_ptr = reinterpret_cast<float *>(torch_data_ptr);
            auto data = std::vector<float>(data_ptr, data_ptr + num_elements);
            auto uint32_vector = pack_fp32_vec_as_bfp4_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            auto buffer = owned_buffer::create<uint32_t>(std::move(uint32_vector));
            auto storage = OwnedStorage{std::move(buffer)};
            // TODO(arakhmati): should it be Layout::TILE?
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
        }
        default: {
            TT_THROW("Unsupported DataType: {}", data_type);
            break;
        }
    }
}

Tensor convert_numpy_tensor_to_tt_tensor(
    const py::handle &np_tensor, std::optional<DataType> optional_data_type = std::nullopt, const std::optional<Tile>& optional_tile = std::nullopt) {
    py::object np = py::module_::import("numpy");
    if (not py::isinstance(np_tensor, np.attr("ndarray"))) {
        TT_THROW("The tensor must be of type numpy.ndarray!");
    }

    auto np_dtype = np_tensor.attr("dtype");
    auto shape = py::cast<std::vector<uint32_t>>(np_tensor.attr("shape"));

    auto contiguous_np_tensor = np.attr("ascontiguousarray")(np_tensor);

    // Override the data type if there is an user-provided one
    // Otherwise, figure it out from numpy dtype
    DataType data_type;
    if (optional_data_type.has_value()) {
        data_type = optional_data_type.value();
    } else if (np_dtype.equal(np.attr("float32"))) {
        data_type = DataType::FLOAT32;
    } else if (np_dtype.equal(np.attr("float16"))) {
        contiguous_np_tensor = contiguous_np_tensor.attr("astype")(np.attr("float32"));
        // TODO(arakhmati): add DataType::FLOAT16?
        data_type = DataType::BFLOAT16;
    } else if (np_dtype.equal(np.attr("int64"))) {
        contiguous_np_tensor = contiguous_np_tensor.attr("astype")(np.attr("int32"));
        // TODO(arakhmati): add DataType::INT64?
        data_type = DataType::UINT32;
    } else if (np_dtype.equal(np.attr("int32"))) {
        // TODO(arakhmati): add DataType::INT32?
        data_type = DataType::UINT32;
    } else if (np_dtype.equal(np.attr("ubyte"))) {
        data_type = DataType::UINT8;
    } else {
        TT_THROW("Unsupported DataType: {}", std::string(py::repr(np_dtype)));
    }

    switch (data_type) {
        case DataType::UINT8: {
            if (not np_dtype.equal(np.attr("ubyte"))) {
                contiguous_np_tensor = contiguous_np_tensor.attr("astype")(np.attr("ubyte"));
            }
            break;
        }
        case DataType::UINT16: {
            if (not np_dtype.equal(np.attr("int32"))) {
                contiguous_np_tensor = contiguous_np_tensor.attr("astype")(np.attr("int16"));
            }
            break;
        }
        case DataType::INT32:
        case DataType::UINT32: {
            if (not np_dtype.equal(np.attr("int32"))) {
                contiguous_np_tensor = contiguous_np_tensor.attr("astype")(np.attr("int32"));
            }
            break;
        }
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
        case DataType::FLOAT32: {
            if (not np_dtype.equal(np.attr("float32"))) {
                contiguous_np_tensor = contiguous_np_tensor.attr("astype")(np.attr("float32"));
            }
            break;
        }
        /*
        case DataType::BFLOAT16: {
            if (not np_dtype.equal(np.attr("bfloat16"))) {
                contiguous_np_tensor = contiguous_np_tensor.attr("to")(np.attr("bfloat16"));
            }
            break;
        }
        */
        default: {
            TT_THROW("Unsupported DataType: {}", data_type);
            break;
        }
    }

    auto on_creation_callback = [tensor = contiguous_np_tensor] { tensor.inc_ref(); };
    auto on_destruction_callback = [tensor = contiguous_np_tensor] { tensor.dec_ref(); };

    auto num_elements = py::cast<std::size_t>(contiguous_np_tensor.attr("size"));
    auto np_data_ptr = py::cast<std::size_t>(
        py::cast<py::tuple>(py::cast<py::dict>(contiguous_np_tensor.attr("__array_interface__"))[py::str("data")])[0]);

    switch (data_type) {
        case DataType::UINT8: {
            auto data_ptr = reinterpret_cast<uint8_t *>(np_data_ptr);
            auto storage = BorrowedStorage(
                borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
        }
        case DataType::UINT16: {
            auto data_ptr = reinterpret_cast<uint16_t *>(np_data_ptr);
            auto storage = BorrowedStorage(
                borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
        }
        case DataType::INT32: {
            auto data_ptr = reinterpret_cast<int32_t *>(np_data_ptr);
            auto storage = BorrowedStorage(
                borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
        }
        case DataType::UINT32: {
            auto data_ptr = reinterpret_cast<uint32_t *>(np_data_ptr);
            auto storage = BorrowedStorage(
                borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
        }
        case DataType::FLOAT32: {
            auto data_ptr = reinterpret_cast<float *>(np_data_ptr);
            auto storage = BorrowedStorage(
                borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
        }
        /*
        case DataType::BFLOAT16: {
            auto data_ptr = reinterpret_cast<bfloat16 *>(np_data_ptr);
            auto storage = BorrowedStorage(
                borrowed_buffer::Buffer(data_ptr, num_elements), on_creation_callback, on_destruction_callback);
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR);
        }
        */
        case DataType::BFLOAT8_B: {
            auto data_ptr = reinterpret_cast<float *>(np_data_ptr);
            auto data = std::vector<float>(data_ptr, data_ptr + num_elements);
            auto uint32_vector = pack_fp32_vec_as_bfp8_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            auto buffer = owned_buffer::create<uint32_t>(std::move(uint32_vector));
            auto storage = OwnedStorage{std::move(buffer)};
            // TODO(arakhmati): should it be Layout::TILE?
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
        }
        case DataType::BFLOAT4_B: {
            auto data_ptr = reinterpret_cast<float *>(np_data_ptr);
            auto data = std::vector<float>(data_ptr, data_ptr + num_elements);
            auto uint32_vector = pack_fp32_vec_as_bfp4_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            auto buffer = owned_buffer::create<uint32_t>(std::move(uint32_vector));
            auto storage = OwnedStorage{std::move(buffer)};
            // TODO(arakhmati): should it be Layout::TILE?
            return Tensor(std::move(storage), shape, data_type, Layout::ROW_MAJOR, optional_tile);
        }
        default: {
            TT_THROW("Unsupported DataType: {}", data_type);
            break;
        }
    }
}

Tensor convert_python_tensor_to_tt_tensor(
    const py::handle &tensor, std::optional<DataType> optional_data_type = std::nullopt, const std::optional<Tile>& optional_tile = std::nullopt, bool enable_borrow = true) {
    GraphTracker::instance().track_function_start("tt::tt_metal::detail::convert_python_tensor_to_tt_tensor", tensor, optional_data_type, enable_borrow);
    py::object torch = py::module_::import("torch");
    py::object np = py::module_::import("numpy");
    if (py::isinstance(tensor, torch.attr("Tensor"))) {
        auto output = convert_torch_tensor_to_tt_tensor(tensor, optional_data_type, optional_tile, enable_borrow);
        output = tt::tt_metal::set_tensor_id(output);
        GraphTracker::instance().track_function_end(output);
        return output;
    } else if (py::isinstance(tensor, np.attr("ndarray"))) {
        auto output = convert_numpy_tensor_to_tt_tensor(tensor, optional_data_type, optional_tile);
        output = tt::tt_metal::set_tensor_id(output);
        GraphTracker::instance().track_function_end(output);
        return output;
    } else {
        TT_THROW("The argument must be of type torch.Tensor or numpy.ndarray!");
    }
}

Tensor convert_python_tensors_to_tt_tensors(py::list tensor_shards, std::optional<DataType> data_type, const  std::optional<Tile> tile, const std::unordered_map<std::string, std::string>& strategy) {
    GraphTracker::instance().track_function_start("tt::tt_metal::detail::convert_python_tensors_to_tt_tensors", tensor_shards, data_type, strategy);
    std::vector<Tensor> tt_shards;
    for (const auto &shard : tensor_shards) {
        tt_shards.push_back(detail::convert_python_tensor_to_tt_tensor(shard, data_type, tile, false));
    }
    std::vector<OwnedBuffer> host_owned_buffers;
    std::vector<ttnn::Shape> host_owned_shapes;
    for (const auto &shard : tt_shards) {
        TT_ASSERT(std::holds_alternative<OwnedStorage>(shard.get_storage()), "Unexpected type {}", tt::stl::get_active_type_name_in_variant(shard.get_storage()));
        host_owned_buffers.push_back(std::get<OwnedStorage>(shard.get_storage()).buffer);
        host_owned_shapes.push_back(shard.get_shape());
    }
    auto distributed_tensor_config = get_distributed_tensor_config(strategy);
    auto storage = MultiDeviceHostStorage{distributed_tensor_config, std::move(host_owned_buffers), host_owned_shapes};

    auto output = Tensor(std::move(storage), tt_shards.at(0).get_shape(), tt_shards.at(0).get_dtype(), Layout::ROW_MAJOR, tt_shards.at(0).get_tile());
    output = tt::tt_metal::set_tensor_id(output);
    GraphTracker::instance().track_function_end(output);
    return output;
}

    OwnedBuffer create_owned_buffer_from_vector_of_floats(std::vector<float>&& data, DataType data_type) {
        switch (data_type) {
            case DataType::BFLOAT8_B: {
                auto uint32_vector = pack_fp32_vec_as_bfp8_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
                return owned_buffer::create<uint32_t>(std::move(uint32_vector));
            }
            case DataType::BFLOAT4_B: {
                auto uint32_vector = pack_fp32_vec_as_bfp4_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
                return owned_buffer::create<uint32_t>(std::move(uint32_vector));
            }
            case DataType::FLOAT32: {
                return owned_buffer::create<float>(std::move(data));
            }
            case DataType::BFLOAT16: {
                std::vector<::bfloat16> bfloat16_data(data.size());
                std::transform(
                    std::begin(data), std::end(data),
                    std::begin(bfloat16_data),
                    [](float value) { return ::bfloat16(value); }
                );
                return owned_buffer::create<::bfloat16>(std::move(bfloat16_data));
            }
            default: {
                TT_THROW("Cannot create a host buffer!");
            }
        }
    }

    py::object convert_tt_tensor_to_torch_tensor(const Tensor& tt_tensor) {
        GraphTracker::instance().track_function_start("tt::tt_metal::detail::convert_tt_tensor_to_torch_tensor", tt_tensor);
        TT_ASSERT(tt_tensor.storage_type() == StorageType::OWNED or tt_tensor.storage_type() == StorageType::BORROWED);

        using namespace pybind11::literals;
        py::object torch = py::module_::import("torch");
        auto frombuffer = torch.attr("frombuffer");
        auto buffer = std::visit(
            [](auto &&storage) -> std::variant<OwnedBuffer, BorrowedBuffer> {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, OwnedStorage>) {
                    return storage.buffer;
                }
                else if constexpr (std::is_same_v<T, DeviceStorage>) {
                    TT_THROW("Device tensor cannot be converted to torch");
                } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                    return storage.buffer;
                } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                    TT_THROW("Tensor with MultiDeviceStorage cannot be converted to torch");
                } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                    TT_THROW("Tensor MultiDeviceHostStorage cannot be converted to torch directly. Use composer(..) functionality.");
                } else {
                    raise_unsupported_storage<T>();
                }
            },
            tt_tensor.get_storage());

        auto tt_dtype = tt_tensor.get_dtype();
        if (tt_dtype == DataType::BFLOAT8_B) {
            TT_ASSERT(std::holds_alternative<OwnedBuffer>(buffer), "Unexpected type {}", tt::stl::get_active_type_name_in_variant(buffer));
            auto uint32_data = std::get<owned_buffer::Buffer<std::uint32_t>>(std::get<OwnedBuffer>(buffer)).get();
            auto float_unpacked_data = unpack_bfp8_tiles_into_float_vec(uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false);
            buffer = owned_buffer::create<float>(std::move(float_unpacked_data));
            tt_dtype = DataType::FLOAT32;
        }
        if (tt_dtype == DataType::BFLOAT4_B) {
            TT_ASSERT(std::holds_alternative<OwnedBuffer>(buffer), "Unexpected type {}", tt::stl::get_active_type_name_in_variant(buffer));
            auto uint32_data = std::get<owned_buffer::Buffer<std::uint32_t>>(std::get<OwnedBuffer>(buffer)).get();
            auto float_unpacked_data = unpack_bfp4_tiles_into_float_vec(uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false);
            buffer = owned_buffer::create<float>(std::move(float_unpacked_data));
            tt_dtype = DataType::FLOAT32;
        }

        const auto tt_dtype_to_torch_dtype = std::map<DataType, py::object> {
            {DataType::UINT8, torch.attr("uint8")},
            {DataType::UINT16, torch.attr("int16")}, // TODO(arakhmati): add DataType::INT16
            {DataType::INT32, torch.attr("int32")},
            {DataType::UINT32, torch.attr("int32")}, // TODO(arakhmati): add DataType::INT32
            {DataType::FLOAT32, torch.attr("float32")},
            {DataType::BFLOAT16, torch.attr("bfloat16")},
        };
        auto torch_dtype = tt_dtype_to_torch_dtype.at(tt_dtype);

        auto shape = tt_tensor.get_shape().value;
        auto torch_shape = std::vector<std::uint32_t>(std::begin(shape), std::end(shape));
        auto tensor = frombuffer(buffer, "dtype"_a=torch_dtype);
        tensor = tensor.attr("reshape")(torch_shape);
        tensor = tensor.attr("contiguous")();
        if (tt_tensor.storage_type() == StorageType::BORROWED) {
            tensor = tensor.attr("clone")();
        }
        GraphTracker::instance().track_function_end(tensor);
        return tensor;
    }

    py::object convert_tt_tensor_to_numpy_tensor(const Tensor &tt_tensor) {
        GraphTracker::instance().track_function_start("tt::tt_metal::detail::convert_tt_tensor_to_torch_tensor", tt_tensor);
        TT_ASSERT(tt_tensor.storage_type() == StorageType::OWNED or tt_tensor.storage_type() == StorageType::BORROWED);

        using namespace pybind11::literals;
        py::object np = py::module_::import("numpy");
        auto frombuffer = np.attr("frombuffer");

        auto buffer = std::visit(
            [](auto &&storage) -> std::variant<OwnedBuffer, BorrowedBuffer> {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, OwnedStorage>) {
                    return storage.buffer;
                } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                    TT_THROW("Device tensor cannot be converted to numpy");
                } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                    return storage.buffer;
                } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                    TT_THROW("Device tensor cannot be converted to numpy");
                } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                    TT_THROW("Device tensor cannot be converted to torch");
                } else {
                    raise_unsupported_storage<T>();
                }
            },
            tt_tensor.get_storage());

        auto tt_dtype = tt_tensor.get_dtype();
        if (tt_dtype == DataType::BFLOAT8_B) {
            TT_ASSERT(std::holds_alternative<OwnedBuffer>(buffer), "Unexpected type {}", tt::stl::get_active_type_name_in_variant(buffer));
            auto uint32_data = std::get<owned_buffer::Buffer<std::uint32_t>>(std::get<OwnedBuffer>(buffer)).get();
            auto float_unpacked_data =
                unpack_bfp8_tiles_into_float_vec(uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false);
            buffer = owned_buffer::create<float>(std::move(float_unpacked_data));
            tt_dtype = DataType::FLOAT32;
        }
        if (tt_dtype == DataType::BFLOAT4_B) {
            TT_ASSERT(std::holds_alternative<OwnedBuffer>(buffer), "Unexpected type {}", tt::stl::get_active_type_name_in_variant(buffer));
            auto uint32_data = std::get<owned_buffer::Buffer<std::uint32_t>>(std::get<OwnedBuffer>(buffer)).get();
            auto float_unpacked_data =
                unpack_bfp4_tiles_into_float_vec(uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false);
            buffer = owned_buffer::create<float>(std::move(float_unpacked_data));
            tt_dtype = DataType::FLOAT32;
        }

        const auto tt_dtype_to_np_dtype = std::map<DataType, py::object>{
            {DataType::UINT8, np.attr("ubyte")},
            {DataType::UINT16, np.attr("int16")},  // TODO(arakhmati): add DataType::INT16
            {DataType::INT32, np.attr("int32")},
            {DataType::UINT32, np.attr("int32")},  // TODO(arakhmati): add DataType::INT32
            {DataType::FLOAT32, np.attr("float32")},
        };
        auto np_dtype = tt_dtype_to_np_dtype.at(tt_dtype);

        auto shape = tt_tensor.get_shape().with_tile_padding();
        auto np_shape = std::vector<std::uint32_t>(std::begin(shape), std::end(shape));
        auto tensor = frombuffer(buffer, "dtype"_a = np_dtype);
        tensor = tensor.attr("reshape")(np_shape);
        tensor = np.attr("ascontiguousarray")(tensor);
        GraphTracker::instance().track_function_end(tensor);
        return tensor;
    }

    auto parse_external_operation(
        const py::function &external_operation,
        const py::args &args,
        const py::kwargs &kwargs,
        std::optional<std::string> function_name_override = std::nullopt) {
        std::string function_name;
        if (function_name_override.has_value()) {
            function_name = function_name_override.value();
        } else {
            function_name = py::cast<std::string>(external_operation.attr("__qualname__"));
        }

        std::vector<Tensor> input_tensors;
        tt::stl::reflection::Attributes attributes;

        auto process_name_and_value = [&function_name, &input_tensors, &attributes](
                                          const auto &name, const auto &value) {
            py::object torch = py::module_::import("torch");
            py::object ttnn = py::module_::import("ttnn");
            if (py::isinstance<Tensor>(value)) {
                // TODO(arakhmati): figure out how to handle this without causing extra memory usage
                // auto tensor = py::cast<Tensor>(value);
                // input_tensors.push_back(tensor);
            } else if (py::isinstance(value, ttnn.attr("Tensor"))) {
                // TODO(arakhmati): figure out how to handle this without causing extra memory usage
                // auto tensor = py::cast<Tensor>(value.attr("value"));
                // input_tensors.push_back(tensor);
            } else if (py::isinstance(value, torch.attr("nn").attr("Module"))) {
                // do nothing
            } else if (py::isinstance(value, torch.attr("Tensor"))) {
                // TODO(arakhmati): figure out how to handle this without causing extra memory usage
                // auto tensor = detail::convert_torch_tensor_to_tt_tensor(value);
                // input_tensors.push_back(tensor);
            } else {
                // TODO(MO): Exclude tensor data as it is not an attribute
                //attributes.push_back({name, fmt::format("{}", value)});
            }
        };

        auto arg_index = 0;
        for (const auto &value : args) {
            auto name = fmt::format("arg_{}", arg_index++);
            process_name_and_value(name, value);
        }

        for (const auto &[name, value] : kwargs) {
            process_name_and_value(py::cast<std::string>(name), value);
        }

        auto operation = tt::tt_metal::operation::ExternalOperation{function_name, attributes};
        return std::make_tuple(operation, input_tensors);
    }

} // namespace detail

void pytensor_module_types(py::module &m_tensor) {
    // Tensor constructors that accept device and .to(device) function use keep alive call policy to communicate that Device needs to outlive Tensor.
    // This is because when tensors on device are destroyed they need to deallocate their buffers via device.
    // keep_alive increases the ref count of the Device object being passed into the constructor and .to() function.
    // For additional info see: https://pybind11.readthedocs.io/en/stable/advanced/functions.html#keep-alive
    auto pyTensor = py::class_<Tensor>(m_tensor, "Tensor", R"doc(

        Class constructor supports tensors of rank 4.
        The constructor takes following arguments:

        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        |  Argument  |                 Description                            |       Data type           |           Valid range              | Required |
        +============+========================================================+===========================+====================================+==========+
        | data       | Data to store in TT tensor                             | List[float/int]           |                                    | Yes      |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | shape      | Shape of TT tensor                                     | List[int[4]]              |                                    | Yes      |
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

void pytensor_module(py::module &m_tensor) {
    m_tensor.def(
        "decorate_external_operation",
        [](const py::function &function, std::optional<std::string> function_name) -> py::function {
            return py::cpp_function(std::function([function, function_name](
                                                        const py::args &args, const py::kwargs &kwargs) {
                ZoneScopedN("TT_DNN_FALLBACK_OP");
                uint32_t device_operation_id = ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id();
                auto [operation, input_tensors] = detail::parse_external_operation(function, args, kwargs, function_name);
                GraphTracker::instance().track_function_start(operation.get_type_name(), args, kwargs);
                detail::log_external_operation(ttnn::CoreIDs::instance().get_python_operation_id(), device_operation_id, operation, input_tensors);

                auto output = function(*args, **kwargs);

                TracyOpTTNNExternal(device_operation_id, operation, input_tensors);
                GraphTracker::instance().track_function_end(output);
                return output;
            }));
        },
        py::arg("function").noconvert(),
        py::arg("function_name").noconvert() = std::nullopt,
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

    auto pyTensor = static_cast<py::class_<Tensor>>(m_tensor.attr("Tensor"));
    pyTensor.def(py::init<ttnn::Tensor &>())
        .def(
            py::init<>([](std::vector<float> &&data,
                            const std::array<uint32_t, 4> &shape,
                            DataType data_type,
                            Layout layout,
                            const std::optional<Tile> &tile) {
                auto owned_buffer = detail::create_owned_buffer_from_vector_of_floats(std::move(data), data_type);
                return Tensor(OwnedStorage{owned_buffer}, ttnn::Shape(shape), data_type, layout, tile);
            }),
            py::arg("data"),
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("layout"),
            py::arg("tile") = std::nullopt,
            py::return_value_policy::move,
            R"doc(
                +---------------+---------------+
                | Argument      | Name          |
                +===============+===============+
                | arg0          | data          |
                +---------------+---------------+
                | arg1          | shape         |
                +---------------+---------------+
                | arg2          | data_type     |
                +---------------+---------------+
                | arg3          | layout        |
                +---------------+---------------+

                Example of creating a TT Tensor on host:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    ttnn.Tensor(
                        py_tensor.reshape(-1).tolist(),
                        py_tensor.size(),
                        ttnn.DataType.BFLOAT16,
                        ttnn.Layout.ROW_MAJOR,
                    )
            )doc")
        .def(
            py::init<>([](std::vector<float> &&data,
                            const std::array<uint32_t, 4> &shape,
                            DataType data_type,
                            Layout layout,
                            Device *device,
                            const std::optional<Tile> &tile) {
                auto owned_buffer = detail::create_owned_buffer_from_vector_of_floats(std::move(data), data_type);
                auto tensor = Tensor(OwnedStorage{owned_buffer}, ttnn::Shape(shape), data_type, layout, tile);
                return tensor.to(device, MemoryConfig{});
            }),
            py::keep_alive<1, 6>(),
            py::arg("data"),
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("layout"),
            py::arg("device") = std::nullopt,
            py::arg("tile") = std::nullopt,
            py::return_value_policy::move,
            R"doc(
                +---------------+---------------+
                | Argument      | Name          |
                +===============+===============+
                | arg0          | data          |
                +---------------+---------------+
                | arg1          | shape         |
                +---------------+---------------+
                | arg2          | data_type     |
                +---------------+---------------+
                | arg3          | layout        |
                +---------------+---------------+
                | arg3          | device        |
                +---------------+---------------+

                Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

                Note that TT Tensor in ROW_MAJOR layout on TT Accelerator device must have size of last dimension divisble by 2.

                Example of creating a TT Tensor on TT accelerator device:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    tt_device = ttnn.CreateDevice(0)
                    // ...
                    ttnn.Tensor(
                        py_tensor.reshape(-1).tolist(),
                        py_tensor.size(),
                        ttnn.DataType.BFLOAT16,
                        ttnn.Layout.ROW_MAJOR,
                        tt_device
                    )
            )doc")
        .def(
            py::init<>([](std::vector<float> &&data,
                            const std::array<uint32_t, 4> &shape,
                            DataType data_type,
                            Layout layout,
                            Device *device,
                            const MemoryConfig &memory_config,
                            const std::optional<Tile> &tile) {
                auto owned_buffer = detail::create_owned_buffer_from_vector_of_floats(std::move(data), data_type);
                auto tensor = Tensor(OwnedStorage{owned_buffer}, ttnn::Shape(shape), data_type, layout, tile);
                return tensor.to(device, memory_config);
            }),
            py::keep_alive<1, 7>(),
            py::arg("data"),
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("layout"),
            py::arg("device") = std::nullopt,
            py::arg("memory_config"),
            py::arg("tile") = std::nullopt,
            py::return_value_policy::move,
            R"doc(
                +---------------+---------------+
                | Argument      | Name          |
                +===============+===============+
                | arg0          | data          |
                +---------------+---------------+
                | arg1          | shape         |
                +---------------+---------------+
                | arg2          | data_type     |
                +---------------+---------------+
                | arg3          | layout        |
                +---------------+---------------+
                | arg4          | device        |
                +---------------+---------------+
                | arg5          | mem_config    |
                +---------------+---------------+

                Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

                Note that TT Tensor in ROW_MAJOR layout on TT Accelerator device must have size of last dimension divisble by 2.

                Example of creating a TT Tensor on TT accelerator device with specified mem_config:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    tt_device = ttnn.CreateDevice(0)
                    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.SINGLE_BANK)
                    // ...
                    ttnn.Tensor(
                        py_tensor.reshape(-1).tolist(),
                        py_tensor.size(),
                        ttnn.DataType.BFLOAT16,
                        ttnn.Layout.ROW_MAJOR,
                        tt_device,
                        mem_config
                    )
            )doc")
        .def(
            py::init<>([](const py::object &tensor,
                            std::optional<DataType> data_type,
                            const std::unordered_map<std::string, std::string> &strategy,
                            const std::optional<Tile> &tile) {
                if (py::isinstance<py::list>(tensor)) {
                    return detail::convert_python_tensors_to_tt_tensors(tensor, data_type, tile, strategy);
                }
                return detail::convert_python_tensor_to_tt_tensor(tensor, data_type, tile);
            }),
            py::arg("tensor"),
            py::arg("data_type") = std::nullopt,
            py::arg("strategy") = std::unordered_map<std::string, std::string>(),
            py::arg("tile") = std::nullopt,
            py::return_value_policy::move,
            R"doc(
                +--------------+------------------------+
                | Argument     | Description            |
                +==============+========================+
                | tensor       | Pytorch or Numpy Tensor|
                +--------------+------------------------+
                | data_type    | TT Tensor data type    |
                +--------------+------------------------+

                Example of creating a TT Tensor that uses torch.Tensor's storage as its own storage:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    ttnn.Tensor(py_tensor)
            )doc")
        .def(
            py::init<>([](const py::object &python_tensor,
                            std::optional<DataType> data_type,
                            Device *device,
                            Layout layout,
                            const MemoryConfig &mem_config,
                            const std::optional<Tile> &tile) {
                auto tensor = detail::convert_python_tensor_to_tt_tensor(python_tensor, data_type, tile);
                auto layout_tensor = tensor.to(layout);
                return layout_tensor.to(device, mem_config);
            }),
            py::arg("tensor"),
            py::arg("data_type") = std::nullopt,
            py::arg("device").noconvert(),
            py::arg("layout").noconvert(),
            py::arg("mem_config").noconvert(),
            py::arg("tile") = std::nullopt,
            py::return_value_policy::move,
            R"doc(
                +--------------+------------------------+
                | Argument     | Description            |
                +==============+========================+
                | tensor       | Pytorch or Numpy Tensor|
                +--------------+------------------------+
                | data_type    | TT Tensor data type    |
                +--------------+------------------------+
                | device       | TT device ptr          |
                +--------------+------------------------+
                | layout       | TT layout              |
                +--------------+------------------------+
                | mem_config   | TT memory_config       |
                +--------------+------------------------+


                Example of creating a TT Tensor that uses torch.Tensor's storage as its own storage:

                .. code-block:: python

                    py_tensor = np.zeros((1, 1, 32, 32))
                    ttnn.Tensor(py_tensor)
            )doc")
        .def_property_readonly("shape", [](const Tensor &self) { return self.get_shape(); })
        .def_property_readonly("dtype", [](const Tensor &self) { return self.get_dtype(); })
        .def_property_readonly("layout", [](const Tensor &self) { return self.get_layout(); })
        .def_property_readonly("tile", [](const Tensor &self) { return self.get_tile(); })
        .def(
            "deallocate",
            [](Tensor &self, bool force) { return self.deallocate(force); },
            py::arg("force") = false,
            R"doc(
                Dellocates all data of a tensor. This either deletes all host data or deallocates tensor data from device memory.
            )doc")
        .def(
            "to",
            py::overload_cast<Device *, const MemoryConfig &>(&Tensor::to, py::const_),
            py::arg("device").noconvert(),
            py::arg("mem_config").noconvert() = MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED},
            py::keep_alive<0, 2>(),
            R"doc(
            Move TT Tensor from host device to TT accelerator device.

            Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

            If ``arg1`` is not supplied, default ``MemoryConfig`` with ``interleaved`` set to ``True``.

            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range           | Required |
            +===========+=================================================+============================+=======================+==========+
            | arg0      | Device to which tensor will be moved            | ttnn.Device                | TT accelerator device | Yes      |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | arg1      | MemoryConfig of tensor of TT accelerator device | ttnn.MemoryConfig          |                       | No       |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_device)
        )doc")
        .def(
            "track_ref_count",
            [](Tensor &self) { return self.track_ref_count(); },
            R"doc(
                Log the reference count (as seen by the main and worker threads) of a tensor as it evolves during runtime.
            )doc")
        .def(
            "to",
            py::overload_cast<MeshDevice *, const MemoryConfig &>(&Tensor::to, py::const_),
            py::arg("mesh_device").noconvert(),
            py::arg("mem_config").noconvert() = MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED},
            py::keep_alive<0, 2>(),
            R"doc(
            Move TT Tensor from host device to TT accelerator device.

            Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

            If ``arg1`` is not supplied, default ``MemoryConfig`` with ``interleaved`` set to ``True``.

            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range           | Required |
            +===========+=================================================+============================+=======================+==========+
            | arg0      | MeshDevice to which tensor will be moved        | ttnn.MeshDevice            | TT accelerator device | Yes      |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | arg1      | MemoryConfig of tensor of TT accelerator device | ttnn.MemoryConfig          |                       | No       |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_device)
        )doc")
        .def("sync", [](Tensor &self) { return self.wait_for_tensor_data_populated(); })
        .def(
            "extract_shard",
            [](const Tensor &self, CoreCoord core) { return self.extract_shard(core); },
            py::arg("core").noconvert(),
            py::keep_alive<0, 2>(),
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
            [](const Tensor &self, const uint32_t &core_id) { return self.extract_shard(core_id); },
            py::arg("core_id").noconvert(),
            py::keep_alive<0, 2>(),
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
            [](const Tensor &self, bool blocking, uint8_t cq_id) { return self.cpu(blocking, cq_id); },
            py::arg("blocking") = true,
            py::arg("cq_id") = ttnn::DefaultQueueId,
            R"doc(
            Move TT Tensor from TT accelerator device to host device.

            .. code-block:: python

                tt_tensor = tt_tensor.cpu()
        )doc")
        .def("cpu_sharded", &Tensor::cpu_sharded, R"doc(
            Move TT Tensor from TT accelerator device to host device in sharded orientation.

            .. code-block:: python

                tt_tensor = tt_tensor.cpu_sharded()
        )doc")
        .def(
            "to",
            py::overload_cast<Layout, Device *>(&Tensor::to, py::const_),
            py::arg("target_layout").noconvert(),
            py::arg("worker") = nullptr,
            R"doc(
            Convert TT Tensor to provided memory layout. Available layouts conversions are:

            * ROW_MAJOR to TILE
            * TILE to ROW_MAJOR

            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range                    | Required |
            +===========+=================================================+============================+================================+==========+
            | arg0      | Target memory layout                            | ttnn.Layout                | ROW_MAJOR, TILE                | Yes      |
            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+
            | arg1      | Worker thread performing layout conversion      | ttnn.Device                | Thread tied to TT accelerator  | No       |
            |           | (optional)                                      |                            | device                         |          |
            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(ttnn.Layout.TILE, worker)
        )doc")
        .def(
            "to",
            py::overload_cast<Layout, MeshDevice*>(&Tensor::to, py::const_),
            py::arg("target_layout").noconvert(),
            py::arg("mesh_device") = nullptr,
            R"doc(
            Convert TT Tensor to provided memory layout. Available layouts conversions are:

            * ROW_MAJOR to TILE
            * TILE to ROW_MAJOR

            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range                    | Required |
            +===========+=================================================+============================+================================+==========+
            | arg0      | Target memory layout                            | ttnn.Layout                | ROW_MAJOR, TILE                | Yes      |
            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+
            | arg1      | Worker thread performing layout conversion      | ttnn.Device                | Thread tied to TT accelerator  | No       |
            |           | (optional)                                      |                            | device                         |          |
            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(ttnn.Layout.TILE, mesh_device)
        )doc")
        .def(
            "pad",
            [](const Tensor &self,
                const std::array<uint32_t, 4> &output_tensor_shape,
                const std::array<uint32_t, 4> &input_tensor_start,
                float pad_value) { return self.pad(ttnn::Shape(output_tensor_shape), ttnn::Shape(input_tensor_start), pad_value); },
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
            [](const Tensor &self,
                const std::array<uint32_t, 4> &output_tensor_start,
                const std::array<uint32_t, 4> &output_tensor_end) {
                return self.unpad(ttnn::Shape(output_tensor_start), ttnn::Shape(output_tensor_end));
            },
            R"doc(
            Unpad this TT Tensor.

            This tensor must be on host and in ROW_MAJOR layout.

            Returns an output tensor from output tensor start indices ``arg0`` to output tensor end indices ``arg1`` (inclusive) of the input tensor.

            +---------------------+----------------------------------------------+--------------+-----------------------------------------------------+----------+
            | Argument            | Description                                  | Data type    | Valid range                                         | Required |
            +=====================+==============================================+==============+=====================================================+==========+
            | arg0                | Start indices of input tensor                | List[int[4]] | Values along each dim must be                       | Yes      |
            |                     |                                              |              |                                                     |          |
            |                     |                                              |              | < input_tensor_shape[i] and <= output_tensor_end[i] |          |
            +---------------------+----------------------------------------------+--------------+-----------------------------------------------------+----------+
            | arg1                | End indices of input tensor in output tensor | List[int[4]] | Values along each dim must be                       | Yes      |
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
            "pad_to_tile", [](const Tensor &self, float pad_value) { return self.pad_to_tile(pad_value); }, R"doc(
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
            [](const Tensor &self, const std::vector<uint32_t> &output_tensor_shape) {
                return self.unpad_from_tile(output_tensor_shape);
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
            "__repr__", [](const Tensor &self) { return self.write_to_string(); }, R"doc(
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
            "volume", [](const Tensor &self) { return self.volume(); }, R"doc(
            Get the volume of the tensor.

            .. code-block:: python

                volume = tt_tensor.volume()

        )doc")
        .def(
            "storage_type", [](const Tensor &self) { return self.storage_type(); }, R"doc(
            Check if the tensor is on host

            .. code-block:: python

                storage_type = tt_tensor.storage_type()

        )doc")
    .def(
        "device",
        [](const Tensor &self) { return self.device(); },
        R"doc(
            Get the device of the tensor.

            .. code-block:: python

                device = tt_tensor.device()

        )doc",
        py::return_value_policy::reference)
    .def(
        "devices",
        [](const Tensor &self) { return self.get_workers(); },
        R"doc(
            Get devices tensor is mapped on to.

            .. code-block:: python

                devices = tt_tensor.devices()

        )doc",
        py::return_value_policy::reference)
        .def(
            "to_torch",
            [](const Tensor &self) -> py::object { return detail::convert_tt_tensor_to_torch_tensor(self); },
            R"doc(
            Convert tensor to torch tensor.

            The tensor must be on host when calling this function.

            .. code-block:: python

                data = tt_tensor.cpu().to_torch() # move TT Tensor to host and convert it to torch tensor

        )doc")
        .def(
            "to_numpy",
            [](const Tensor &self) -> py::object { return detail::convert_tt_tensor_to_numpy_tensor(self); },
            R"doc(
            Convert tensor to numpy tensor.

            The tensor must be on host when calling this function.

            .. code-block:: python

                data = tt_tensor.cpu().to_numpy() # move TT Tensor to host and convert it to numpy tensor

        )doc")
        .def(
            "buffer",
            [](const Tensor &self) -> std::variant<OwnedBuffer, BorrowedBuffer> {
                return std::visit(
                    [](auto &&storage) -> std::variant<OwnedBuffer, BorrowedBuffer> {
                        using T = std::decay_t<decltype(storage)>;
                        if constexpr (std::is_same_v<T, OwnedStorage>) {
                            return storage.buffer;
                        } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                            TT_THROW("Device storage doesn't support buffer method");
                        } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                            return storage.buffer;
                        } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                            TT_THROW("MultiDeviceStorage doesn't support buffer method");
                        } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                            TT_THROW("MultiDeviceHostStorage doesn't support buffer method");
                        } else {
                            raise_unsupported_storage<T>();
                        }
                    },
                    self.get_storage());
            },
            R"doc(
            Get the underlying buffer.

            The tensor must be on the cpu when calling this function.

            .. code-block:: python

                buffer = tt_tensor.cpu().buffer() # move TT Tensor to host and get the buffer

        )doc")
        .def(
            "buffer_address",
            [](const Tensor &self) -> uint32_t {
                return std::visit(
                    [](auto &&storage) -> uint32_t {
                        using T = std::decay_t<decltype(storage)>;
                        if constexpr (std::is_same_v<T, OwnedStorage>) {
                            TT_THROW("OwnedStorage doesn't support buffer_address method");
                        } else if constexpr (std::is_same_v<T, DeviceStorage>) {
                            return storage.buffer->address();
                        } else if constexpr (std::is_same_v<T, BorrowedStorage>) {
                            TT_THROW("BorrowedStorage doesn't support buffer_address method");
                        } else if constexpr (std::is_same_v<T, MultiDeviceStorage>) {
                            TT_THROW("MultiDeviceStorage doesn't support buffer_address method");
                        } else if constexpr (std::is_same_v<T, MultiDeviceHostStorage>) {
                            TT_THROW("MultiDeviceHostStorage doesn't support buffer_address method");
                        } else {
                            raise_unsupported_storage<T>();
                        }
                    },
                    self.get_storage());
            },
            R"doc(
            Get the address of the underlying buffer.

            The tensor must be on the single device when calling this function.

            .. code-block:: python

                address = tt_tensor.buffer_address()

        )doc")
        .def(
            "get_layout", [](const Tensor &self) { return self.get_layout(); }, R"doc(
            Get memory layout of TT Tensor.

            .. code-block:: python

                layout = tt_tensor.get_layout()

        )doc")
        .def(
            "get_tile", [](const Tensor &self) { return self.get_tile(); }, R"doc(
            Get tile dims of TT Tensor.

            .. code-block:: python

                tile = tt_tensor.get_tile()

        )doc")
        .def(
            "memory_config", [](const Tensor &self) { return self.memory_config(); }, R"doc(
            Get buffer type of TT Tensor.

            .. code-block:: python

                memory_config = tt_tensor.memory_config()

        )doc")
        .def(
            "is_allocated", [](const Tensor &self) { return self.is_allocated(); }, R"doc(
            Check if TT Tensor is allocated.

            .. code-block:: python

                is_sharded = tt_tensor.is_sharded()

        )doc")
        .def("is_contiguous", [](const Tensor &self) -> bool { return self.is_contiguous(); })
        .def(
            "is_sharded", [](const Tensor &self) { return self.is_sharded(); }, R"doc(
            Check if TT Tensor is sharded.

            .. code-block:: python

                is_sharded = tt_tensor.is_sharded()

        )doc")
        .def(
            "get_dtype", [](const Tensor &self) { return self.get_dtype(); }, R"doc(
            Get dtype of TT Tensor.

            .. code-block:: python

                dtype = tt_tensor.get_dtype()
        )doc")
        .def(
            "reshape",
            [](Tensor &self, int N, int C, int H, int W) { return self.reshape(N, C, H, W); },
            R"doc(
                Reshapes TT tensor

                .. code-block:: python

                    reshaped_tensor = tt_tensor.reshape(N, C, H, W)
            )doc")
        .def(
            "reshape",
            [](Tensor &self, const ttnn::Shape &shape) -> Tensor { return self.reshape(shape); },
            R"doc(
                Reshapes TT tensor

                .. code-block:: python

                    reshaped_tensor = tt_tensor.reshape((4, 3, 32))
            )doc")
        .def_property(
            "tensor_id",
            [](const Tensor &self) { return self.tensor_id; },
            [](Tensor &self, std::size_t tensor_id) { self.tensor_id = tensor_id; });
}

}  // namespace ttnn::tensor
