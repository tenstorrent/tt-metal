// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "creation.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bfloat16_type_caster.hpp"  // NOLINT - for nanobind bfloat16 binding support.
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/types.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::creation {
namespace {

template <typename creation_operation_t, typename fill_value_t>
auto create_nanobind_full_overload() {
    return ttnn::nanobind_overload_t{
        [](const creation_operation_t& self,
           const std::vector<uint32_t>& shape,
           const fill_value_t fill_value,
           const std::optional<DataType>& dtype,
           const std::optional<Layout>& layout,
           const std::optional<std::reference_wrapper<MeshDevice>> device,
           const std::optional<MemoryConfig>& memory_config,
           std::optional<ttnn::Tensor>& optional_output_tensor) -> ttnn::Tensor {
            return self(ttnn::Shape(shape), fill_value, dtype, layout, device, memory_config, optional_output_tensor);
        },
        nb::arg("shape"),
        nb::arg("fill_value"),
        nb::arg("dtype") = nb::none(),
        nb::arg("layout") = nb::none(),
        nb::arg("device") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("optional_tensor") = nb::none()};
}

template <typename creation_operation_t, typename fill_value_t>
auto create_nanobind_full_like_overload() {
    return ttnn::nanobind_overload_t{
        [](const creation_operation_t& self,
           const ttnn::Tensor& tensor,
           const fill_value_t fill_value,
           const std::optional<DataType>& dtype,
           const std::optional<Layout>& layout,
           const std::optional<std::reference_wrapper<MeshDevice>> device,
           const std::optional<MemoryConfig>& memory_config,
           std::optional<ttnn::Tensor>& optional_output_tensor) -> ttnn::Tensor {
            return self(tensor, fill_value, dtype, layout, device, memory_config, optional_output_tensor);
        },
        nb::arg("tensor"),
        nb::arg("fill_value"),
        nb::arg("dtype") = nb::none(),
        nb::arg("layout") = nb::none(),
        nb::arg("device") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("optional_tensor") = nb::none()};
}

// TODO_NANOBIND: buffer api -> ndarray
template <typename creation_operation_t>
auto create_nanobind_from_buffer_overload() {
    return ttnn::nanobind_overload_t{
        [](const creation_operation_t& self,
           const nb::object& buffer,
           const Shape& shape,
           const DataType dtype,
           MeshDevice* device,
           const std::optional<Layout>& layout,
           const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
            // Overloading this with templates is not working quite as expected,
            // the problem is that the buffer is a nb::object, so we can't deduce the type of the data.
            // and sometimes the wrong type is handling the data.
            // For instance, a list of int16 can be interpreted as a list of int32 and the data will be a missmatch
            // in further validations.
            switch (dtype) {
                case DataType::UINT8: {
                    auto cpp_buffer = nb::cast<std::vector<uint8_t>>(buffer);
                    return self(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
                }
                case DataType::UINT16: {
                    auto cpp_buffer = nb::cast<std::vector<uint16_t>>(buffer);
                    return self(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
                }
                case DataType::INT32: {
                    auto cpp_buffer = nb::cast<std::vector<int32_t>>(buffer);
                    return self(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
                }
                case DataType::UINT32: {
                    auto cpp_buffer = nb::cast<std::vector<uint32_t>>(buffer);
                    return self(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
                }
                case DataType::FLOAT32: {
                    auto cpp_buffer = nb::cast<std::vector<float>>(buffer);
                    return self(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
                }
                case DataType::BFLOAT16: {
                    auto cpp_buffer = nb::cast<std::vector<::bfloat16>>(buffer);
                    return self(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
                }
                case DataType::BFLOAT8_B:
                case DataType::BFLOAT4_B:
                case DataType::INVALID: {
                    // convert_to_data_type() in types.hpp has not an implementation for bfloat8_b and bfloat4_b
                    // Both are empty structs, so let's not allow users to use them for this particular operation
                    TT_THROW("Unreachable");
                }
            }
            // This is a fallback to make the compiler happy.
            TT_THROW("Unreachable");
        },
        nb::arg("buffer"),
        nb::arg("shape"),
        nb::arg("dtype"),
        nb::arg("device"),
        nb::arg("layout") = std::nullopt,
        nb::arg("memory_config") = std::nullopt};
}

template <typename creation_operation_t>
void bind_full_operation(nb::module_& mod, const creation_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor of the specified shape and fills it with the specified scalar value.

        Args:
            shape (ttnn.Shape): The shape of the tensor.
            fill_value (float): The value to fill the tensor with.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device | ttnn.MeshDevice, optional): The device on which the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.

        Note:
            ROW_MAJOR_LAYOUT requires last dimension (shape[-1]) to be a multiple of 2 with dtype BFLOAT16 or UINT16.
            TILE_LAYOUT requires width (shape[-1]) and height (shape[-2]) dimension to be multiple of 32.

        Returns:
            ttnn.Tensor: A filled tensor of specified shape and value.

        Example:
            >>> filled_tensor = ttnn.full(shape=[2, 2], fill_value=7.0, dtype=ttnn.bfloat16)
            >>> print(filled_tensor)
            ttnn.Tensor([[[[7.0,  7.0],
                            [7.0,  7.0]]]], shape=Shape([2, 2]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
        )doc",
        operation.base_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        create_nanobind_full_overload<creation_operation_t, float>(),
        create_nanobind_full_overload<creation_operation_t, int>());
}

template <typename creation_operation_t>
void bind_full_operation_with_hard_coded_value(
    nb::module_& mod,
    const creation_operation_t& operation,
    const std::string& value_string,
    const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor with the specified shape and fills it with the value of {1}.

        Args:
            shape (ttnn.Shape): The shape of the tensor.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device | ttnn.MeshDevice, optional): The device on which the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `None`.

        Note:
            ROW_MAJOR_LAYOUT requires last dimension (shape[-1]) to be a multiple of 2 with dtype BFLOAT16 or UINT16.
            TILE_LAYOUT requires requires width (shape[-1]), height (shape[-2]) dimension to be multiple of 32.

        Returns:
            ttnn.Tensor: A tensor filled with {1}.

        Note:
            {2}

        Example:
            >>> tensor = ttnn.{0}(shape=[1, 2, 2, 2], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            >>> print(tensor)
            ttnn.Tensor([[[[{1}, {1}],
                            [{1}, {1}]],
                            [[{1}, {1}],
                            [{1}, {1}]]]]], shape=Shape([1, 2, 2, 2]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
        )doc",
        operation.base_name(),
        value_string,
        info_doc);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const creation_operation_t& self,
               const std::vector<uint32_t>& shape,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<MeshDevice>> device,
               const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(ttnn::Shape{shape}, dtype, layout, device, memory_config);
            },
            nb::arg("shape"),
            nb::arg("dtype") = nb::none(),
            nb::arg("layout") = nb::none(),
            nb::arg("device") = nb::none(),
            nb::arg("memory_config") = nb::none()},
        // Accept MeshDevice* directly (Python MeshDevice maps to pointer; None maps to nullptr)
        ttnn::nanobind_overload_t{
            [](const creation_operation_t& self,
               const std::vector<uint32_t>& shape,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               MeshDevice* device,
               const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                std::optional<std::reference_wrapper<MeshDevice>> device_ref =
                    device ? std::optional<std::reference_wrapper<MeshDevice>>(std::ref(*device)) : std::nullopt;
                return self(ttnn::Shape{shape}, dtype, layout, device_ref, memory_config);
            },
            nb::arg("shape"),
            nb::arg("dtype") = nb::none(),
            nb::arg("layout") = nb::none(),
            nb::arg("device") = nb::none(),
            nb::arg("memory_config") = nb::none()});
}

template <typename creation_operation_t>
void bind_full_like_operation(nb::module_& mod, const creation_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor of the same shape as the input tensor and fills it with the specified scalar value. The data type, layout, device, and memory configuration of the resulting tensor can be specified.

        Args:
            tensor (ttnn.Tensor): The tensor to use as a template for the shape of the new tensor.
            fill_value (float | int): The value to fill the tensor with.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device | ttnn.MeshDevice, optional): The device on which the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: A filled tensor.

        Example:
            >>> tensor = ttnn.zeros(shape=(2, 3), dtype=ttnn.bfloat16)
            >>> filled_tensor = ttnn.full_like(tensor, fill_value=5.0, dtype=ttnn.bfloat16)
            >>> print(filled_tensor)
            ttnn.Tensor([[[[5.0,  5.0,  5.0],
                            [5.0,  5.0,  5.0]]]], shape=Shape([2, 3]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
        )doc",
        operation.base_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        create_nanobind_full_like_overload<creation_operation_t, float>(),
        create_nanobind_full_like_overload<creation_operation_t, int>());
}

template <typename creation_operation_t>
void bind_full_like_operation_with_hard_coded_value(
    nb::module_& mod,
    const creation_operation_t& operation,
    const std::string& value_string,
    const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor of the same shape as the input tensor and fills it with the value of {1}. The data type, layout, device, and memory configuration of the resulting tensor can be specified.

        Args:
            tensor (ttnn.Tensor): The tensor to use as a template for the shape of the new tensor.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device | ttnn.MeshDevice, optional): The device on which the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: A tensor filled with {1}.

        Note:
            {2}

        Example:
            >>> tensor = ttnn.{0}(ttnn.from_torch(torch.randn(1, 2, 2, 2), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            >>> output_tensor = ttnn.{0}(tensor=input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            >>> print(output_tensor)
            ttnn.Tensor([[[[{1}, {1}],
                            [{1}, {1}]],
                            [[{1}, {1}],
                            [{1}, {1}]]]]], shape=Shape([1, 2, 2, 2]), dtype=DataType::BFLOAT16, layout=Layout::TILE_LAYOUT)
        )doc",
        operation.base_name(),
        value_string,
        info_doc);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const creation_operation_t& self,
               const ttnn::Tensor& tensor,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<MeshDevice>> device,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<ttnn::Tensor>& optional_output_tensor) -> ttnn::Tensor {
                return self(tensor, dtype, layout, device, memory_config, optional_output_tensor);
            },
            nb::arg("tensor"),
            nb::arg("dtype") = nb::none(),
            nb::arg("layout") = nb::none(),
            nb::arg("device") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("optional_tensor") = nb::none()});
}

template <typename creation_operation_t>
void bind_arange_operation(nb::module_& mod, const creation_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Creates a tensor with values ranging from `start` (inclusive) to `end` (exclusive) with a specified `step` size. The data type, device, layout and memory configuration of the resulting tensor can be specified.

        Args:
            start (int, optional): The start of the range. Defaults to 0.
            end (int): The end of the range (exclusive).
            step (int, optional): The step size between consecutive values. Defaults to 1.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `ttnn.bfloat16`.
            device (ttnn.Device, optional): The device where the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration for the tensor. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.
            layout (ttnn.Layout, optional): The tensor layout. Defaults to `ttnn.ROW_MAJOR`.

        Returns:
            ttnn.Tensor: A tensor containing evenly spaced values within the specified range.

        Example:
            >>> tensor = ttnn.arange(start=0, end=10, step=2, dtype=ttnn.float32)
            >>> print(tensor)
            ttnn.Tensor([ 0.00000,  2.00000,  ...,  6.00000,  8.00000], shape=Shape([5]), dtype=DataType::FLOAT32, layout=Layout::ROW_MAJOR)
        )doc",
        operation.base_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const creation_operation_t& self,
               const int64_t start,
               const int64_t end,
               const int64_t step,
               const DataType dtype,
               const std::optional<std::reference_wrapper<MeshDevice>> device,
               const MemoryConfig& memory_config,
               const Layout layout) -> ttnn::Tensor {
                return self(start, end, step, dtype, device, memory_config, layout);
            },
            nb::arg("start"),
            nb::arg("end"),
            nb::arg("step") = 1,
            nb::kw_only(),
            nb::arg("dtype") = DataType::BFLOAT16,
            nb::arg("device") = nb::none(),
            nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,
            nb::arg("layout") = Layout::ROW_MAJOR},
        ttnn::nanobind_overload_t{
            [](const creation_operation_t& self,
               const int64_t end,
               const DataType dtype,
               const std::optional<std::reference_wrapper<MeshDevice>> device,
               const MemoryConfig& memory_config,
               const Layout layout) -> ttnn::Tensor { return self(end, dtype, device, memory_config, layout); },
            nb::arg("end"),
            nb::kw_only(),
            nb::arg("dtype") = DataType::BFLOAT16,
            nb::arg("device") = nb::none(),
            nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,
            nb::arg("layout") = Layout::ROW_MAJOR});
}

template <typename creation_operation_t>
void bind_empty_operation(nb::module_& mod, const creation_operation_t& operation, const std::string& info_doc = "") {
    auto doc = fmt::format(
        R"doc(
        Creates a device tensor with uninitialized values of the specified shape, data type, layout, and memory configuration.

        Args:
            shape (List[int]): The shape of the tensor to be created.
            dtype (ttnn.DataType, optional): The tensor data type. Defaults to `ttnn.bfloat16`.
            layout (ttnn.Layout, optional): The tensor layout. Defaults to `ttnn.ROW_MAJOR`.
            device (ttnn.Device | ttnn.MeshDevice): The device where the tensor will be allocated.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration for the operation. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.

        Returns:
            ttnn.Tensor: The output uninitialized tensor.

        Note:
            {1}

        Example:
            >>> tensor = ttnn.empty(shape=[2, 3], device=device)
            >>> print(tensor)
            ttnn.Tensor([[[[0.9, 0.21, 0.5], [0.67, 0.11, 0.30]]]], shape=Shape([2, 3]), dtype=DataType::BFLOAT16, layout=Layout::TILE)
        )doc",
        operation.base_name(),
        info_doc);

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const creation_operation_t& self,
               const std::vector<uint32_t>& shape,
               const DataType& dtype,
               const Layout& layout,
               MeshDevice* device,
               const MemoryConfig& memory_config) -> ttnn::Tensor {
                return self(ttnn::Shape{shape}, dtype, layout, device, memory_config);
            },
            nb::arg("shape"),
            nb::arg("dtype") = DataType::BFLOAT16,
            nb::arg("layout") = Layout::ROW_MAJOR,
            nb::arg("device"),
            nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG});
}

template <typename creation_operation_t>
void bind_from_buffer_operation(nb::module_& mod, const creation_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Creates a device tensor with values from a buffer of the specified, data type, layout, and memory configuration.

        Args:
            buffer (List[Any]): The buffer to be used to create the tensor.
            shape (ttnn.Shape): The shape of the tensor to be created.
            dtype (ttnn.DataType): The tensor data type.
            device (ttnn.Device | ttnn.MeshDevice): The device where the tensor will be allocated.
            layout (ttnn.Layout, optional): The tensor layout. Defaults to `ttnn.ROW_MAJOR` unless `dtype` is `ttnn.bfloat4` or `ttnn.bfloat8`, in which case it defaults to `ttnn.TILE`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration for the operation. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.

        Returns:
            ttnn.Tensor: A tensor with the values from the buffer.

        Example:
            >>> tensor = ttnn.{0}(buffer=[1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=ttnn.int32, device=device)
            >>> print(tensor)
            ttnn.Tensor([[1, 2, 3], [4, 5, 6]], shape=Shape([2, 3]), dtype=DataType::INT32, layout=Layout::ROW_MAJOR)
        )doc",
        operation.base_name());

    bind_registered_operation(mod, operation, doc, create_nanobind_from_buffer_overload<creation_operation_t>());
}

template <typename creation_operation_t>
void bind_empty_like_operation(nb::module_& mod, const creation_operation_t& operation) {
    auto doc = fmt::format(
        R"doc(
        Creates a new tensor with the same shape as the given `reference`, but without initializing its values. The data type, layout, device, and memory configuration of the new tensor can be specified.

        Args:
            reference (ttnn.Tensor): The reference tensor whose shape will be used for the output tensor.

        Keyword Args:
            dtype (ttnn.DataType, optional): The desired data type of the output tensor. Defaults to `ttnn.bfloat16`.
            layout (ttnn.Layout, optional): The desired layout of the output tensor. Defaults to `ttnn.ROW_MAJOR`.
            device (ttnn.Device | ttnn.MeshDevice, optional): The device where the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration for the operation. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.

        Returns:
            ttnn.Tensor: The output uninitialized tensor with the same shape as the reference tensor.

        Example:
            >>> reference = ttnn.from_torch(torch.randn(2, 3), dtype=ttnn.bfloat16)
            >>> tensor = ttnn.empty_like(reference, dtype=ttnn.float32)
            >>> print(tensor)
            ttnn.Tensor([[[[0.87, 0.45, 0.22], [0.60, 0.75, 0.25]]]], shape=Shape([2, 3]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)
        )doc",
        operation.base_name());

    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const creation_operation_t& self,
               const ttnn::Tensor& reference,
               const std::optional<DataType>& dtype,
               const std::optional<Layout>& layout,
               const std::optional<std::reference_wrapper<MeshDevice>> device,
               const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                return self(reference, dtype, layout, device, memory_config);
            },
            nb::arg("tensor"),
            nb::kw_only(),
            nb::arg("dtype") = DataType::BFLOAT16,
            nb::arg("layout") = Layout::ROW_MAJOR,
            nb::arg("device") = nb::none(),
            nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG});
}

}  // namespace

void py_module(nb::module_& mod) {
    bind_full_operation(mod, ttnn::full);
    bind_full_operation_with_hard_coded_value(
        mod,
        ttnn::zeros,
        "0.0",
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, FLOAT32       |       ROW_MAJOR, TILE           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc");

    bind_full_operation_with_hard_coded_value(mod, ttnn::ones, "1.0");

    bind_full_like_operation(mod, ttnn::full_like);
    bind_full_like_operation_with_hard_coded_value(
        mod,
        ttnn::zeros_like,
        "0.0",
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, FLOAT32       |       ROW_MAJOR, TILE           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc");
    bind_full_like_operation_with_hard_coded_value(mod, ttnn::ones_like, "1.0");

    bind_arange_operation(mod, ttnn::arange);

    bind_empty_operation(
        mod,
        ttnn::empty,
        R"doc(Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, FLOAT32       |       ROW_MAJOR, TILE           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT_8                |          TILE                   |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+)doc");
    bind_empty_like_operation(mod, ttnn::empty_like);

    bind_from_buffer_operation(mod, ttnn::from_buffer);
}

}  // namespace ttnn::operations::creation
