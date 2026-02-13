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
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/nanobind_helpers.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"  // NOLINT - for nanobind SmallVector binding support.
#include "ttnn-nanobind/types.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::creation {
namespace {

// Helper lambdas for full operations with explicit fill_value
auto full_float_lambda = [](const ttsl::SmallVector<uint32_t>& shape,
                            const float fill_value,
                            const std::optional<DataType>& dtype,
                            const std::optional<Layout>& layout,
                            const std::optional<MeshDevice*> device,
                            const std::optional<MemoryConfig>& memory_config,
                            std::optional<ttnn::Tensor>& optional_output_tensor) -> ttnn::Tensor {
    return ttnn::full(
        ttnn::Shape(shape), fill_value, dtype, layout, nbh::rewrap_optional(device), memory_config, optional_output_tensor);
};

auto full_int_lambda = [](const ttsl::SmallVector<uint32_t>& shape,
                          const int fill_value,
                          const std::optional<DataType>& dtype,
                          const std::optional<Layout>& layout,
                          const std::optional<MeshDevice*> device,
                          const std::optional<MemoryConfig>& memory_config,
                          std::optional<ttnn::Tensor>& optional_output_tensor) -> ttnn::Tensor {
    return ttnn::full(
        ttnn::Shape(shape), fill_value, dtype, layout, nbh::rewrap_optional(device), memory_config, optional_output_tensor);
};

// Helper lambdas for full_like operations
auto full_like_float_lambda = [](const ttnn::Tensor& tensor,
                                 const float fill_value,
                                 const std::optional<DataType>& dtype,
                                 const std::optional<Layout>& layout,
                                 const std::optional<MeshDevice*> device,
                                 const std::optional<MemoryConfig>& memory_config,
                                 std::optional<ttnn::Tensor>& optional_output_tensor) -> ttnn::Tensor {
    return ttnn::full_like(
        tensor, fill_value, dtype, layout, nbh::rewrap_optional(device), memory_config, optional_output_tensor);
};

auto full_like_int_lambda = [](const ttnn::Tensor& tensor,
                               const int fill_value,
                               const std::optional<DataType>& dtype,
                               const std::optional<Layout>& layout,
                               const std::optional<MeshDevice*> device,
                               const std::optional<MemoryConfig>& memory_config,
                               std::optional<ttnn::Tensor>& optional_output_tensor) -> ttnn::Tensor {
    return ttnn::full_like(
        tensor, fill_value, dtype, layout, nbh::rewrap_optional(device), memory_config, optional_output_tensor);
};

// Helper lambdas for zeros/ones operations
auto zeros_lambda = [](const ttsl::SmallVector<uint32_t>& shape,
                       const std::optional<DataType>& dtype,
                       const std::optional<Layout>& layout,
                       const std::optional<MeshDevice*> device,
                       const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
    return ttnn::zeros(ttnn::Shape{shape}, dtype, layout, nbh::rewrap_optional(device), memory_config);
};

auto ones_lambda = [](const ttsl::SmallVector<uint32_t>& shape,
                      const std::optional<DataType>& dtype,
                      const std::optional<Layout>& layout,
                      const std::optional<MeshDevice*> device,
                      const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
    return ttnn::ones(ttnn::Shape{shape}, dtype, layout, nbh::rewrap_optional(device), memory_config);
};

// Helper lambdas for zeros_like/ones_like operations
auto zeros_like_lambda = [](const ttnn::Tensor& tensor,
                            const std::optional<DataType>& dtype,
                            const std::optional<Layout>& layout,
                            const std::optional<MeshDevice*> device,
                            const std::optional<MemoryConfig>& memory_config,
                            std::optional<ttnn::Tensor>& optional_output_tensor) -> ttnn::Tensor {
    return ttnn::zeros_like(tensor, dtype, layout, nbh::rewrap_optional(device), memory_config, optional_output_tensor);
};

auto ones_like_lambda = [](const ttnn::Tensor& tensor,
                           const std::optional<DataType>& dtype,
                           const std::optional<Layout>& layout,
                           const std::optional<MeshDevice*> device,
                           const std::optional<MemoryConfig>& memory_config,
                           std::optional<ttnn::Tensor>& optional_output_tensor) -> ttnn::Tensor {
    return ttnn::ones_like(tensor, dtype, layout, nbh::rewrap_optional(device), memory_config, optional_output_tensor);
};

// Helper lambda for from_buffer operation
auto from_buffer_lambda = [](const nb::object& buffer,
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
            return ttnn::from_buffer(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
        }
        case DataType::UINT16: {
            auto cpp_buffer = nb::cast<std::vector<uint16_t>>(buffer);
            return ttnn::from_buffer(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
        }
        case DataType::INT32: {
            auto cpp_buffer = nb::cast<std::vector<int32_t>>(buffer);
            return ttnn::from_buffer(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
        }
        case DataType::UINT32: {
            auto cpp_buffer = nb::cast<std::vector<uint32_t>>(buffer);
            return ttnn::from_buffer(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
        }
        case DataType::FLOAT32: {
            auto cpp_buffer = nb::cast<std::vector<float>>(buffer);
            return ttnn::from_buffer(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
        }
        case DataType::BFLOAT16: {
            auto cpp_buffer = nb::cast<std::vector<::bfloat16>>(buffer);
            return ttnn::from_buffer(std::move(cpp_buffer), shape, dtype, device, layout, memory_config);
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
};

}  // namespace

void py_module(nb::module_& mod) {
    // Bind full operation
    {
        const auto* doc = R"doc(
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
        )doc";

        ttnn::bind_function<"full">(
            mod,
            doc,
            ttnn::overload_t(
                full_float_lambda,
                nb::keep_alive<0, 6>(),
                nb::arg("shape"),
                nb::arg("fill_value"),
                nb::arg("dtype") = nb::none(),
                nb::arg("layout") = nb::none(),
                nb::arg("device") = nb::none(),
                nb::arg("memory_config") = nb::none(),
                nb::arg("optional_tensor") = nb::none()),
            ttnn::overload_t(
                full_int_lambda,
                nb::keep_alive<0, 6>(),
                nb::arg("shape"),
                nb::arg("fill_value"),
                nb::arg("dtype") = nb::none(),
                nb::arg("layout") = nb::none(),
                nb::arg("device") = nb::none(),
                nb::arg("memory_config") = nb::none(),
                nb::arg("optional_tensor") = nb::none()));
    }

    // Bind zeros operation
    {
        const auto* doc = R"doc(
        Creates a tensor with the specified shape and fills it with the value of 0.0.

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
            ttnn.Tensor: A tensor filled with 0.0.

        Note:
            Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, FLOAT32       |       ROW_MAJOR, TILE           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+
        )doc";

        ttnn::bind_function<"zeros">(
            mod,
            doc,
            ttnn::overload_t(
                zeros_lambda,
                nb::keep_alive<0, 5>(),
                nb::arg("shape"),
                nb::arg("dtype") = nb::none(),
                nb::arg("layout") = nb::none(),
                nb::arg("device") = nb::none(),
                nb::arg("memory_config") = nb::none()));
    }

    // Bind ones operation
    {
        const auto* doc = R"doc(
        Creates a tensor with the specified shape and fills it with the value of 1.0.

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
            ttnn.Tensor: A tensor filled with 1.0.

        Note:
            
        )doc";

        ttnn::bind_function<"ones">(
            mod,
            doc,
            ttnn::overload_t(
                ones_lambda,
                nb::keep_alive<0, 5>(),
                nb::arg("shape"),
                nb::arg("dtype") = nb::none(),
                nb::arg("layout") = nb::none(),
                nb::arg("device") = nb::none(),
                nb::arg("memory_config") = nb::none()));
    }

    // Bind full_like operation
    {
        const auto* doc = R"doc(
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
        )doc";

        ttnn::bind_function<"full_like">(
            mod,
            doc,
            ttnn::overload_t(
                full_like_float_lambda,
                nb::keep_alive<0, 6>(),
                nb::arg("tensor"),
                nb::arg("fill_value"),
                nb::arg("dtype") = nb::none(),
                nb::arg("layout") = nb::none(),
                nb::arg("device") = nb::none(),
                nb::arg("memory_config") = nb::none(),
                nb::arg("optional_tensor") = nb::none()),
            ttnn::overload_t(
                full_like_int_lambda,
                nb::keep_alive<0, 6>(),
                nb::arg("tensor"),
                nb::arg("fill_value"),
                nb::arg("dtype") = nb::none(),
                nb::arg("layout") = nb::none(),
                nb::arg("device") = nb::none(),
                nb::arg("memory_config") = nb::none(),
                nb::arg("optional_tensor") = nb::none()));
    }

    // Bind zeros_like operation
    {
        const auto* doc = R"doc(
        Creates a tensor of the same shape as the input tensor and fills it with the value of 0.0. The data type, layout, device, and memory configuration of the resulting tensor can be specified.

        Args:
            tensor (ttnn.Tensor): The tensor to use as a template for the shape of the new tensor.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device | ttnn.MeshDevice, optional): The device on which the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: A tensor filled with 0.0.

        Note:
            Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, FLOAT32       |       ROW_MAJOR, TILE           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+
        )doc";

        ttnn::bind_function<"zeros_like">(
            mod,
            doc,
            ttnn::overload_t(
                zeros_like_lambda,
                nb::keep_alive<0, 5>(),
                nb::arg("tensor"),
                nb::arg("dtype") = nb::none(),
                nb::arg("layout") = nb::none(),
                nb::arg("device") = nb::none(),
                nb::arg("memory_config") = nb::none(),
                nb::arg("optional_tensor") = nb::none()));
    }

    // Bind ones_like operation
    {
        const auto* doc = R"doc(
        Creates a tensor of the same shape as the input tensor and fills it with the value of 1.0. The data type, layout, device, and memory configuration of the resulting tensor can be specified.

        Args:
            tensor (ttnn.Tensor): The tensor to use as a template for the shape of the new tensor.
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `None`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `None`.
            device (ttnn.Device | ttnn.MeshDevice, optional): The device on which the tensor will be allocated. Defaults to `None`.
            memory_config (ttnn.MemoryConfig, optional): The memory configuration of the tensor. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): Preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: A tensor filled with 1.0.

        Note:
            
        )doc";

        ttnn::bind_function<"ones_like">(
            mod,
            doc,
            ttnn::overload_t(
                ones_like_lambda,
                nb::keep_alive<0, 5>(),
                nb::arg("tensor"),
                nb::arg("dtype") = nb::none(),
                nb::arg("layout") = nb::none(),
                nb::arg("device") = nb::none(),
                nb::arg("memory_config") = nb::none(),
                nb::arg("optional_tensor") = nb::none()));
    }

    // Bind arange operation
    {
        const auto* doc = R"doc(
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
        )doc";

        ttnn::bind_function<"arange">(
            mod,
            doc,
            ttnn::overload_t(
                [](const int64_t start,
                   const int64_t end,
                   const int64_t step,
                   const DataType dtype,
                   const std::optional<MeshDevice*> device,
                   const MemoryConfig& memory_config,
                   const Layout layout) -> ttnn::Tensor {
                    return ttnn::arange(start, end, step, dtype, nbh::rewrap_optional(device), memory_config, layout);
                },
                nb::keep_alive<0, 6>(),
                nb::arg("start"),
                nb::arg("end"),
                nb::arg("step") = 1,
                nb::kw_only(),
                nb::arg("dtype") = DataType::BFLOAT16,
                nb::arg("device") = nb::none(),
                nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,
                nb::arg("layout") = Layout::ROW_MAJOR),
            ttnn::overload_t(
                [](const int64_t end,
                   const DataType dtype,
                   const std::optional<MeshDevice*> device,
                   const MemoryConfig& memory_config,
                   const Layout layout) -> ttnn::Tensor {
                    return ttnn::arange(end, dtype, nbh::rewrap_optional(device), memory_config, layout);
                },
                nb::keep_alive<0, 4>(),
                nb::arg("end"),
                nb::kw_only(),
                nb::arg("dtype") = DataType::BFLOAT16,
                nb::arg("device") = nb::none(),
                nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG,
                nb::arg("layout") = Layout::ROW_MAJOR));
    }

    // Bind empty operation
    {
        const auto* doc = R"doc(
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
            Supported dtypes, layouts, and ranks:

        +----------------------------+---------------------------------+-------------------+
        |     Dtypes                 |         Layouts                 |     Ranks         |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT16, FLOAT32       |       ROW_MAJOR, TILE           |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+
        |    BFLOAT_8                |          TILE                   |      2, 3, 4      |
        +----------------------------+---------------------------------+-------------------+
        )doc";

        ttnn::bind_function<"empty">(
            mod,
            doc,
            ttnn::overload_t(
                [](const ttsl::SmallVector<uint32_t>& shape,
                   const DataType& dtype,
                   const Layout& layout,
                   MeshDevice* device,
                   const MemoryConfig& memory_config) -> ttnn::Tensor {
                    return ttnn::empty(ttnn::Shape{shape}, dtype, layout, device, memory_config);
                },
                nb::keep_alive<0, 5>(),
                nb::arg("shape"),
                nb::arg("dtype") = DataType::BFLOAT16,
                nb::arg("layout") = Layout::ROW_MAJOR,
                nb::arg("device"),
                nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG));
    }

    // Bind empty_like operation
    {
        const auto* doc = R"doc(
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
        )doc";

        ttnn::bind_function<"empty_like">(
            mod,
            doc,
            ttnn::overload_t(
                [](const ttnn::Tensor& reference,
                   const std::optional<DataType>& dtype,
                   const std::optional<Layout>& layout,
                   const std::optional<MeshDevice*> device,
                   const std::optional<MemoryConfig>& memory_config) -> ttnn::Tensor {
                    return ttnn::empty_like(reference, dtype, layout, nbh::rewrap_optional(device), memory_config);
                },
                nb::keep_alive<0, 5>(),
                nb::arg("tensor"),
                nb::kw_only(),
                nb::arg("dtype") = DataType::BFLOAT16,
                nb::arg("layout") = Layout::ROW_MAJOR,
                nb::arg("device") = nb::none(),
                nb::arg("memory_config") = ttnn::DRAM_MEMORY_CONFIG));
    }

    // Bind from_buffer operation
    {
        const auto* doc = R"doc(
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
        )doc";

        ttnn::bind_function<"from_buffer">(
            mod,
            doc,
            ttnn::overload_t(
                from_buffer_lambda,
                nb::keep_alive<0, 5>(),
                nb::arg("buffer"),
                nb::arg("shape"),
                nb::arg("dtype"),
                nb::arg("device"),
                nb::arg("layout") = std::nullopt,
                nb::arg("memory_config") = std::nullopt));
    }
}

}  // namespace ttnn::operations::creation
