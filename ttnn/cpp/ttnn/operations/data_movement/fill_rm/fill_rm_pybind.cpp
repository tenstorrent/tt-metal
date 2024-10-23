// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fill_rm_pybind.hpp"
#include "fill_rm.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"


namespace ttnn::operations::data_movement {
namespace detail {
namespace py = pybind11;

void bind_fill_rm_op(py::module& module) {
    auto doc = fmt::format(
        R"doc(
            Generates an NCHW row-major tensor and fill it with high values up to
            hOnes, wOnes in each HW tile with the rest padded with high values. So
            for H=2, W=3, hFill=1, wFill=2 the following tensor will be generated:

            .. code-block::

                +------------> W
                | hi hi lo
                | lo lo lo
                |
                v H

            H, W are expected to be multiples of 32.

            The 'any' Tensor arg is only used to pass the device and resulting
            tensor dtype.

            val_hi/lo are expected to be floats.

            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | Argument | Description                                                           | Data type             | Valid range            | Required |
            +==========+=======================================================================+=======================+========================+==========+
            | N        | Batch count of output tensor                                          | int                   | N > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | C        | Channel count of output tensor                                        | int                   | C > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | H        | Height count of output tensor                                         | int                   | H > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | W        | Width count of output tensor                                          | int                   | W > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | hOnes    | Height of high values region                                          | int                   | hOnes <= H             | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | wOnes    | Width of high values region                                           | int                   | wOnes <= W             | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | any      | Any input tensor with desired device and data types for output tensor | tt_lib.tensor.Tensor  |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | val_hi   | High value to use                                                     | float                 |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | val_lo   | Low value to use                                                      | float                 |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+

            Args:
                N (number): Batch count of output tensor.
                C (number): Channel count of output tensor.
                H (number): Height count of output tensor.
                W (number): Width count of output tensor.
                hOnes (number): Height of high values region.
                wOnes (number): Width of high values region.
                any (ttnn.tensor): Any input tensor with desired device and data types for output tensor. value greater than 0
                val_hi (number): High value to use.
                val_lo (number): Low value to use.

            Keyword args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                ttnn.Tensor: the output tensor.

        )doc",
        ttnn::fill_rm.base_name());

    using OperationType = decltype(ttnn::fill_rm);
    ttnn::bind_registered_operation(
        module,
        ttnn::fill_rm,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
                uint32_t N,
                uint32_t C,
                uint32_t H,
                uint32_t W,
                uint32_t hOnes,
                uint32_t wOnes,
                const Tensor& any,
                const float val_hi,
                const float val_lo,
                const std::optional<MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, N, C, H, W, hOnes, wOnes, any, val_hi, val_lo, memory_config);
                },
            py::arg("N"),
            py::arg("C"),
            py::arg("H"),
            py::arg("W"),
            py::arg("hOnes"),
            py::arg("wOnes"),
            py::arg("any"),
            py::arg("val_hi"),
            py::arg("val_lo"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0});
}

void bind_fill_ones_rm_op(py::module& module) {
    auto doc = fmt::format(
        R"doc(
            Same as ``fill_rm``, but ``val_hi`` is set to ``1`` and ``val_lo`` is
            ``0``.

            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | Argument | Description                                                           | Data type             | Valid range            | Required |
            +==========+=======================================================================+=======================+========================+==========+
            | N        | Batch count of output tensor                                          | int                   | N > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | C        | Channel count of output tensor                                        | int                   | C > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | H        | Height count of output tensor                                         | int                   | H > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | W        | Width count of output tensor                                          | int                   | W > 0                  | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | hOnes    | Height of high values region                                          | int                   | hOnes <= H             | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | wOnes    | Width of high values region                                           | int                   | wOnes <= W             | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | any      | Any input tensor with desired device and data types for output tensor | tt_lib.tensor.Tensor  |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+

            Args:
                N (number): Batch count of output tensor.
                C (number): Channel count of output tensor.
                H (number): Height count of output tensor.
                W (number): Width count of output tensor.
                hOnes (number): Height of high values region.
                wOnes (number): Width of high values region.
                any (ttnn.tensor): Any input tensor with desired device and data types for output tensor. value greater than 0

            Keyword args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                ttnn.Tensor: the output tensor.
        )doc",
        ttnn::fill_ones_rm.base_name());

    using OperationType = decltype(ttnn::fill_ones_rm);
    ttnn::bind_registered_operation(
        module,
        ttnn::fill_ones_rm,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
                uint32_t N,
                uint32_t C,
                uint32_t H,
                uint32_t W,
                uint32_t hOnes,
                uint32_t wOnes,
                const Tensor& any,
                const std::optional<MemoryConfig>& memory_config,
                uint8_t queue_id) {
                    return self(queue_id, N, C, H, W, hOnes, wOnes, any, memory_config);
                },
            py::arg("N"),
            py::arg("C"),
            py::arg("H"),
            py::arg("W"),
            py::arg("hOnes"),
            py::arg("wOnes"),
            py::arg("any"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0});

}

void bind_full_op(py::module& module) {
    auto doc = fmt::format(
        R"doc(
            Creates a tensor of shape (N, C, H, W) filled with the specified value.

            This is a wrapper for `fill_rm` that accepts a shape as a vector of integers
            and fills the entire tensor with the given `fill_value`.

            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | Argument | Description                                                           | Data type             | Valid range            | Required |
            +==========+=======================================================================+=======================+========================+==========+
            | shape     | Shape of the tensor as a vector of 4 integers [N, C, H, W]           | std::vector<int>      | Length <= 4            | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | fill_value| Value to fill the entire tensor with                                 | float                 |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | device    | Device to allocate tensor on                                         | float                 |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+

            Args:
                shape (list of int): The shape of the tensor in the form of [N, C, H, W].
                fill_value (float): Value to fill the entire tensor.
                device (ttnn.Device): Device to allocate the tensor on.

            Returns:
                ttnn.Tensor: The output tensor filled with `fill_value`.
        )doc",
        "full_tensor_op",
        ttnn::full.base_name());

    using OperationType = decltype(ttnn::full);
    ttnn::bind_registered_operation(
        module,
        ttnn::full,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const std::vector<uint32_t>& shape,
               const float fill_value,
               ttnn::Device* device,
               const std::optional<MemoryConfig>& memory_config,
               uint8_t queue_id) {
                   return self(queue_id, shape, fill_value, device, memory_config);
               },
            py::arg("shape"),
            py::arg("fill_value"),
            py::arg("device"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0});
}

void bind_fill_op(py::module& module) {
    auto doc = fmt::format(
        R"doc(
            Fills a tensor with the specified value.

            This is a wrapper for `fill_rm` that sets `hOnes` and `wOnes` to `0` and passes in N,C,H,W as the tensor shape,
            returning the tensor filled with the given `fill_value`.

            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | Argument | Description                                                           | Data type             | Valid range            | Required |
            +==========+=======================================================================+=======================+========================+==========+
            | fill_value| Value to fill the entire tensor with                                 | float                 |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+
            | any      | Any input tensor with desired device and data types for output tensor | tt_lib.tensor.Tensor  |                        | Yes      |
            +----------+-----------------------------------------------------------------------+-----------------------+------------------------+----------+

            Args:
                fill_value (number): Value to fill the entire tensor.
                any (ttnn.tensor): Any input tensor with desired device and data types for output tensor.

            Keyword args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

            Returns:
                ttnn.Tensor: the output tensor filled with `fill_value`.
        )doc",
        ttnn::fill.base_name());

    using OperationType = decltype(ttnn::fill);
    ttnn::bind_registered_operation(
        module,
        ttnn::fill,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const float fill_value,
               const Tensor& any,
               const std::optional<MemoryConfig>& memory_config,
               uint8_t queue_id) {
                   return self(queue_id, fill_value, any, memory_config);
               },
            py::arg("fill_value"),
            py::arg("any"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0});
}
} //detail

void bind_fill_rm(py::module& module) {
   detail::bind_fill_rm_op(module);
   detail::bind_fill_ones_rm_op(module);
   detail::bind_full_op(module);
   detail::bind_fill_op(module);
}

}  // namespace ttnn::operations::data_movement::detail
