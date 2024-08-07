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

} //detail

void bind_fill_rm(py::module& module) {
   detail::bind_fill_rm_op(module);
   detail::bind_fill_ones_rm_op(module);
}

}  // namespace ttnn::operations::data_movement::detail
