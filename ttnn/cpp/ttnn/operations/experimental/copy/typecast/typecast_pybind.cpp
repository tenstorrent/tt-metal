// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"

#include "typecast_pybind.hpp"
#include "typecast.hpp"

namespace ttnn::operations::experimental::copy::detail {

namespace py = pybind11;

void py_bind_typecast(py::module& module) {
    auto doc = R"doc(
        Returns a new tensor which is a typecast of input tensor with new datatype``{0}``.

        Input tensors must be on device, in ROW MAJOR or TILE layout, and have matching data type.

        Datatype must be one of the following types BFLOAT16, BFLOAT8_B, BFLOAT4_B, UINT32, INT32, UINT16 and UINT8.

        Output tensor will be on device, in same layout, and have the given data type.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Required"

            "input_tensors", "Input tensors to typecast", "List of Tensors", "Yes"
            "dtype", "datatype of typecast", "Datatype", "Yes"
            "output_mem_config", "Layout of tensor in TT Accelerator device memory banks", "MemoryConfig", "No"
    )doc";

    bind_registered_operation(
        module,
        ttnn::experimental::typecast,
        doc,
        ttnn::pybind_overload_t{[](const decltype(ttnn::experimental::typecast)& self,
                                   const ttnn::Tensor& input_tensor,
                                   const ttnn::DataType dtype,
                                   const std::optional<ttnn::MemoryConfig> memory_config,
                                   const std::optional<ttnn::Tensor>& optional_output_tensor,
                                   uint8_t queue_id) {
                                    return self(queue_id, input_tensor, dtype, memory_config, optional_output_tensor);
                                },
                                py::arg("input_tensor").noconvert(),
                                py::arg("dtype").noconvert(),
                                py::arg("memory_config") = std::nullopt,
                                py::arg("optional_output_tensor") = std::nullopt,
                                py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::experimental::copy::detail
