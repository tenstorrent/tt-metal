// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"

#include "typecast.hpp"

namespace ttnn::operations::experimental::copy::detail {

void bind_typecast(nb::module_& mod) {
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
        mod,
        ttnn::experimental::typecast,
        doc,
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::experimental::typecast)& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::DataType dtype,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& optional_output_tensor) {
                return self(input_tensor, dtype, memory_config, optional_output_tensor);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("dtype").noconvert(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("optional_output_tensor") = nb::none()});
}

}  // namespace ttnn::operations::experimental::copy::detail
