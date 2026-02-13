// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "reshape.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::data_movement {

void bind_reshape(nb::module_& mod) {
    const auto* doc =
        R"doc(

        Returns a tensor with the new shape of ``[W, Z, Y, X]``. The X dimension of input and output tensor must have same size.

        Equivalent pytorch code:

        .. code-block:: python

            input_tensor = torch.arange(4.)
            W = 1
            Z = 1
            Y = 2
            X = 2
            output_tensor = torch.reshape(input_tensor, (W, Z, Y, X))


        Args:
            * :attr:`input_tensor`: Input Tensor.
            * :attr:`W`: W dim of tensor.
            * :attr:`Z`: Z dim of tensor.
            * :attr:`Y`: Y dim of tensor.
            * :attr:`X`: X dim of tensor.

        Keyword Args:
            * :attr:`memory_config`: Memory Config of the output tensor

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 4), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.reshape(tensor, 1, 1, 2, 2)

        )doc";

    ttnn::bind_function<"reshape_on_device">(
        mod,
        doc,
        ttnn::overload_t(
            nb::overload_cast<const ttnn::Tensor&, int, int, int, int, const std::optional<ttnn::MemoryConfig>&>(
                &ttnn::reshape_on_device),
            nb::arg("input_tensor"),
            nb::arg("W"),
            nb::arg("Z"),
            nb::arg("Y"),
            nb::arg("X"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none()));
}

}  // namespace ttnn::operations::data_movement
