// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/plusone/plusone.hpp"
#include "ttnn/operations/experimental/plusone/plusone_pybind.hpp"

namespace ttnn::operations::experimental::plusone::detail {
namespace py = pybind11;
void bind_experimental_plusone_operation(py::module& module) {
    auto doc =
        R"doc(
            Returns input tensor elements increased by 1.
            Input tensor must have UINT32 data type, ROW_MAJOR layout, and 1-D shape.
            This op only gives decent performance for small tensors (up to 100 elements).
            This op allows you to skip the addition on negative entries using the skip_negative_entries flag. If enabled, only positive entries will be incremented by checking if tensor values overflow INT32_MAX / are negative.
            This op also allows you to specify the core to use in the sub_core_grids argument.
            If the input tensor is L1 sharded on the sub core grid, each individual shard will be incremented with output residing in L1 of same sub core grid.
            If the input tensor is DRAM interleaved, only 1 core should be used as the sub core grid (uses 1 core by default).
            Equivalent pytorch code:

            .. code-block:: python

                return torch.add(input_tensor, 1)

            Args:
                * :attr:`input_tensor`: Input Tensor for plusone.
                * :attr:`sub_core_grids`: Sub core grid of cores where the addition would take place
                * :attr:`skip_negative_entries`: bool flag to skip incrementing values that are negative or overflow past INT32_MAX. Defaults to False

        )doc";

    using OperationType = decltype(ttnn::plus_one);
    bind_registered_operation(
        module,
        ttnn::plus_one,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<CoreRangeSet>& sub_core_grids,
               bool skip_negative_entries) { return self(input_tensor, sub_core_grids, skip_negative_entries); },
            py::arg("input_tensor").noconvert(),
            py::arg("sub_core_grids") = std::nullopt,
            py::arg("skip_negative_entries") = false});
}

}  // namespace ttnn::operations::experimental::plusone::detail
