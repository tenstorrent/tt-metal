// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed_pybind.hpp"

#include "manual_seed.hpp"

#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::reduction::detail {
namespace py = pybind11;

void bind_manual_seed_operation(py::module& module) {
    auto doc = R"doc(
            Sets a seed to pseudo random number generators (PRNGs) on the specified device.

            This operation allows users to either set a single seed value to all PRNGs in the device, or to specify potentially different seed values to PRNGs at the cores assigned to the provided user IDs.

            Args:
                device (ttnn.MeshDevice): The device on which to set the manual seed.
                seeds (int or ttnn.Tensor): A single integer seed or a tensor of seeds to initialize the random number generator.

            Keyword Args:
                user_ids (int or ttnn.Tensor, optional): An optional user ID or tensor of user IDs associated with the seeds.

            Returns:
                Tensor: An empty tensor, as this operation does not produce a meaningful output. To be changed in the future.

            Note:

                Supported dtypes and layout for seeds tensor values:

                .. list-table::
                    :header-rows: 1

                    * - Dtypes
                      - Layouts
                    * - X
                      - X
                    * - X
                      - X

                Supported dtypes and layout for user_ids tensor values:

                .. list-table::
                    :header-rows: 1

                    * - Dtypes
                      - Layouts
                    * - X, X
                      - X
        )doc";
    // TODO: To be filled when implementing device logic
    using OperationType = decltype(ttnn::manual_seed);
    bind_registered_operation(
        module,
        ttnn::manual_seed,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               MeshDevice& device,
               std::variant<uint32_t, ttnn::Tensor> seeds,
               std::optional<std::variant<uint32_t, ttnn::Tensor>> user_ids) -> Tensor {
                return self(device, seeds, user_ids);
            },
            py::arg("device"),
            py::arg("seeds"),
            py::kw_only(),
            py::arg("user_ids") = std::nullopt});
}

}  // namespace ttnn::operations::reduction::detail
