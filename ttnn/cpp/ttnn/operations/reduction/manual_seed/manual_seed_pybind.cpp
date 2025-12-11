// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed_pybind.hpp"

#include "manual_seed.hpp"

#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::reduction::detail {
namespace py = pybind11;

void bind_manual_seed_operation(py::module& module) {
    const auto* doc = R"doc(
            Sets a seed to pseudo random number generators (PRNGs) on the specified device.

            This operation allows users to either set a single seed value to all PRNGs in the device, or to specify potentially different seed values to PRNGs at the cores assigned to the provided user IDs.

            Args:
                seeds (uint32_t or ttnn.Tensor): A single integer seed or a tensor of seeds to initialize the random number generator.

            Keyword Args:
                device (ttnn.MeshDevice, optional): The device on which to set the manual seed. Provided only if user_ids is uint32_t or None.
                user_ids (uint32_t or ttnn.Tensor, optional): An optional user ID or tensor of user IDs associated with the seeds.
                sub_core_grids (optional): Custom core range set must be provided for multi-user execution. Core IDs are constrained to numbers 0 to 31.
            Returns:
                Tensor (ttnn.Tensor): An empty tensor, as this operation does not produce a meaningful output. To be changed in the future.

            Note:

                Supported dtypes and layout for seeds tensor values:

                .. list-table::
                    :header-rows: 1

                    * - Dtypes
                      - Layouts
                    * - UINT32
                      - ROW_MAJOR_LAYOUT

                Supported dtypes and layout for user_ids tensor values:

                .. list-table::
                    :header-rows: 1

                    * - Dtypes
                      - Layouts
                    * - UINT32
                      - ROW_MAJOR_LAYOUT
        )doc";
    using OperationType = decltype(ttnn::manual_seed);
    bind_registered_operation(
        module,
        ttnn::manual_seed,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const std::variant<uint32_t, ttnn::Tensor>& seeds,
               const std::optional<std::reference_wrapper<MeshDevice>> device,
               const std::optional<std::variant<uint32_t, ttnn::Tensor>>& user_ids,
               const std::optional<CoreRangeSet>& sub_core_grids) -> Tensor {
                return self(seeds, device, user_ids, sub_core_grids);
            },
            py::arg("seeds") = 0,
            py::kw_only(),
            py::arg("device") = std::nullopt,
            py::arg("user_ids") = std::nullopt,
            py::arg("sub_core_grids") = std::nullopt});
}

}  // namespace ttnn::operations::reduction::detail
