// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed_nanobind.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "manual_seed.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/nanobind_helpers.hpp"

namespace ttnn::operations::reduction::detail {

void bind_manual_seed_operation(nb::module_& mod) {
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
        mod,
        ttnn::manual_seed,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const std::variant<uint32_t, ttnn::Tensor>& seeds,
               const std::optional<MeshDevice*> device,
               const std::optional<std::variant<uint32_t, ttnn::Tensor>>& user_ids,
               const std::optional<CoreRangeSet>& sub_core_grids) -> Tensor {
                return self(seeds, nbh::rewrap_optional(device), user_ids, sub_core_grids);
            },
            nb::arg("seeds") = 0,
            nb::kw_only(),
            nb::arg("device") = nb::none(),
            nb::arg("user_ids") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()});
}

}  // namespace ttnn::operations::reduction::detail
