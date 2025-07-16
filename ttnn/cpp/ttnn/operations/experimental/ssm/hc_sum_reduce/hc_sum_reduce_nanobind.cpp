// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hc_sum_reduce_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "hc_sum_reduce.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace ttnn::operations::experimental::ssm::detail {

namespace nb = nanobind;

void bind_hc_sum_reduce(nb::module_& mod) {
    using OperationType = decltype(ttnn::experimental::hc_sum_reduce);

    const auto doc = R"doc(
        Performs a custom reduction along dim 3 which is used in the SSM block of the Mamba architecture. Performs the following PyTorch equivalent (where latent_size = 32):
            x = torch.sum(x.reshape(1, 1, shape[2], shape[3] // latent_size, latent_size), dim=-1).reshape(1, 1, shape[2], shape[3] // latent_size)
    )doc";

    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::hc_sum_reduce,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DataType> dtype,
               const std::optional<MathFidelity> math_fidelity,
               QueueId queue_id) { return self(queue_id, input, memory_config, dtype, math_fidelity); },
            nb::arg("input"),
            nb::kw_only(),
            nb::arg("memory_config") = std::nullopt,
            nb::arg("dtype") = std::nullopt,
            nb::arg("math_fidelity") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::ssm::detail
