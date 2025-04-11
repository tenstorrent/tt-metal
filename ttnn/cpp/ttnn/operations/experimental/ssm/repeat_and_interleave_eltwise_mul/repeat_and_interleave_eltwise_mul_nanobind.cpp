// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_and_interleave_eltwise_mul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "repeat_and_interleave_eltwise_mul.hpp"
#include "cpp/ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::ssm::detail {

void bind_repeat_and_interleave_eltwise_mul(nb::module_& mod) {
    using OperationType = decltype(ttnn::experimental::repeat_and_interleave_eltwise_mul);

    const auto doc = R"doc(
        Performs a special eltwise multiply for SSM models. Given tensor A with shape [1, 1, 32, 32] and tensor B with shape [1, 1, 32, W] where W is some multiple of 32, perform the following PyTorch equivalent:
            A.repeat(1, 1, 1, W) * B.repeat_interleave(32, dim=-1))doc";

    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::repeat_and_interleave_eltwise_mul,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& a,
               const ttnn::Tensor& b,
               const std::optional<MemoryConfig>& memory_config,
               const std::optional<DataType> dtype,
               const std::optional<MathFidelity> math_fidelity,
               QueueId queue_id) { return self(queue_id, a, b, memory_config, dtype, math_fidelity); },
            nb::arg("a"),
            nb::arg("b"),
            nb::kw_only(),
            nb::arg("memory_config") = std::nullopt,
            nb::arg("dtype") = std::nullopt,
            nb::arg("math_fidelity") = std::nullopt,
            nb::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::ssm::detail
