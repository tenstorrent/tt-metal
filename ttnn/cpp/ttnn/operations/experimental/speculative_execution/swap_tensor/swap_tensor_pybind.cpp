// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swap_tensor_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "swap_tensor.hpp"
#include "cpp/pybind11/decorators.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::speculative_execution {

void py_bind_swap_tensor(py::module& module) {
    namespace py = pybind11;

    auto doc =
        R"doc(
        Swap the tensor between two devices.
        )doc";

    using OperationType = decltype(ttnn::experimental::swap_tensor);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::swap_tensor,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
               uint32_t num_links,
               uint8_t queue_id) { return self(queue_id, input_tensor, multi_device_global_semaphore, num_links); },
            py::arg("input_tensor").noconvert(),
            py::kw_only(),
            py::arg("multi_device_global_semaphore").noconvert(),
            py::arg("num_links") = 1,
            py::arg("queue_id") = 0,
        });
}
}  // namespace ttnn::operations::experimental::speculative_execution
