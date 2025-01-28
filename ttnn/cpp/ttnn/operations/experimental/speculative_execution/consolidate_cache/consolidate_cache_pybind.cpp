// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "consolidate_cache_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "consolidate_cache.hpp"
#include "cpp/pybind11/decorators.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::speculative_execution {

void py_bind_consolidate_cache(py::module& module) {
    namespace py = pybind11;

    auto doc =
        R"doc(
        Consolidate kv cache between two devices.
        )doc";

    using OperationType = decltype(ttnn::experimental::consolidate_cache);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::consolidate_cache,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& priority_tensor,
               const ttnn::Tensor& other_priority_tensor,
               const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
               uint32_t num_links,
               uint8_t queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    priority_tensor,
                    other_priority_tensor,
                    multi_device_global_semaphore,
                    num_links);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("priority_tensor").noconvert(),
            py::arg("other_priority_tensor").noconvert(),
            py::kw_only(),
            py::arg("multi_device_global_semaphore").noconvert(),
            py::arg("num_links") = 1,
            py::arg("queue_id") = 0,
        });
}
}  // namespace ttnn::operations::experimental::speculative_execution
