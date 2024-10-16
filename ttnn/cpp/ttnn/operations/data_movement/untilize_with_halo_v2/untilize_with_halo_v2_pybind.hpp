// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "untilize_with_halo_v2.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_untilize_with_halo_v2(py::module& module) {
    auto doc =
        R"doc(

            Untilizes input tiled data to row major format and constructs halo'd output shards.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.
            padding_config (ttnn.Tensor): Padding config tensor.
            local_config (ttnn.Tensor): Local config tensor.
            remote_config (ttnn.Tensor): Remote config tensor.

        Keyword Args:
            pad_val (int, optional): pad value.
            ncores_nhw (int, optional): Number of cores per NHW..
            max_out_nsticks_per_core (int, optional): Max output nsticks per core.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            remote_read (bool, optional): Remote read.  Defaults to `False`.
            transpose_mcast (bool, optional): Transpose mcast.  Defaults to `False`.
            queue_id (int, optional): command queue id. Defaults to `0`.
        Returns:
            ttnn.Tensor: the output tensor.

        )doc";

    using OperationType = decltype(ttnn::untilize_with_halo_v2);
    ttnn::bind_registered_operation(module,
                                    ttnn::untilize_with_halo_v2,
                                    doc,
                                    ttnn::pybind_overload_t{
                                        [](const OperationType& self,
                                           const ttnn::Tensor& input_tensor,
                                           const ttnn::Tensor& padding_config,
                                           const ttnn::Tensor& local_config,
                                           const ttnn::Tensor& remote_config,
                                           const uint32_t pad_val,
                                           const uint32_t ncores_nhw,
                                           const uint32_t max_out_nsticks_per_core,
                                           const std::optional<MemoryConfig>& memory_config,
                                           const bool remote_read,
                                           const bool transpose_mcast,
                                           uint8_t queue_id) {
                                            return self(queue_id,
                                                        input_tensor,
                                                        padding_config,
                                                        local_config,
                                                        remote_config,
                                                        pad_val,
                                                        ncores_nhw,
                                                        max_out_nsticks_per_core,
                                                        memory_config,
                                                        remote_read,
                                                        transpose_mcast);
                                        },
                                        py::arg("input_tensor"),
                                        py::arg("padding_config"),
                                        py::arg("local_config"),
                                        py::arg("remote_config"),
                                        py::kw_only(),
                                        py::arg("pad_val"),
                                        py::arg("ncores_nhw"),
                                        py::arg("max_out_nsticks_per_core"),
                                        py::arg("memory_config") = std::nullopt,
                                        py::arg("remote_read") = false,
                                        py::arg("transpose_mcast") = false,
                                        py::arg("queue_id") = 0,
                                    });
}
}  // namespace ttnn::operations::data_movement::detail
