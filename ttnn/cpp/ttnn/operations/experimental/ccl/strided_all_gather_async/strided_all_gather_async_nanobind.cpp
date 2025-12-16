// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "strided_all_gather_async_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/strided_all_gather_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace {

template <typename ccl_operation_t>
void bind_strided_all_gather_async(nb::module_& mod, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        mod,
        operation,
        doc,
        ttnn::nanobind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<ttnn::Tensor>& persistent_output_buffer,
               const int32_t dim,
               const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<uint32_t> cluster_axis,
               std::optional<uint32_t> num_workers_per_link,
               std::optional<uint32_t> num_buffers_per_channel,
               std::optional<uint32_t> mm_cores_y,
               std::optional<uint32_t> mm_block_ht,
               std::optional<uint32_t> mm_block_wt) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    persistent_output_buffer,
                    dim,
                    multi_device_global_semaphore,
                    num_links,
                    memory_config,
                    topology,
                    cluster_axis,
                    num_workers_per_link,
                    num_buffers_per_channel,
                    mm_cores_y,
                    mm_block_ht,
                    mm_block_wt);
            },
            nb::arg("input_tensor"),
            nb::arg("persistent_output_buffer"),
            nb::arg("dim"),
            nb::arg("multi_device_global_semaphore"),
            nb::kw_only(),
            nb::arg("num_links") = 1,
            nb::arg("memory_config") = nb::none(),
            nb::arg("topology") = nb::cast(ttnn::ccl::Topology::Ring),
            nb::arg("num_workers_per_link") = nb::none(),
            nb::arg("num_buffers_per_channel") = nb::none(),
            nb::arg("mm_cores_y") = nb::none(),
            nb::arg("mm_block_ht") = nb::none(),
            nb::arg("mm_block_wt") = nb::none()});
}

}  // namespace
