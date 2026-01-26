// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "selective_reduce_combine_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "selective_reduce_combine.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::ccl::moe {

void bind_selective_reduce_combine(nb::module_& mod) {
    const auto* doc = R"doc()doc";

    using OperationType = decltype(ttnn::selective_reduce_combine);
    ttnn::bind_registered_operation(
        mod,
        ttnn::selective_reduce_combine,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& dense_input_tensor,
               const ttnn::Tensor& dense_metadata_tensor,
               const ttnn::Tensor& dense_token_counts_tensor,
               const uint32_t hidden_size,
               const uint32_t batch_size,
               const uint32_t seq_size,
               const uint32_t select_experts_k,
               const uint32_t experts,
               const std::optional<uint32_t>& axis,
               tt::tt_fabric::Topology topology,
               const uint32_t num_links,
               const uint32_t num_token_parallel_cores,
               const uint32_t num_data_parallel_cores,
               const CoreRangeSet worker_core_range_set,
               const CoreRangeSet mux_core_range_set,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& optional_output_tensor) {
                return self(
                    dense_input_tensor,
                    dense_metadata_tensor,
                    dense_token_counts_tensor,
                    hidden_size,
                    batch_size,
                    seq_size,
                    select_experts_k,
                    experts,
                    axis,
                    topology,
                    num_links,
                    num_token_parallel_cores,
                    num_data_parallel_cores,
                    worker_core_range_set,
                    mux_core_range_set,
                    memory_config,
                    optional_output_tensor);
            },
            nb::arg("dense_input_tensor").noconvert(),
            nb::arg("dense_metadata_tensor").noconvert(),
            nb::arg("dense_token_counts_tensor").noconvert(),
            nb::arg("hidden_size"),
            nb::arg("batch_size"),
            nb::arg("seq_size"),
            nb::arg("select_experts_k"),
            nb::arg("experts"),
            nb::arg("cluster_axis"),
            nb::arg("topology"),
            nb::arg("num_links"),
            nb::arg("num_token_parallel_cores"),
            nb::arg("num_data_parallel_cores"),
            nb::arg("worker_core_range_set").noconvert(),
            nb::arg("mux_core_range_set").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none()});
}

}  // namespace ttnn::operations::ccl::moe
