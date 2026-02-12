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

namespace ttnn::operations::experimental::ccl::moe {

void bind_selective_reduce_combine(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Selective reduce combine operation sparsifies dense blocks of tokens coming out of MoE compute megakernel and sends tokens back to their originating devices.
        B = local batch size/batch size per device
        S = local sequence length/sequence length per device
        H = hidden size
        K = selected experts per token
        D = total number of devices
        A = cluster axis to dispatch along
        D[A] = number of devices along the cluster axis, just D if cluster axis is not specified.
        E = local experts/experts per device
        T = total number of tokens per device = B * S
        P = Fabric max packet size in BF16 elements
        Args:
            dense_input_tensor (ttnn.Tensor): Dense expert contributions from MoE compute. It is expected to be structured as follows
                L1 tensor
                Hidden padded up to fabric max packet size, only last shard has padding
                shape: [`data_parallel_core_dim`, `token_parallel_core_dim`, E, T/token_parallel_core_dim, H[P]/num_data_parallel_dim]

                Height sharded,
                shard shape: [local_experts * T / token_parallel_core_dim, H[P] / num_data_parallel_dim]


                Generally expect num_data_parallel_dim=num_token_parallel_dim=4
                This can be though of as a logical grid by a physical grid is not required as long as the ordering is
                consistent with the cores provided by `worker_core_range_set`

                                        Tokens split (num_data_parallel_dim)
                                               _________
                tokens Distributed             |C C C C|
                evenly as possible             |C C C C|
                (token_parallel_core_dim)      |C C C C|
                                               |C C C C|
                                               ---------

                Active tokens are distributed evenly among the height shards, earlier shards get remainders.

                Example. E0: 6 active tokens, E1: 3 active tokens
                ---------------------------------
                Shard 0,0:     E0T0S0
                               E0T1S0
                               E0T2S0
                               000000
                               000000
                               ...
                               Batch/token_parallel_core_dim rows
                               E1T0S0
                               E1T1S0
                               000000
                               000000
                               ......
                               batch/token_parallel_core_dim rows
                ----------------------------------
                Shard 1,0:     E0T3S0
                               E0T4S0
                               E0T5S0
                               000000
                               000000
                               ...
                               Batch/token_parallel_core_dim rows
                               E1T2S0
                               000000
                               000000
                               000000
                               ......
                               batch/token_parallel_core_dim rows
                ----------------------------------


            dense_metadata_tensor (ttnn.Tensor): Metadata tensor provided by all_to_all_dispatch_selective_tilize
            dense_token_maps_tensor (ttnn.Tensor): Sparse output tokens tensor provided by all_to_all_dispatch_selective_tilize
            dense_token_counts_tensor (ttnn.Tensor): Active token counts tensor all_to_all_dispatch_selective_tilize
            hidden_size (int): H
            batch_size (int): B*D[A]
            seq_size (int): S
            select_experts_k (int): K
            experts (int) E*D[A]
            axis (int): A
            topology (ttnn.Topology): Line or Ring supported
            num_links (int): Number of fabric links to utilize
            token_parallel_core_dim (int): token shard dimension (described above)
            data_parallel_core_dim (int): hidden dim shard dimension (described above)
            worker_core_range_set (ttnn.CoreRangeSet): Available cores for running op, should be consistent with dense_input_tensor sharding
            mux_core_range_set (ttnn.CoreRangeSet): Available cores for mux workers. Should have no overlap with `worker_core_range_set`

        Keyword Args:
            memory_config (ttnn.memory_config): optional output mem config
            output_tensor (ttnn.Tensor): Optional preallocated output tensor
            optional_cross_device_semaphore (ttnn.GlobalSemaphore): Optional preallocated signaling semaphore

        Returns:
            (ttnn.Tensor) [T*D[A], E*D[A], H] sparse output tensor.

        )doc";

    using OperationType = decltype(ttnn::selective_reduce_combine);
    ttnn::bind_registered_operation(
        mod,
        ttnn::selective_reduce_combine,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& dense_input_tensor,
               const ttnn::Tensor& dense_metadata_tensor,
               const ttnn::Tensor& dense_token_maps_tensor,
               const ttnn::Tensor& dense_token_counts_tensor,
               const uint32_t hidden_size,
               const uint32_t batch_size,
               const uint32_t seq_size,
               const uint32_t select_experts_k,
               const uint32_t experts,
               const std::optional<uint32_t>& axis,
               tt::tt_fabric::Topology topology,
               const uint32_t num_links,
               const uint32_t token_parallel_core_dim,
               const uint32_t data_parallel_core_dim,
               const CoreRangeSet& worker_core_range_set,
               const CoreRangeSet& mux_core_range_set,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& optional_output_tensor,
               const std::optional<GlobalSemaphore>& optional_cross_device_semaphore) {
                return self(
                    dense_input_tensor,
                    dense_metadata_tensor,
                    dense_token_maps_tensor,
                    dense_token_counts_tensor,
                    hidden_size,
                    batch_size,
                    seq_size,
                    select_experts_k,
                    experts,
                    axis,
                    topology,
                    num_links,
                    token_parallel_core_dim,
                    data_parallel_core_dim,
                    worker_core_range_set,
                    mux_core_range_set,
                    memory_config,
                    optional_output_tensor,
                    optional_cross_device_semaphore);
            },
            nb::arg("dense_input_tensor").noconvert(),
            nb::arg("dense_metadata_tensor").noconvert(),
            nb::arg("dense_token_maps_tensor").noconvert(),
            nb::arg("dense_token_counts_tensor").noconvert(),
            nb::arg("hidden_size"),
            nb::arg("batch_size"),
            nb::arg("seq_size"),
            nb::arg("select_experts_k"),
            nb::arg("experts"),
            nb::arg("cluster_axis"),
            nb::arg("topology"),
            nb::arg("num_links"),
            nb::arg("token_parallel_core_dim"),
            nb::arg("data_parallel_core_dim"),
            nb::arg("worker_core_range_set").noconvert(),
            nb::arg("mux_core_range_set").noconvert(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("optional_cross_device_semaphore") = nb::none()});
}

}  // namespace ttnn::operations::experimental::ccl::moe
