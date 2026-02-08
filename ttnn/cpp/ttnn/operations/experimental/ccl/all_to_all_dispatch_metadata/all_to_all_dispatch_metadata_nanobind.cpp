// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_metadata_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/decorators.hpp"
#include "all_to_all_dispatch_metadata.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::experimental::ccl {

void bind_all_to_all_dispatch_metadata_enums(nb::module_& mod) {
    nb::enum_<DispatchAlgorithm>(mod, "DispatchAlgorithm")
        .value("BROADCAST", DispatchAlgorithm::BROADCAST, "Broadcast all tokens to ALL devices")
        .value("SPARSE_UNICAST", DispatchAlgorithm::SPARSE_UNICAST, "Send to each target device individually")
        .value("SPARSE_MCAST_LINEAR", DispatchAlgorithm::SPARSE_MCAST_LINEAR, "Sparse multicast in single direction")
        .value(
            "SPARSE_MCAST_SHORTEST_PATH",
            DispatchAlgorithm::SPARSE_MCAST_SHORTEST_PATH,
            "Sparse multicast with bidirectional shortest path routing (default)")
        .value(
            "SPARSE_MCAST_SPLIT_BW",
            DispatchAlgorithm::SPARSE_MCAST_SPLIT_BW,
            "Sparse multicast, split token data 50/50 between directions");

    nb::enum_<WorkerMode>(mod, "WorkerMode")
        .value("DIRECT", WorkerMode::DIRECT, "Direct EDM, 1 worker per link (default)")
        .value("MUX_TOKEN_SPLIT", WorkerMode::MUX_TOKEN_SPLIT, "Mux enabled, tokens distributed across workers")
        .value(
            "MUX_PAYLOAD_SPLIT",
            WorkerMode::MUX_PAYLOAD_SPLIT,
            "Workers on same link split token payload (not yet implemented)");
}

void bind_all_to_all_dispatch_metadata(nb::module_& mod) {
    // Bind the enums first
    bind_all_to_all_dispatch_metadata_enums(mod);
    const auto* doc =
        R"doc(
        All to all dispatch metadata operation for dispatching the input tokens to devices with the selected experts, based on the expert indices and expert mapping tensors. If cluster axis is specified then we dispatch the tokens to the experts only on that axis. This operation sends tokens to their selected experts, with empty rows for tokens that did not select any experts on that device.
        B = local batch size/batch size per device
        S = local sequence length/sequence length per device
        H = hidden size
        K = selected experts per token
        D = total number of devices
        A = cluster axis to dispatch along
        D[A] = number of devices along the cluster axis, just D if cluster axis is not specified.
        E = local experts/experts per device
        T = total number of tokens per device = B * S

        Args:
            input_tensor (ttnn.Tensor): The input tensor containing the tokens to dispatch. The tensor is expected to be [B, S, 1, H] per device, sharded along either the batch dimension or the sequence dimension, such that the global shape is either [B*D[A], S, 1, H] or [B, S*D[A], 1, H]. Each row is a token. The tensor is expected to be in Row Major, Interleaved format. It is duplicated on the non-cluster axis.
            expert_indices_tensor (ttnn.Tensor): The expert indices tensor containing the ranking of the experts for each token. The tensor is expected to be [B, S, 1, K] per device, sharded identically to the input_tensor. Each value in the row is an expert index, which corresponds to a row index in the expert mapping tensor. This tensor only contains the expert ranking for the tokens local to that device. The tensor is expected to be in Row Major, Interleaved format. It is duplicated on the non-cluster axis.
            expert_scores_tensor (ttnn.Tensor): The expert scores tensor containing the scores for each selected expert for each token. The tensor is expected to be [B, S, 1, K] per device, sharded identically to the expert_indices_tensor. Each value in the row is a score corresponding to the expert at the same position in the expert_indices_tensor. The tensor is expected to be in bfloat16, Row Major, Interleaved format. It is duplicated on the non-cluster axis.
            expert_mapping_tensor (ttnn.Tensor): The one-hot encoded expert to device mapping tensor containing the location of the experts among each device and each mesh. The tensor is expected to be [1, 1, E, D] per device, fully replicated across all devices. Each row corresponds to an expert, and the value in each corresponding column is 1 if the expert is on the device, 0 otherwise. The tensor is expected to be in Row Major, Interleaved format. This tensor is expected to be the same across all devices.

        Keyword Args:
            cluster_axis (int, optional): the cluster axis to dispatch along. Defaults to `None` though we assert out when it is not specified.
            num_links (number, optional): the number of cross-device links to use for dispatching the tokens. Defaults to `None`, for which the number of links is determined automatically.
            output_tensors (Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor], optional): the optional output tensors to use for the dispatched tokens, indices, and scores. Defaults to `None`.
            drain_sync_tilizer_core (Tuple[int, int], optional): the core coordinate where indices and scores are L1 sharded for selective_tilize. Defaults to `(0, 0)`.
            worker_mode (ttnn.WorkerMode, optional): the worker mode for distributing workers across links. Defaults to `ttnn.WorkerMode.DIRECT`.
            dispatch_algorithm (ttnn.DispatchAlgorithm, optional): the algorithm for routing tokens to destination devices. Defaults to `ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH`.
            worker_core_range_set (ttnn.CoreRangeSet, optional): the cores to use for dispatch workers. Defaults to cores (0,0) to (0,7) - 8 cores for 4 links.
            mux_core_range_set (ttnn.CoreRangeSet, optional): the cores to use for mux workers when worker_mode uses mux. Defaults to cores (1,0) to (1,7) - 8 cores (2 per link × 4 links).

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]: The sparse output tokens tensor, indices tensor, and scores tensor. The output tensor on each device is sparsely populated with all the tokens that are dispatched to that device. The non-dispatched tokens have placeholder rows populated with garbage. The indices and scores tensors are all-gathered and L1 sharded to the drain_sync_tilizer_core.

            output_tensor: The output tensor is expected to be [1, T*D[A], H] per device, sharded fully such that we have [D, T*D[A], H] total when gathered along dimension 0. Each row is either a token if that token was dispatched to that device, or a placeholder row if that token was not dispatched to that device. The tensor is expected to be in Row Major, Interleaved format.
            expert_indices_tensor: The indices tensor is expected to be [1, T*D[A], K] per device. Each row contains all the expert indices selected for each token on the mesh. This is equivalent to an all-gather of the expert indices. The tensor is L1 sharded to drain_sync_tilizer_core.
            expert_scores_tensor: The scores tensor has the same shape as the indices tensor and contains the corresponding scores. The tensor is L1 sharded to drain_sync_tilizer_core.

        Example:
            >>> output_tensor, indices_tensor, scores_tensor = ttnn.experimental.all_to_all_dispatch_metadata(
                            input_tensor,
                            expert_indices_tensor,
                            expert_scores_tensor,
                            expert_mapping_tensor,
                            cluster_axis=cluster_axis,
                            num_links=num_links,
                            drain_sync_tilizer_core=(0, 0),
                            worker_mode=ttnn.WorkerMode.DIRECT,
                            dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH)
        )doc";

    using OperationType = decltype(ttnn::experimental::all_to_all_dispatch_metadata);
    ttnn::bind_registered_operation(
        mod,
        ttnn::experimental::all_to_all_dispatch_metadata,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::Tensor& expert_indices_tensor,
               const ttnn::Tensor& expert_scores_tensor,
               const ttnn::Tensor& expert_mapping_tensor,
               const std::optional<uint32_t> cluster_axis,
               const std::optional<std::array<ttnn::Tensor, 3>>& output_tensors,
               const std::optional<uint32_t> num_links,
               const std::optional<std::array<uint32_t, 2>>& drain_sync_tilizer_core,
               WorkerMode worker_mode,
               DispatchAlgorithm dispatch_algorithm,
               const std::optional<CoreRangeSet>& worker_core_range_set,
               const std::optional<CoreRangeSet>& mux_core_range_set,
               const std::optional<GlobalSemaphore>& cross_device_semaphore) /*-> std::array*/ {
                std::optional<CoreCoord> drain_core = std::nullopt;
                if (drain_sync_tilizer_core.has_value()) {
                    drain_core = CoreCoord(drain_sync_tilizer_core->at(0), drain_sync_tilizer_core->at(1));
                }
                return self(
                    input_tensor,
                    expert_indices_tensor,
                    expert_scores_tensor,
                    expert_mapping_tensor,
                    cluster_axis,
                    output_tensors,
                    num_links,
                    drain_core,
                    worker_mode,
                    dispatch_algorithm,
                    worker_core_range_set,
                    mux_core_range_set,
                    cross_device_semaphore);
            },
            nb::arg("input_tensor").noconvert(),
            nb::arg("expert_indices_tensor").noconvert(),
            nb::arg("expert_scores_tensor").noconvert(),
            nb::arg("expert_mapping_tensor").noconvert(),
            nb::kw_only(),
            nb::arg("cluster_axis") = nb::none(),
            nb::arg("output_tensors") = nb::none(),
            nb::arg("num_links") = nb::none(),
            nb::arg("drain_sync_tilizer_core") = nb::none(),
            nb::arg("worker_mode") = WorkerMode::DIRECT,
            nb::arg("dispatch_algorithm") = DispatchAlgorithm::SPARSE_MCAST_SHORTEST_PATH,
            nb::arg("worker_core_range_set") = nb::none(),
            nb::arg("mux_core_range_set") = nb::none(),
            nb::arg("cross_device_semaphore") = nb::none()});
}

}  // namespace ttnn::operations::experimental::ccl
