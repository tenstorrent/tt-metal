// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compressor_state_exchange.hpp"

#include <array>
#include <optional>
#include <string_view>

#include "device/compressor_state_select.hpp"
#include "tt-metalium/experimental/fabric/fabric.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/point_to_point/point_to_point.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::compressor_state_exchange {
namespace {

constexpr uint32_t kStateRows = 64;
constexpr uint32_t kHeadDim = 512;

void validate_state(const ttnn::Tensor& tensor, std::string_view name) {
    TT_FATAL(tensor.storage_type() == ttnn::StorageType::DEVICE, "{} must be a device tensor", name);
    TT_FATAL(tensor.device() != nullptr, "{} must have an associated mesh device", name);
    TT_FATAL(tensor.dtype() == ttnn::DataType::BFLOAT16, "{} must use BFLOAT16", name);
    TT_FATAL(tensor.layout() == tt::tt_metal::Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(!tensor.is_sharded(), "{} must use interleaved memory", name);
    TT_FATAL(tensor.logical_shape().rank() == 4, "{} must be rank 4", name);
    TT_FATAL(tensor.logical_shape()[-1] == kHeadDim, "{} must have head dimension {}", name, kHeadDim);
}

void validate_pair(
    const ttnn::Tensor& local_state,
    const ttnn::Tensor& initial_state,
    std::string_view local_name,
    std::string_view initial_name) {
    validate_state(local_state, local_name);
    validate_state(initial_state, initial_name);
    TT_FATAL(
        local_state.device() == initial_state.device(),
        "{} and {} must be on the same mesh device",
        local_name,
        initial_name);
    TT_FATAL(
        local_state.logical_shape() == initial_state.logical_shape(),
        "{} and {} must have identical logical shapes",
        local_name,
        initial_name);
    TT_FATAL(
        local_state.tensor_spec() == initial_state.tensor_spec(),
        "{} and {} must have identical tensor specs",
        local_name,
        initial_name);
}

ttnn::Tensor shift_state(
    const ttnn::Tensor& local_state,
    const ttnn::Tensor& initial_state,
    uint32_t cluster_axis,
    ::ttnn::ccl::Topology topology) {
    if (tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig())) {
        auto gathered_state = ttnn::all_gather(
            local_state,
            /*dim=*/2,
            cluster_axis,
            local_state.memory_config());
        return ttnn::prim::compressor_state_select(gathered_state, initial_state, cluster_axis);
    }

    auto output = ttnn::clone(
        initial_state,
        /*dtype=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt);
    const auto mesh_shape = local_state.device()->shape();
    const uint32_t sp_factor = mesh_shape[cluster_axis];
    const uint32_t lanes = mesh_shape[1 - cluster_axis];

    for (uint32_t receiver_rank = 1; receiver_rank < sp_factor; ++receiver_rank) {
        const uint32_t sender_rank = receiver_rank - 1;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const std::array<uint32_t, 2> sender = cluster_axis == 0 ? std::array<uint32_t, 2>{sender_rank, lane}
                                                                     : std::array<uint32_t, 2>{lane, sender_rank};
            const std::array<uint32_t, 2> receiver = cluster_axis == 0 ? std::array<uint32_t, 2>{receiver_rank, lane}
                                                                       : std::array<uint32_t, 2>{lane, receiver_rank};
            output = ttnn::point_to_point(
                local_state,
                ttnn::MeshCoordinate{receiver[0], receiver[1]},
                ttnn::MeshCoordinate{sender[0], sender[1]},
                topology,
                output);
        }
    }
    return output;
}

}  // namespace

std::tuple<ttnn::Tensor, ttnn::Tensor> compressor_state_exchange(
    const ttnn::Tensor& local_kv_state,
    const ttnn::Tensor& local_score_state,
    const ttnn::Tensor& initial_kv_state,
    const ttnn::Tensor& initial_score_state,
    uint32_t cluster_axis,
    ::ttnn::ccl::Topology topology) {
    validate_pair(local_kv_state, initial_kv_state, "local_kv_state", "initial_kv_state");
    validate_pair(local_score_state, initial_score_state, "local_score_state", "initial_score_state");
    TT_FATAL(
        local_kv_state.tensor_spec() == local_score_state.tensor_spec(),
        "KV and score states must have identical tensor specs");
    TT_FATAL(
        local_kv_state.device() == local_score_state.device(), "KV and score states must be on the same mesh device");

    const auto mesh_shape = local_kv_state.device()->shape();
    TT_FATAL(mesh_shape.dims() == 2, "compressor_state_exchange requires a 2D mesh, got {}", mesh_shape);
    TT_FATAL(cluster_axis < 2, "cluster_axis must be 0 or 1, got {}", cluster_axis);
    TT_FATAL(
        local_kv_state.logical_shape()[-2] == kStateRows,
        "Each device's local state must contain exactly {} rows",
        kStateRows);

    auto predecessor_kv = shift_state(local_kv_state, initial_kv_state, cluster_axis, topology);
    auto predecessor_score = shift_state(local_score_state, initial_score_state, cluster_axis, topology);
    return {predecessor_kv, predecessor_score};
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::compressor_state_exchange
