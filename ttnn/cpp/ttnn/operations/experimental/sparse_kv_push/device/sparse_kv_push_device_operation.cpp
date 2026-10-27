// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

namespace {
constexpr uint32_t kRowDim = 2;
}  // namespace

ttsl::hash::hash_t SparseKvPushDeviceOperation::compute_program_hash(
    const SparseKvPushParams& args, const SparseKvPushInputs& tensor_args) {
    return tt::tt_metal::operation::hash_operation<SparseKvPushDeviceOperation>(
        sparse_push::program_hash(push_config(args, tensor_args)));
}

void SparseKvPushDeviceOperation::validate_on_program_cache_miss(
    const SparseKvPushParams& args, const SparseKvPushInputs& tensor_args) {
    TT_FATAL(args.part <= 2, "sparse_kv_push part must be 0 (all rows), 1 (prefix) or 2 (remainder)");
    sparse_push::validate_config(push_config(args, tensor_args));
}

void SparseKvPushDeviceOperation::validate_on_program_cache_hit(
    const SparseKvPushParams& args, const SparseKvPushInputs& tensor_args) {
    sparse_push::validate_runtime(push_config(args, tensor_args));
}

SparseKvPushDeviceOperation::spec_return_value_t SparseKvPushDeviceOperation::compute_output_specs(
    const SparseKvPushParams&, const SparseKvPushInputs& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}

SparseKvPushDeviceOperation::topology_return_value_t SparseKvPushDeviceOperation::compute_output_topologies(
    const SparseKvPushParams&, const SparseKvPushInputs& tensor_args) {
    // Like an all-gather over the rows: placements sharding dim 2 become replicated.
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_topology = input_tensor.tensor_topology();
    auto placements = input_topology.placements();
    const uint32_t rank = input_tensor.logical_shape().rank();
    for (auto& placement : placements) {
        if (ttnn::operations::ccl::common::placement_shards_tensor_dim(placement, kRowDim, rank)) {
            placement = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
    }
    return {tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(), std::move(placements), input_topology.mesh_coords())};
}

SparseKvPushDeviceOperation::tensor_return_value_t SparseKvPushDeviceOperation::create_output_tensors(
    const SparseKvPushParams&, const SparseKvPushInputs& tensor_args) {
    return tensor_args.output_tensor;
}

SparseKvPushDeviceOperation::program_factory_t SparseKvPushDeviceOperation::select_program_factory(
    const SparseKvPushParams&, const SparseKvPushInputs&) {
    return SparseKvPushProgramFactory{};
}

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::prim {

Tensor sparse_kv_push(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& ready_semaphore,
    const GlobalSemaphore& data_valid_semaphore,
    uint32_t part,
    uint32_t num_links,
    uint32_t batch_index,
    const std::optional<Tensor>& batch_index_tensor,
    uint32_t batch_slot_num_layers,
    uint32_t batch_slot_layer_idx,
    const std::optional<CoreRangeSet>& core_grid) {
    using namespace ttnn::operations::experimental::sparse_kv_push;
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "sparse_kv_push input must be on device");
    const auto grid = core_grid.value_or(mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, mesh_device->get_sub_device_ids().at(0)));
    return ttnn::device_operation::launch<SparseKvPushDeviceOperation>(
        SparseKvPushParams{
            .stripe_chunk = stripe_chunk,
            .max_rows = max_rows,
            .part = part,
            .num_links = num_links,
            .core_grid = grid,
            .fabric_config = tt::tt_fabric::GetFabricConfig(),
            .ready_semaphore = ready_semaphore,
            .data_valid_semaphore = data_valid_semaphore,
            .batch_index = batch_index,
            .batch_slot_num_layers = batch_slot_num_layers,
            .batch_slot_layer_idx = batch_slot_layer_idx},
        SparseKvPushInputs{
            .input_tensor = input_tensor,
            .output_tensor = output_tensor,
            .table = table,
            .batch_index_tensor = batch_index_tensor});
}

}  // namespace ttnn::prim
