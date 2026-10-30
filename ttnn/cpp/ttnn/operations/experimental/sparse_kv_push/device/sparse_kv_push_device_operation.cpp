// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_device_operation.hpp"

#include <tt-metalium/math.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"
#include "sparse_kv_push_common.hpp"
#include "kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

namespace {

constexpr uint32_t kRowDim = 2;  // the cache is sharded over its rows across every chip

void validate_semaphores(const SparseKvPushParams& params, const SparseKvPushInputs& tensor_args) {
    TT_FATAL(
        params.ready_semaphore.has_value() && params.data_valid_semaphore.has_value(),
        "the sparse KV push needs caller-owned ready and data_valid semaphores");
    const auto& ready = *params.ready_semaphore;
    const auto& data_valid = *params.data_valid_semaphore;
    TT_FATAL(ready.address() != data_valid.address(), "the sparse KV push's two semaphores must differ");
    auto* mesh_device = tensor_args.input_tensor.device();
    TT_FATAL(
        ready.device() == mesh_device && data_valid.device() == mesh_device,
        "the sparse KV push's semaphores must be created on the input tensor's mesh device");
    const auto ready_buffer_type = std::get<1>(ready.attribute_values());
    const auto data_valid_buffer_type = std::get<1>(data_valid.attribute_values());
    const bool has_l1_small = mesh_device->allocator()->get_bank_size(BufferType::L1_SMALL) > 0;
    const auto expected_buffer_type = has_l1_small ? BufferType::L1_SMALL : BufferType::L1;
    TT_FATAL(
        ready_buffer_type == expected_buffer_type && data_valid_buffer_type == expected_buffer_type,
        "the sparse KV push's semaphores must use buffer type {}, got ready={} and data_valid={}",
        expected_buffer_type,
        ready_buffer_type,
        data_valid_buffer_type);
    // The kernels use them on every push core; elsewhere their address is unallocated L1.
    // attribute_values() returns its tuple by value: copy, don't bind a reference to the temporary.
    const CoreRangeSet ready_cores = std::get<0>(ready.attribute_values());
    const CoreRangeSet data_valid_cores = std::get<0>(data_valid.attribute_values());
    TT_FATAL(
        ready_cores.contains(params.core_grid) && data_valid_cores.contains(params.core_grid),
        "the sparse KV push's semaphores must cover its core grid {}, got ready={} and data_valid={}",
        params.core_grid,
        ready_cores,
        data_valid_cores);
}

void validate_batch_slot(const SparseKvPushParams& params, const SparseKvPushInputs& tensor_args) {
    const uint32_t cache_batch = tensor_args.input_tensor.padded_shape()[0];
    if (tensor_args.batch_index_tensor.has_value()) {
        TT_FATAL(
            params.batch_slot_layer_idx < params.batch_slot_num_layers,
            "sparse KV push batch_slot_layer_idx {} must be < batch_slot_num_layers {}",
            params.batch_slot_layer_idx,
            params.batch_slot_num_layers);
        TT_FATAL(
            params.batch_slot_num_layers <= cache_batch && cache_batch % params.batch_slot_num_layers == 0,
            "sparse KV push batch_slot_num_layers {} must divide the cache batch {}",
            params.batch_slot_num_layers,
            cache_batch);
    } else {
        TT_FATAL(
            params.batch_index < cache_batch,
            "sparse KV push batch_index {} must be < cache batch {}",
            params.batch_index,
            cache_batch);
    }
}

void validate_batch_index_tensor(const Tensor& batch_index_tensor) {
    TT_FATAL(
        batch_index_tensor.storage_type() == StorageType::DEVICE && batch_index_tensor.buffer() != nullptr,
        "sparse KV push batch_index_tensor must be allocated on device");
    TT_FATAL(
        batch_index_tensor.dtype() == DataType::UINT32 && batch_index_tensor.layout() == Layout::ROW_MAJOR &&
            batch_index_tensor.logical_volume() == 1 &&
            batch_index_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            batch_index_tensor.memory_config().buffer_type() == BufferType::DRAM,
        "sparse KV push batch_index_tensor must be a 1-element UINT32 ROW_MAJOR interleaved DRAM tensor");
}

// Row-major DRAM, one row per page. Each packet is one contiguous write into one bank of an interleaved output; the
// input is read through a tensor accessor, so it may also be sharded (the model's cache is ND-sharded).
void validate_rows(const Tensor& tensor, const char* name, bool interleaved) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "sparse KV push {} must be allocated on device",
        name);
    const auto& shape = tensor.padded_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[1] == 1, "sparse KV push {} must be [batch, 1, rows, width], got {}", name, shape);
    const auto& memory_config = tensor.memory_config();
    TT_FATAL(
        tensor.layout() == Layout::ROW_MAJOR && memory_config.buffer_type() == BufferType::DRAM,
        "sparse KV push {} must be a ROW_MAJOR DRAM tensor",
        name);
    TT_FATAL(
        !interleaved || memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "sparse KV push {} must be interleaved, got {}",
        name,
        memory_config);
}

// What the program hash leaves out, checked on every dispatch: buffers (their addresses are patched in), the devices
// they live on and the mesh distribution (neither is part of a tensor spec), the semaphores and the cache slot.
void validate_runtime(const SparseKvPushParams& params, const SparseKvPushInputs& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    TT_FATAL(input.buffer() != nullptr, "sparse KV push input must be allocated on device");
    TT_FATAL(tensor_args.output_tensor.buffer() != nullptr, "sparse KV push output must be allocated on device");
    TT_FATAL(tensor_args.table.buffer() != nullptr, "sparse KV push table must be allocated on device");
    TT_FATAL(
        !tensor_args.batch_index_tensor.has_value() || tensor_args.batch_index_tensor->buffer() != nullptr,
        "sparse KV push batch_index_tensor must be allocated on device");
    auto* mesh_device = input.device();
    TT_FATAL(
        tensor_args.output_tensor.device() == mesh_device && tensor_args.table.device() == mesh_device &&
            (!tensor_args.batch_index_tensor.has_value() || tensor_args.batch_index_tensor->device() == mesh_device),
        "sparse KV push tensors must be on one mesh device");
    const uint32_t num_devices = input.device()->shape().mesh_size();
    TT_FATAL(
        ttnn::operations::ccl::common::has_row_major_mesh_coordinates(input),
        "the sparse KV push requires row-major tensor mesh coordinates");
    TT_FATAL(
        ttnn::operations::ccl::common::tensor_dim_shard_factor(input, kRowDim) == num_devices,
        "the sparse KV push requires the input to be sharded over dim {} across all {} devices",
        kRowDim,
        num_devices);
    validate_semaphores(params, tensor_args);
    validate_batch_slot(params, tensor_args);
}

void validate_config(const SparseKvPushParams& params, const SparseKvPushInputs& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = tensor_args.output_tensor;
    const auto& table = tensor_args.table;
    validate_rows(input, "input", /*interleaved=*/false);
    validate_rows(output, "output", /*interleaved=*/true);
    auto* mesh_device = input.device();
    const auto mesh_shape = mesh_device->shape();
    const uint32_t num_devices = mesh_shape.mesh_size();
    TT_FATAL(mesh_shape.dims() == 2, "the sparse KV push needs a 2D device mesh, got {}", mesh_shape);
    TT_FATAL(
        num_devices <= ::sparse_push::plan_table::kMaxDevices,
        "the plan table holds at most {} owners, the mesh has {}",
        ::sparse_push::plan_table::kMaxDevices,
        num_devices);
    TT_FATAL(
        params.fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY,
        "the sparse KV push needs a 2D torus fabric (FABRIC_2D_TORUS_XY), got {}",
        params.fabric_config);
    TT_FATAL(
        params.num_links >= 1 && params.num_links <= 2,
        "the sparse KV push supports 1 or 2 links, got {}",
        params.num_links);
    TT_FATAL(
        params.core_grid.num_cores() >= cores_needed(params.num_links),
        "the sparse KV push needs {} cores on {} links, core_grid has {}",
        cores_needed(params.num_links),
        params.num_links,
        params.core_grid.num_cores());
    validate_stripe_chunk(params.stripe_chunk, params.max_rows);

    if (tensor_args.batch_index_tensor.has_value()) {
        validate_batch_index_tensor(*tensor_args.batch_index_tensor);
    }

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    // A chip owns every num_devices-th block of the cache, so its share of a max_rows cache is at most this many rows.
    const uint32_t local_rows = tt::div_up(params.max_rows / params.stripe_chunk, num_devices) * params.stripe_chunk;
    TT_FATAL(
        input_shape[2] >= local_rows,
        "sparse KV push input has {} rows per chip, fewer than the {} a {}-row cache puts on a chip",
        input_shape[2],
        local_rows,
        params.max_rows);
    TT_FATAL(
        output.dtype() == input.dtype() && output.tensor_spec().page_config() == input.tensor_spec().page_config(),
        "sparse KV push output dtype and page config must match the input");
    TT_FATAL(
        output_shape[0] == 1 && output_shape[3] == input_shape[3],
        "sparse KV push output must be [1, 1, rows, {}], got {}",
        input_shape[3],
        output_shape);
    TT_FATAL(
        output.buffer()->num_pages() == output_shape[2], "sparse KV push output needs one page per compact-buffer row");
    const uint32_t page_bytes = input.buffer()->aligned_page_size();
    TT_FATAL(
        output.buffer()->aligned_page_size() == page_bytes,
        "sparse KV push input and output row pages differ ({} vs {})",
        page_bytes,
        output.buffer()->aligned_page_size());
    // The exact need is sum_o round_up(count_o, unit) over the requester's per-owner counts, known only on device.
    const uint32_t unit = packet_shape(mesh_device, page_bytes).unit();
    TT_FATAL(
        output_shape[2] >= num_devices * unit,
        "sparse KV push output needs at least {} rows (a {}-row bank-lane unit per owner), got {}",
        num_devices * unit,
        unit,
        output_shape[2]);

    const auto& table_shape = table.padded_shape();
    const auto layout = PlanTableLayout::make(params.max_rows, params.stripe_chunk);
    TT_FATAL(
        table.storage_type() == StorageType::DEVICE && table.buffer() != nullptr,
        "sparse KV push table must be allocated on device");
    TT_FATAL(
        table.dtype() == DataType::UINT32 && table.layout() == Layout::ROW_MAJOR &&
            table.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            table.memory_config().buffer_type() == BufferType::DRAM,
        "sparse KV push table must be a UINT32 ROW_MAJOR interleaved DRAM tensor");
    TT_FATAL(
        table_shape.rank() == 4 && table_shape[0] == 1 && table_shape[1] == 1 &&
            table_shape[2] == layout.pages * num_devices && table_shape[3] == PlanTableLayout::kPageWords,
        "sparse KV push table must be [1, 1, {}, {}] ({} pages per device), got {}",
        layout.pages * num_devices,
        PlanTableLayout::kPageWords,
        layout.pages,
        table_shape);

    validate_runtime(params, tensor_args);
}

ttsl::hash::hash_t program_hash(const SparseKvPushParams& params, const SparseKvPushInputs& tensor_args) {
    // Routes are resolved from the fabric node ids, so they are part of the key.
    auto* mesh_device = tensor_args.input_tensor.device();
    std::vector<tt::tt_fabric::FabricNodeId> fabric_nodes;
    fabric_nodes.reserve(mesh_device->shape().mesh_size());
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape())) {
        fabric_nodes.push_back(mesh_device->get_fabric_node_id(coord));
    }
    const auto batch_index_spec =
        tensor_args.batch_index_tensor.has_value()
            ? std::optional<tt::tt_metal::TensorSpec>(tensor_args.batch_index_tensor->tensor_spec())
            : std::nullopt;
    // The output's row count is a compile-time arg (the reader's write bound), so its whole spec is keyed.
    return ttsl::hash::hash_objects_with_default_seed(
        params.stripe_chunk,
        params.max_rows,
        params.part,
        params.num_links,
        params.core_grid,
        params.fabric_config,
        mesh_device->shape(),
        fabric_nodes,
        tensor_args.input_tensor.tensor_spec(),
        tensor_args.output_tensor.tensor_spec(),
        tensor_args.table.tensor_spec(),
        batch_index_spec);
}

}  // namespace

ttsl::hash::hash_t SparseKvPushDeviceOperation::compute_program_hash(
    const SparseKvPushParams& args, const SparseKvPushInputs& tensor_args) {
    return tt::tt_metal::operation::hash_operation<SparseKvPushDeviceOperation>(program_hash(args, tensor_args));
}

void SparseKvPushDeviceOperation::validate_on_program_cache_miss(
    const SparseKvPushParams& args, const SparseKvPushInputs& tensor_args) {
    validate_config(args, tensor_args);
}

void SparseKvPushDeviceOperation::validate_on_program_cache_hit(
    const SparseKvPushParams& args, const SparseKvPushInputs& tensor_args) {
    validate_runtime(args, tensor_args);
}

SparseKvPushDeviceOperation::spec_return_value_t SparseKvPushDeviceOperation::compute_output_specs(
    const SparseKvPushParams&, const SparseKvPushInputs& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}

SparseKvPushDeviceOperation::topology_return_value_t SparseKvPushDeviceOperation::compute_output_topologies(
    const SparseKvPushParams&, const SparseKvPushInputs& tensor_args) {
    // Each chip's compact buffer holds the rows its own top-k needs, so the buffers differ: like the input, the output
    // is sharded over its rows across every chip.
    return {tensor_args.input_tensor.tensor_topology()};
}

SparseKvPushDeviceOperation::tensor_return_value_t SparseKvPushDeviceOperation::create_output_tensors(
    const SparseKvPushParams&, const SparseKvPushInputs& tensor_args) {
    return tensor_args.output_tensor;
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
    ::sparse_push::PushPart part,
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
