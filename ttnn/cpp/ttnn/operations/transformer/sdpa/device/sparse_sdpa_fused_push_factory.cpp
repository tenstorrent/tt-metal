// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The next layer's prefix rows (part 1) are pushed on push_core_grid while attention runs; the next layer pushes
// the rest (part 2) itself, since those rows are only written by this chunk.

#include "ttnn/operations/transformer/sdpa/device/sparse_sdpa_device_operation.hpp"
#include "ttnn/operations/experimental/sparse_kv_push/device/sparse_push_program.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

namespace {

namespace sparse_push = ttnn::operations::experimental::sparse_kv_push::sparse_push;

// After sparse_sdpa's reader, writer and compute.
constexpr sparse_push::Kernels kPushKernels{.reader = 3, .writer = 4};
constexpr uint32_t kPrefixPart = 1;

sparse_push::PushConfig push_config(const SparseSDPAParams& attrs, const SparseSDPAInputs& t) {
    const auto& push = *attrs.fused_push;
    return sparse_push::PushConfig{
        .input = *t.push_input,
        .output = *t.push_output,
        .table = *t.push_table,
        .batch_index_tensor = t.push_batch_index_tensor,
        .stripe_chunk = push.stripe_chunk,
        .max_rows = push.max_rows,
        .part = kPrefixPart,
        .num_links = push.num_links,
        .core_grid = push.push_core_grid,
        .fabric_config = push.fabric_config,
        .ready_semaphore = push.ready_semaphore,
        .data_valid_semaphore = push.data_valid_semaphore,
        .batch_index = push.input_batch_index,
        .batch_slot_num_layers = push.batch_slot_num_layers,
        .batch_slot_layer_idx = push.batch_slot_layer_idx};
}

}  // namespace

tt::tt_metal::ProgramDescriptor SparseSDPAOperation::SparseSDPAPushFusedFactory::create_descriptor(
    const operation_attributes_t& attrs,
    const tensor_args_t& t,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "fused push programs are per device");
    auto desc = SparseSDPAProgramFactory::create_descriptor(attrs, t, output);
    const auto kernels = sparse_push::append(desc, push_config(attrs, t), *mesh_dispatch_coordinate);
    TT_FATAL(
        kernels.reader == kPushKernels.reader && kernels.writer == kPushKernels.writer,
        "sparse_sdpa kernel layout changed");
    return desc;
}

void SparseSDPAOperation::SparseSDPAPushFusedFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    SparseSDPAProgramFactory::override_runtime_arguments(
        program, operation_attributes, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
    sparse_push::override_runtime_arguments(program, kPushKernels, push_config(operation_attributes, tensor_args));
}

void SparseSDPAOperation::SparseSDPAPushFusedFactory::validate_push(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    TT_FATAL(
        t.push_input.has_value() && t.push_table.has_value() && t.push_output.has_value(),
        "sparse_sdpa fused push needs push_input, push_table and push_output");
    TT_FATAL(
        !attrs.worker_core_grid.intersects(attrs.fused_push->push_core_grid),
        "sparse_sdpa fused push needs sub_core_grids {} disjoint from push_core_grid {}",
        attrs.worker_core_grid,
        attrs.fused_push->push_core_grid);
    sparse_push::validate(push_config(attrs, t));
}

void SparseSDPAOperation::SparseSDPAPushFusedFactory::validate_push_runtime(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    sparse_push::validate_runtime(push_config(attrs, t));
}

ttsl::hash::hash_t SparseSDPAOperation::SparseSDPAPushFusedFactory::push_hash(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // Missing push tensors fail validation, which runs after the hash.
    if (!t.push_input.has_value() || !t.push_table.has_value() || !t.push_output.has_value()) {
        return 0;
    }
    return sparse_push::program_hash(push_config(attrs, t));
}

}  // namespace ttnn::prim
