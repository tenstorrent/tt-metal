// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_program_factory.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

namespace {

// The push is the program's only content, so its kernels come first.
constexpr sparse_push::Kernels kKernels{.reader = 0, .writer = 1};

}  // namespace

sparse_push::PushConfig push_config(const SparseKvPushParams& params, const SparseKvPushInputs& tensor_args) {
    return sparse_push::PushConfig{
        .input = tensor_args.input_tensor,
        .output = tensor_args.output_tensor,
        .table = tensor_args.table,
        .batch_index_tensor = tensor_args.batch_index_tensor,
        .stripe_chunk = params.stripe_chunk,
        .max_rows = params.max_rows,
        .part = params.part,
        .num_links = params.num_links,
        .core_grid = params.core_grid,
        .fabric_config = params.fabric_config,
        .ready_semaphore = params.ready_semaphore,
        .data_valid_semaphore = params.data_valid_semaphore,
        .batch_index = params.batch_index,
        .batch_slot_num_layers = params.batch_slot_num_layers,
        .batch_slot_layer_idx = params.batch_slot_layer_idx};
}

tt::tt_metal::ProgramDescriptor SparseKvPushProgramFactory::create_descriptor(
    const SparseKvPushParams& operation_attributes,
    const SparseKvPushInputs& tensor_args,
    Tensor& /*output_tensor*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "sparse_kv_push programs are per device");
    tt::tt_metal::ProgramDescriptor desc;
    const auto kernels =
        sparse_push::append(desc, push_config(operation_attributes, tensor_args), *mesh_dispatch_coordinate);
    TT_FATAL(
        kernels.reader == kKernels.reader && kernels.writer == kKernels.writer, "sparse_kv_push kernel order changed");
    return desc;
}

void SparseKvPushProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SparseKvPushParams& operation_attributes,
    const SparseKvPushInputs& tensor_args,
    Tensor& /*output_tensor*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    sparse_push::override_runtime_arguments(program, kKernels, push_config(operation_attributes, tensor_args));
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
