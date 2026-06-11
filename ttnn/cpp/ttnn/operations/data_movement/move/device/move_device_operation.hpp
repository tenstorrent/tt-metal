// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>
#include "ttnn/tensor/tensor.hpp"
#include "move_device_operation_types.hpp"
#include "move_program_factory.hpp"
#include "move_overlap_program_factory.hpp"
#include "move_sharded_program_factory.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

namespace ttnn::prim {

struct MoveDeviceOperation {
    // Type aliases
    using operation_attributes_t = ttnn::prim::MoveOperationAttributes;
    using tensor_args_t = ttnn::prim::MoveTensorArgs;
    using tensor_return_value_t = Tensor;
    using spec_return_value_t = ttnn::TensorSpec;

    using program_factory_t = std::variant<MoveProgramFactory, MoveOverlapProgramFactory, MoveShardedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    // Sharded in-place move bakes address-DERIVED reader args (chunk size = output_addr - input_addr,
    // and num_chunks/remainder from it) that change whenever the buffers are re-allocated. They cannot
    // be expressed as a Buffer* binding, so we re-derive and re-apply them on every cache hit here;
    // the src/dst CB `.buffer` bindings are patched by the framework. Declaring this opts the op into
    // the descriptor fast-path (no create_descriptor() rebuild). Returns empty for the non-sharded
    // strategies (they already fast-path via Buffer* rt-arg bindings).
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim

namespace ttnn::prim {
ttnn::prim::MoveDeviceOperation::tensor_return_value_t move(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const ttnn::prim::MoveOpParallelizationStrategy& move_op_parallelization_strategy);
}  // namespace ttnn::prim
