// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include "ttnn/tensor/tensor.hpp"
#include "move_codegen_program_factory.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

// Codegen-backed counterpart to MoveDeviceOperation (device/move_device_operation.hpp). Only the
// non-sharded interleaved TILE/ROW_MAJOR paths are in scope (see manifests/move.yaml coverage);
// validate rejects everything else via supported_by_codegen().
struct MoveCodegenDeviceOperation {
    // Type aliases
    using operation_attributes_t = ttnn::prim::MoveCodegenOperationAttributes;
    using tensor_args_t = ttnn::prim::MoveCodegenTensorArgs;
    using tensor_return_value_t = Tensor;
    using spec_return_value_t = ttnn::TensorSpec;

    using program_factory_t = std::variant<MoveCodegenProgramFactory>;

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
};

}  // namespace ttnn::prim

namespace ttnn::prim {
ttnn::prim::MoveCodegenDeviceOperation::tensor_return_value_t move_codegen(
    const Tensor& input_tensor, const Tensor& output_tensor, const tt::tt_metal::MemoryConfig& output_mem_config);
}  // namespace ttnn::prim
