// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/roll/device/roll_device_operation_types.hpp"
#include "ttnn/operations/data_movement/roll/device/roll_program_factory.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

namespace ttnn::prim {

struct RollDeviceOperation {
    using operation_attributes_t = RollParams;
    using tensor_args_t = RollInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RollShardedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    // Cache-hit fast path: re-derive per-dispatch state from create_descriptor for the current tensors
    // and re-apply to the cached program via apply_descriptor_runtime_args -- no rebuild.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&,
        const std::optional<ttnn::MeshCoordinate>& = std::nullopt);
};

// Single-dim native sharded roll. shift is normalized to [0, dim_size); dim is absolute.
RollDeviceOperation::tensor_return_value_t roll_sharded(
    const Tensor& input, uint32_t shift, int32_t dim, const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim
