// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/ternary/common/ternary_op_types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

namespace ttnn::operations::ternary {

struct TernaryDeviceOperation {
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {
        TernaryOpType ternary_op_type;
        TernaryVariant ternary_variant;
        TernaryBroadcastType broadcast_type;
        tt::tt_metal::MemoryConfig memory_config;
        DataType input_dtype;
        const CoreRangeSet worker_grid;
        std::optional<DataType> dtype;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;
        std::optional<CoreRangeSet> sub_core_grids;

        std::optional<ScalarVariant> scalar_input_a;
        std::optional<ScalarVariant> scalar_input_b;

        ttsl::hash::hash_t to_hash() const;

        DataType get_dtype() const;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        std::optional<Tensor> input_tensor_b;
        std::optional<Tensor> input_tensor_c;
        std::optional<Tensor> optional_output_tensor;
    };

    struct TernaryProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        // Cache-hit re-apply of ALL per-core runtime args, IN PLACE on the cached program -- no
        // descriptor rebuild and no kernel recompile, but the work split and every shape-dependent
        // arg (strides, D/N/C/Ht/Wt, per-core tile counts, start ids, freq/counter, packed scalar,
        // buffer addresses) ARE re-derived for the current tensors, via the same shared builder
        // create_descriptor() uses.  compute_program_hash coarsens each input to its padded volume,
        // so one cached program is shared across shapes whose dims differ but multiply to the same
        // product; re-applying only addresses left the rest frozen at the first-miss shape and
        // silently corrupted the result (issue #54235).  Sharded, tensor-backed CBs are re-pointed
        // here too.  See the .cpp.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<TernaryProgramFactory>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
};

}  // namespace ttnn::operations::ternary

namespace ttnn::prim {

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    ttnn::operations::ternary::ScalarVariant scalar,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    const Tensor& input_b,
    float scalar_c,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

ttnn::operations::ternary::TernaryDeviceOperation::tensor_return_value_t ternary(
    ttnn::operations::ternary::TernaryOpType op_type,
    const Tensor& input_a,
    float scalar_b,
    const Tensor& input_c,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::prim
