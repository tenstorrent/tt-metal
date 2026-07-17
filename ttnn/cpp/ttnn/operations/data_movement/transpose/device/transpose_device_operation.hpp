// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"
#include "transpose_cn_program_factory.hpp"
#include "transpose_hc_rm_program_factory.hpp"
#include "transpose_hc_sharded_program_factory.hpp"
#include "transpose_hc_tiled_interleaved_program_factory.hpp"
#include "transpose_hc_tiled_program_factory.hpp"
#include "transpose_wh_program_factory.hpp"
#include "transpose_wh_sharded_program_factory.hpp"
#include "transpose_wh_sharded_rm_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <variant>
#include "ttnn/operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

namespace ttnn::prim {

struct TransposeDeviceOperation {
    using operation_attributes_t = TransposeParams;
    using tensor_args_t = TransposeInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        TransposeWHProgramFactory,
        TransposeWHShardedProgramFactory,
        TransposeWHShardedRMProgramFactory,
        TransposeHCTiledInterleavedProgramFactory,
        TransposeHCTiledProgramFactory,
        TransposeHCRMProgramFactory,
        TransposeHCShardedProgramFactory,
        TransposeCNProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static TransposeOpParallelizationStrategy get_parallelization_strategy(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, const Tensor& output);

    // Cache-hit re-apply of all per-dispatch state (per-core args + tensor-backed CB/buffer addresses)
    // from the same factory the miss path picks. See transpose_device_operation.cpp.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim

namespace ttnn::prim {
ttnn::Tensor transpose(
    const Tensor& input_tensor,
    ttnn::prim::TransposeOpDim dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    float pad_value = 0.0f);
}  // namespace ttnn::prim
