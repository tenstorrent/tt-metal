// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "bcast_device_operation_types.hpp"
#include "bcast_multi_core_h_program_factory.hpp"
#include "bcast_sharded_h_program_factory.hpp"
#include "bcast_sharded_h_optimised_program_factory.hpp"
#include "bcast_multi_core_w_program_factory.hpp"
#include "bcast_multi_core_hw_program_factory.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <optional>
#include <variant>
#include <vector>

namespace ttnn::prim {

struct BcastDeviceOperation {
    using operation_attributes_t = BcastParams;
    using tensor_args_t = BcastInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        BcastMultiCoreHProgramFactory,
        BcastShardedHProgramFactory,
        BcastShardedHOptimisedProgramFactory,
        BcastMultiCoreWProgramFactory,
        BcastMultiCoreHWProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    // Opts the op into the descriptor fast-path (no create_descriptor() rebuild on a cache hit).
    // Every factory binds all per-dispatch tensor addresses as CB `.buffer` or Buffer* rt-args
    // (multi-core h/hw/w bind src0/src1/dst; the sharded-h factories bind src0/dst via CB and src1
    // via a Buffer* rt-arg at reader arg index 0). All other runtime args are shape/geometry-derived
    // and covered by the program hash, so there is nothing to re-apply here.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&,
        const std::optional<ttnn::MeshCoordinate>& = std::nullopt);
};

}  // namespace ttnn::prim

namespace ttnn::prim {
BcastDeviceOperation::tensor_return_value_t bcast(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttnn::BcastOpMath bcast_op,
    ttnn::BcastOpDim bcast_dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool in_place,
    const std::optional<Tensor>& preallocated_output);
}  // namespace ttnn::prim
