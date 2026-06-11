// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "concat_program_factory.hpp"
#include "concat_s2s_tiled_program_factory.hpp"
#include "concat_s2s_rm_program_factory.hpp"
#include "concat_s2s_multi_program_factory.hpp"
#include "concat_s2i_program_factory.hpp"

#include "concat_device_operation_types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <optional>
#include <vector>

namespace ttnn::prim {

struct ConcatDeviceOperation {
    using operation_attributes_t = ConcatParams;
    using tensor_args_t = ConcatInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        ConcatProgramFactory,
        ConcatS2STiledProgramFactory,
        ConcatS2SRMProgramFactory,
        ConcatS2SMultiProgramFactory,
        ConcatS2IProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors);

    // Opts the (shared) Concat op into the descriptor fast-path. Only ConcatS2IProgramFactory bakes a raw
    // output address into a per-core runtime arg (writer arg 0) that no binding patches (its input addresses
    // ARE CB `.buffer`-bound), so that address is re-applied here. Every other factory either fast-paths via
    // Buffer* rt-arg bindings (ConcatProgramFactory, interleaved) or is fully CB `.buffer`-bound with only
    // shape/offset-derived rt-args (s2s tiled / rm / multi), so this returns {} for them.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::prim

namespace ttnn::prim {
ttnn::prim::ConcatDeviceOperation::tensor_return_value_t concat(
    const std::vector<Tensor>& input_tensors,
    std::int64_t dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids = std::nullopt);
}  // namespace ttnn::prim

namespace ttnn::operations::data_movement {

// Ref: https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
// Notes: Non-empty tensors provided must have the same shape, except in the cat dimension.
Tensor concat_impl(
    const std::vector<Tensor>& input_tensors,
    std::int64_t dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::operations::data_movement
