// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "tilize_multi_core_default_program_factory.hpp"
#include "tilize_multi_core_block_program_factory.hpp"
#include "tilize_single_core_program_factory.hpp"
#include "tilize_multi_core_sharded_program_factory.hpp"
#include "tilize_device_operation_types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/program_descriptor_patching.hpp>

namespace ttnn::prim {

struct TilizeDeviceOperation {
    using operation_attributes_t = ttnn::prim::TilizeParams;
    using tensor_args_t = ttnn::prim::TilizeInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        TilizeMultiCoreDefaultProgramFactory,
        TilizeMultiCoreBlockProgramFactory,
        TilizeSingleCoreProgramFactory,
        TilizeMultiCoreShardedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);

    // #48928: the sharded factory is pure CB-bound; opt into the descriptor fast-path on a cache hit.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&,
        const std::optional<ttnn::MeshCoordinate>& = std::nullopt);
};

ttnn::Tensor tilize(
    const Tensor& input_tensors,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    bool use_multicore,
    bool enough_space_width,
    bool enough_space_height,
    bool use_low_perf,
    const std::optional<CoreRangeSet>& sub_core_grids,
    tt::tt_metal::Tile tile);
}  // namespace ttnn::prim
