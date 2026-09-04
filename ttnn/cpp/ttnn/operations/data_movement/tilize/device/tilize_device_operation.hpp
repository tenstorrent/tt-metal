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
#include "tilize_multi_core_sharded_retile_program_factory.hpp"
#include "tilize_multi_core_retile_program_factory.hpp"
#include "tilize_device_operation_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct TilizeDeviceOperation {
    using operation_attributes_t = ttnn::prim::TilizeParams;
    using tensor_args_t = ttnn::prim::TilizeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        TilizeMultiCoreDefaultProgramFactory,
        TilizeMultiCoreBlockProgramFactory,
        TilizeSingleCoreProgramFactory,
        TilizeMultiCoreShardedProgramFactory,
        TilizeMultiCoreShardedRetileProgramFactory,
        TilizeMultiCoreRetileProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

// Re-point slot 0 of every core's args for one kernel. Shared by the tilize factories' cache-hit
// hooks so the slot layout the factories all bake has a single home.
void patch_tilize_kernel_slot0(tt::tt_metal::Program& program, uint32_t kernel_idx, uint32_t address);

ttnn::Tensor tilize(
    const Tensor& input_tensors,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    bool use_multicore,
    bool enough_space_height,
    bool use_low_perf,
    const tt::tt_metal::Tile& tile,
    const std::optional<CoreRangeSet>& sub_core_grids);
}  // namespace ttnn::prim
