// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "factories/untilize_single_core_program_factory.hpp"
#include "factories/untilize_multi_core_sub_core_grids_program_factory.hpp"
#include "factories/untilize_multi_core_block_program_factory.hpp"
#include "factories/untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.hpp"
#include "factories/untilize_multi_core_parallelize_column_program_factory.hpp"
#include "factories/untilize_multi_core_program_factory.hpp"
#include "untilize_device_operation_types.hpp"

namespace ttnn::operations::data_movement {

uint32_t get_pf_type(bool output_is_sharded, const Tensor& tensor);

namespace untilize_helpers {

uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks);

}  // namespace untilize_helpers

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {

struct UntilizeDeviceOperation {
    using operation_attributes_t = UntilizeOperationAttributes;
    using tensor_args_t = UntilizeTensorArgs;
    using shape_return_value_t = UntilizeShapeReturnValue;
    using tensor_return_value_t = UntilizeTensorReturnValue;
    using spec_return_value_t = UntilizeSpecReturnValue;
    using program_factory_t = std::variant<
        UntilizeSingleCoreProgramFactory,
        UntilizeMultiCoreSubCoreGridsProgramFactory,
        UntilizeMultiCoreBlockProgramFactory,
        UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory,
        UntilizeMultiCoreParallelizeColumnProgramFactory,
        UntilizeMultiCoreProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        tt::tt_metal::MemoryConfig output_mem_config,
        bool use_multicore,
        bool use_pack_untilize,
        bool fp32_dest_acc_en,
        std::optional<CoreRangeSet> sub_core_grids,
        bool enough_space_width,
        bool enough_space_height,
        uint32_t pf_type);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& op_attr, const tensor_args_t& inputs, tensor_return_value_t& output);
};

Tensor untilize(
    const Tensor& input,
    tt::tt_metal::MemoryConfig output_mem_config,
    bool use_multicore,
    bool use_pack_untilize,
    bool fp32_dest_acc_en,
    std::optional<CoreRangeSet> sub_core_grids,
    bool enough_space_width,
    bool enough_space_height,
    uint32_t pf_type);

}  // namespace ttnn::prim
