// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gather_device_operation_types.hpp"

#include <optional>
#include <variant>

#include "ttnn/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct GatherDeviceOperation {
    using operation_attributes_t = GatherParams;
    using tensor_args_t = GatherInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SingleRowSingleCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };

    struct SingleRowMultiCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };

    // ROW_MAJOR variants: row/column split at stick granularity; noc_async_*_sharded with per-shard page size.
    struct RmSingleRowSingleCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };

    struct RmSingleRowMultiCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };

    using program_factory_t =
        std::variant<SingleRowSingleCore, SingleRowMultiCore, RmSingleRowSingleCore, RmSingleRowMultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};

// True for a BLOCK/WIDTH-sharded placement whose shard row (shard width * element_size) is not a
// multiple of the buffer alignment. The device stores each shard row at a pitch of the aligned page
// size, but the RM factories hand `noc_async_*_sharded` the unaligned shard row as both the page
// pitch and the per-page payload, so the two agree only for aligned shard rows. ttnn::gather moves
// such tensors to interleaved SRAM before the RM factories run.
bool has_subaligned_shard_row(const MemoryConfig& memory_config, uint32_t element_size);

Tensor gather(
    const Tensor& input_tensor,
    int8_t dim,
    const Tensor& input_index_tensor,
    bool sparse_grad,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& output_tensors,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::prim
