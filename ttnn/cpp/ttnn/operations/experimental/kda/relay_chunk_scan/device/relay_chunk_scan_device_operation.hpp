// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>

#include "ttnn/operation.hpp"
#include "relay_chunk_scan_program_factory.hpp"

namespace ttnn::experimental::prim {

struct RelayChunkScanOperation {
    using operation_attributes_t = RelayChunkScanParams;
    using tensor_args_t = RelayChunkScanInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<RelayChunkScanProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::tuple<Tensor, Tensor> relay_chunk_scan(
    const Tensor& packed,
    const Tensor& terms,
    const Tensor& initial_state,
    const Tensor& final_e,
    const Tensor& final_b,
    const Tensor& output_gate,
    const Tensor& norm_weight,
    const Tensor& actual_start,
    const std::optional<Tensor>& actual_end,
    const Tensor& landing,
    uint32_t num_heads,
    uint32_t gate_column_offset,
    float norm_epsilon,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig&,
    const std::array<uint32_t, 3>& semaphore_addresses,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t num_links,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const std::optional<Tensor>& slab_index = std::nullopt,
    uint32_t slab_index_scale = 1,
    uint32_t slab_index_offset = 0);

}  // namespace ttnn::experimental::prim
