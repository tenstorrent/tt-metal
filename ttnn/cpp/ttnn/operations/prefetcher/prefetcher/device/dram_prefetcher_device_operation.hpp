// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/global_circular_buffer.hpp>
#include "dram_prefetcher_device_operation_types.hpp"
#include "dram_prefetcher_program_factory.hpp"

namespace ttnn::operations::dram_prefetcher {

struct DramPrefetcherOperation {
    using operation_attributes_t = DramPrefetcherParams;
    using tensor_args_t = DramPrefetcherInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::DramPrefetcherProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::dram_prefetcher

namespace ttnn::prim {
ttnn::operations::dram_prefetcher::DramPrefetcherOperation::tensor_return_value_t dram_prefetcher(
    std::vector<ttnn::Tensor>& tensors,
    uint32_t num_layers,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    bool enable_performance_mode);
}  // namespace ttnn::prim
