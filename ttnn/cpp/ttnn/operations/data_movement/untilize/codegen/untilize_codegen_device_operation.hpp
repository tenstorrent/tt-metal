// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/types.hpp"
#include "untilize_codegen_program_factory.hpp"

namespace ttnn::prim {

struct UntilizeCodegenTensorArgs {
    Tensor input;
    // Device-held extent (trace-safe): a 1-element uint32 DRAM tensor holding a chunk start. When set,
    // only the gathered prefix is untilized: g = min(round_up(start + gathered_slab_global,
    // gathered_slab_global), H) rows of the input height H, split into num_segments equal segments
    // (rank-major, as high_bw_all_gather's output), the first g / num_segments rows of each.
    // Output rows past that in each segment are left as they are. The same extent rule as
    // high_bw_all_gather's gathered_prefix_tensor / gathered_slab_global.
    std::optional<Tensor> gathered_prefix_tensor = std::nullopt;
};

struct UntilizeCodegenOperationAttributes {
    tt::tt_metal::MemoryConfig output_mem_config;
    uint32_t gathered_slab_global = 0;  // with gathered_prefix_tensor: the slab the extent rounds up to
    uint32_t num_segments = 1;  // with gathered_prefix_tensor: equal row segments, each untilized to g / num_segments
};

using UntilizeCodegenTensorReturnValue = Tensor;
using UntilizeCodegenSpecReturnValue = tt::tt_metal::TensorSpec;

struct UntilizeCodegenDeviceOperation {
    using operation_attributes_t = UntilizeCodegenOperationAttributes;
    using tensor_args_t = UntilizeCodegenTensorArgs;
    using spec_return_value_t = UntilizeCodegenSpecReturnValue;
    using tensor_return_value_t = UntilizeCodegenTensorReturnValue;
    using program_factory_t = std::variant<UntilizeCodegenProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor untilize_codegen(
    const Tensor& input,
    tt::tt_metal::MemoryConfig output_mem_config,
    const std::optional<Tensor>& gathered_prefix_tensor = std::nullopt,
    uint32_t gathered_slab_global = 0,
    uint32_t num_segments = 1);

}  // namespace ttnn::prim
