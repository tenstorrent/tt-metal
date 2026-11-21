// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

// The output spec for an input of logical shape (B, S, H*D): (B, H, ceil(S/32),
// 32*D) ROW_MAJOR bfloat16. Exposed so a consumer can size the output before
// calling the op, e.g. for an L1 fit check.
tt::tt_metal::TensorSpec untilize_and_split_heads_output_spec(
    const ttnn::Shape& input_shape, uint32_t num_heads, const MemoryConfig& memory_config);

// One output page is one (b, h, tile row): the D/32 input tiles that hold it,
// untilized. A reader fetches those tiles, compute untilizes one page at a
// time, and a writer stores the 32 rows as one page; pages are split evenly
// over the compute grid.
struct UntilizeAndSplitHeadsDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_heads = 0;
        MemoryConfig output_memory_config;
    };

    struct tensor_args_t {
        const Tensor& input;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::Tensor untilize_and_split_heads(
    const Tensor& input, uint32_t num_heads, const std::optional<MemoryConfig>& memory_config);

}  // namespace ttnn::prim
