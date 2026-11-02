// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::fused_msda {

// What fused_msda_head_major_value builds from a packed value of logical shape
// (B, S, H*D): (B, H, ceil(S/32), 32*D) ROW_MAJOR bfloat16. The producer's output
// spec and fused_msda's L1 fit check both use it.
tt::tt_metal::TensorSpec head_major_value_spec(
    const ttnn::Shape& packed_shape, uint32_t num_heads, const MemoryConfig& memory_config);

// Tiled packed value (B, S, H*D) -> ROW_MAJOR head-major (B, H, ceil(S/32), 32*D),
// the layout fused_msda reads a horizontal corner pair from in one NoC read.
// One output page is one (b, h, 32-key block): the D/32 input tiles that hold
// it, untilized. Rows past S in a head's last page carry the input's tile
// padding; fused_msda never reads them.
struct FusedMSDAHeadMajorValueOperation {
    struct operation_attributes_t {
        uint32_t num_heads = 0;
        MemoryConfig output_memory_config;
    };

    struct tensor_args_t {
        const Tensor& value;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::fused_msda

namespace ttnn::prim {

ttnn::Tensor fused_msda_head_major_value(
    const Tensor& value, uint32_t num_heads, const std::optional<MemoryConfig>& memory_config);

}  // namespace ttnn::prim
