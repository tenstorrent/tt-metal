// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::operations::experimental::matmul_decode {

// Wire contract with reader_*_width_sharded.cpp kernels.
enum class HubRole : uint32_t {
    Plain = 0,
    Hub0 = 1,
    Hub1 = 2,
};

struct MatmulDecodeDeviceOperation {
    struct operation_attributes_t {
        int M;
        int N;
        int K;
        std::optional<MemoryConfig> output_mem_config;
        std::optional<DataType> output_dtype;
        bool partial_width_sharded = false;
        int batch = 1;
        int b_blocks = 1;
        int n_blocks = 1;
        // DRAM-sender GlobalCircularBuffer feeding in1 from the tensor prefetcher.
        // When set, the weight is a DRAM ND-sharded tensor (one slab per receiver)
        // and the receiver grid comes from the GCB, not from a legacy shard spec.
        std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb = std::nullopt;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        const Tensor& input_tensor_b;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct FullWidthSharded {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct PartialWidthSharded {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct BatchedWidthSharded {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<FullWidthSharded, PartialWidthSharded, BatchedWidthSharded>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // The default reflection hash cannot tell two identically shaped GlobalCircularBuffers apart
    // (GlobalCircularBuffer::attribute_names carries no address), so the GCB's addresses are folded
    // in on top of everything the default hash already covered. See the .cpp for why that matters.
    // Note that declaring this opts the op out of attribute-level canonical-key collision
    // resolution, so cache separation rests on the hash alone.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::matmul_decode

namespace ttnn::prim {
ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation::tensor_return_value_t matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded = false,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt);
}  // namespace ttnn::prim
