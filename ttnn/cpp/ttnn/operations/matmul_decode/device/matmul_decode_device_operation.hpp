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

namespace ttnn::operations::matmul_decode {

// -----------------------------------------------------------------------------
// MatmulDecodeDeviceOperation
//
// TEMPLATE / SKELETON ONLY -- this is intentionally NOT a functional matmul.
// It mirrors the structure of the example device operation
// (ttnn/cpp/ttnn/operations/examples/example) so it can be fleshed out into a
// real decode-optimized matmul. Fill in the program factories with the actual
// reader / compute / writer kernels and runtime args to make it functional.
// -----------------------------------------------------------------------------
struct MatmulDecodeDeviceOperation {
    // Non-tensor configuration for the operation.
    struct operation_attributes_t {
        int M;
        int N;
        int K;
        MemoryConfig output_mem_config;
        std::optional<DataType> output_dtype;
    };

    // Tensors passed in/out of the operation.
    struct tensor_args_t {
        const Tensor& input_tensor_a;
        const Tensor& input_tensor_b;
    };

    // Output spec / tensor types. A single matmul output here.
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    // -------------------------------------------------------------------------
    // Descriptor-based program factories.
    //
    // Each factory returns a ProgramDescriptor. The framework handles program
    // construction, caching, and runtime argument patching automatically.
    // -------------------------------------------------------------------------

    // Full width-sharded: keeps the full output width resident across the core
    // grid, with each core owning a contiguous slice of the N (width) dimension.
    struct FullWidthSharded {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Multi-core: distributes output tiles across the available core grid.
    struct MultiCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<FullWidthSharded, MultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::matmul_decode

namespace ttnn::prim {
ttnn::operations::matmul_decode::MatmulDecodeDeviceOperation::tensor_return_value_t matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt);
}  // namespace ttnn::prim
