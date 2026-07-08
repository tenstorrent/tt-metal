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

namespace ttnn::operations::experimental::matmul_decode {

// Role of each reader core in the two-hub gather/broadcast scheme (both program
// factories). Passed to the reader kernel as a runtime arg; the explicit values
// are the wire contract shared with the reader_*_width_sharded.cpp kernels.
enum class HubRole : uint32_t {
    Plain = 0,  // plain core (pure receiver and/or sender)
    Hub0 = 1,   // hub 0: start corner, broadcasts on NOC0
    Hub1 = 2,   // hub 1: end corner, broadcasts on NOC1
};

// -----------------------------------------------------------------------------
// MatmulDecodeDeviceOperation
//
// Decode-optimized matmul C = A @ B for L1 width-sharded operands. A ([M, K]) is
// width(K)-sharded and gathered onto every core; B is width-sharded. Dispatches to
// one of two program factories (see FullWidthSharded / PartialWidthSharded below)
// based on the partial_width_sharded attribute.
// -----------------------------------------------------------------------------
struct MatmulDecodeDeviceOperation {
    // Non-tensor configuration for the operation.
    struct operation_attributes_t {
        int M;
        int N;
        int K;
        std::optional<MemoryConfig> output_mem_config;
        std::optional<DataType> output_dtype;
        // Selects the program factory: true -> PartialWidthSharded (B sharded along both
        // K and N with a cross-core K-reduction); false -> FullWidthSharded (B width(N)-sharded).
        // Ignored when A is rank-3 (batched), which always dispatches to BatchedWidthSharded.
        bool partial_width_sharded = false;
        // Batched (rank-3) geometry. A is [batch, M, K] and the weights are folded along
        // BOTH the batch (B) and N dimensions across b_blocks * n_blocks cores. Unused for the
        // rank-2 factories (kept at 1). See BatchedWidthSharded for the fold layout.
        int batch = 1;
        int b_blocks = 1;
        int n_blocks = 1;
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

    // Partial width-sharded: B is sharded along BOTH K and N. The N dimension is
    // split across N_blocks cores and the K dimension across K_blocks cores, so a
    // single core holds only a [K/K_blocks, N/N_blocks] block of B (expressed as a
    // width-sharded tensor after the caller reshapes/permutes B). Each core computes
    // a partial product over its K-slice; the K_blocks partials for each N-slice are
    // then reduced (summed) onto the base core that owns that N-slice of the output.
    struct PartialWidthSharded {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Batched width-sharded: a batched matmul C[b] = A[b] @ B[b] for a rank-3 activation
    // A ([batch, M, K]). Unlike PartialWidthSharded (which folds the weights along K and N and
    // reduces the K-partials across cores), this folds the weights along BOTH the batch (B) and
    // N dimensions: the batch is split across b_blocks and N across n_blocks, so a single core
    // owns a [Bc, K, Nc] block of the weights (Bc = batch / b_blocks, Nc = N / n_blocks),
    // expressed as a width-sharded tensor [Bc * K, b_blocks * N] with shard [Bc * K, Nc] laid out
    // b-major (core c owns b_idx = c / n_blocks, n_idx = c % n_blocks). The batched matmul is
    // block-diagonal, so there is NO cross-core reduction: each core independently computes its
    // own [Bc, M, Nc] output block. The output mirrors the weight fold -- a width-sharded tensor
    // [Bc * M, b_blocks * N] with shard [Bc * M, Nc] -- which the caller unfolds back to
    // [batch, M, N].
    struct BatchedWidthSharded {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<FullWidthSharded, PartialWidthSharded, BatchedWidthSharded>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

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
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
}  // namespace ttnn::prim
