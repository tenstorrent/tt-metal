// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::matmul_decode {

// -----------------------------------------------------------------------------
// GateUpMatmulDecodeDeviceOperation
//
// Fused GeGLU gate+up projection: ONE gather of the activation A, TWO weights
// (gate_b, up_b), TWO outputs (gate, up). Both weights are partial-width-sharded
// resident-L1 bf8_b tensors laid out on the SAME core grid (K_blocks * N_blocks
// cores, k-major) as a single matmul_decode(partial_width_sharded=True) call.
//
//   gate = gelu(A @ gate_w)   (tanh-approx gelu, fused into phase-2 pack)
//   up   =      A @ up_w
//
// The reader gathers the full A onto every core exactly ONCE; the compute kernel
// runs the partial matmul twice (one per resident weight) over that single gathered
// A; the writer cross-core-reduces BOTH partials to two reduce CBs on the base core;
// phase-2 reduces both and packs two output shards (gate gets the fused gelu). This
// halves the per-MLP x-gather + reduce/dispatch relative to two separate
// matmul_decode calls.
//
// Mirrors MatmulDecodeDeviceOperation::PartialWidthSharded (same A-reshard reader,
// same geometry, same K_blocks=2 pairwise reduce) but threads a second weight + a
// second output through every stage.
// -----------------------------------------------------------------------------
struct GateUpMatmulDecodeDeviceOperation {
    struct operation_attributes_t {
        int M;
        int N;  // output width of EACH of gate/up (they share dims)
        int K;
        MemoryConfig output_mem_config;
        std::optional<DataType> output_dtype;
        std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
        // tanh-approx (true) vs exact-erf (false) gelu for the gate activation.
        bool fused_gelu_approx = false;
        // A is passed interleaved/sharded and the reader reshards it across reshard_cores
        // sender cores (folds the caller's to_memory_config reshard into the op).
        bool reshard_input = false;
        uint32_t reshard_cores = 2;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;  // activation A (shared, gathered once)
        const Tensor& gate_b;          // partial-width-sharded gate weight
        const Tensor& up_b;            // partial-width-sharded up weight
    };

    // Two outputs: {gate, up}, both [..., M, N] width-sharded across N_blocks cores.
    using spec_return_value_t = std::array<ttnn::TensorSpec, 2>;
    using tensor_return_value_t = std::array<Tensor, 2>;

    // Single program factory: the gate+up dual-weight partial-width-sharded matmul.
    struct GateUpPartialWidthSharded {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<GateUpPartialWidthSharded>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::matmul_decode

namespace ttnn::prim {
ttnn::operations::matmul_decode::GateUpMatmulDecodeDeviceOperation::tensor_return_value_t gate_up_matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& gate_b,
    const Tensor& up_b,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool fused_gelu_approx = false,
    bool reshard_input = false,
    uint32_t reshard_cores = 2);
}  // namespace ttnn::prim
