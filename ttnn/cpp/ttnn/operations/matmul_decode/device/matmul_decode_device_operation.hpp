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
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
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
        // When true, force the partial width-sharded program factory (B sharded along
        // both K and N with a cross-core K-reduction). When false, the factory is chosen
        // automatically from the input layouts.
        bool partial_width_sharded = false;
        // deep-plan_13: explicit fat-fill override (sweep tunable). When unset the
        // program factory auto-derives (out_subblock_h, out_subblock_w) from the per-core
        // M/N tile counts via the ported native get_subblock_sizes, bounded by the DST cap.
        // out_subblock_h>1 (M-fill) is P0-A gated (factory MMD_ENABLE_M_FILL); v1 default
        // is out_w-only (out_subblock_h=1).
        std::optional<uint32_t> out_subblock_h = std::nullopt;
        std::optional<uint32_t> out_subblock_w = std::nullopt;
        // deep-plan_14 Lever 0: end-to-end knob plumbing. in0_block_w is a genuinely-new
        // attribute (the compute kernel hardcoded constexpr in0_block_w=1). Larger in0_block_w
        // reduces matmul_block invocations / improves K-reuse on large-K shapes. Default 1
        // (byte-identical to today). k_stream + k_slice_tiles carry the (Phase-B, gated)
        // WIDTH-temporal streaming knobs; default off (one-shot path unchanged). They are
        // threaded NOW so the nanobind docstring can advertise "stream_k" (the blocked wrapper
        // flips _MATMUL_DECODE_HAS_STREAM_K on that substring) even before the kernel bodies land.
        uint32_t in0_block_w = 1;
        bool k_stream = false;
        uint32_t k_slice_tiles = 0;  // 0 == auto/unused (full-K one-shot)
        // Fully-resolved compute kernel config (math fidelity / fp32 dest acc / etc.)
        // threaded down to the program factories. Resolved at op-invocation time in
        // ttnn::prim::matmul_decode via init_device_compute_kernel_config so that the
        // factories never hardcode fp32_dest_acc_en. DEFAULT (no user config passed)
        // resolves to fp32_dest_acc_en=false, math_fidelity=HiFi4 (the op's fidelity
        // floor). Pass a DeviceComputeKernelConfig with fp32_dest_acc_en=true to opt in.
        DeviceComputeKernelConfig compute_kernel_config;
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

    // Multi-core: distributes output tiles across the available core grid.
    struct MultiCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<FullWidthSharded, PartialWidthSharded, MultiCore>;

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
    bool partial_width_sharded = false,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    // deep-plan_14 Lever 0: settable fat-fill + temporal knobs (defaults preserve today).
    std::optional<uint32_t> out_subblock_h = std::nullopt,
    std::optional<uint32_t> out_subblock_w = std::nullopt,
    uint32_t in0_block_w = 1,
    bool k_stream = false,
    uint32_t k_slice_tiles = 0);
}  // namespace ttnn::prim
