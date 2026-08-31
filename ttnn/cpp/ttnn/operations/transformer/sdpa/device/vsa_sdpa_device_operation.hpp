// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include <optional>
#include <variant>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct VsaSdpaOperation {
    using operation_attributes_t = VsaSdpaParams;
    using tensor_args_t = VsaSdpaInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct VsaSdpaProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& attrs, const tensor_args_t& t, tensor_return_value_t& output);

        // Cache-hit re-apply of buffer addresses and the work split. Unlike sparse_sdpa_msa, every shape is
        // hashed (no runtime-T patching), so only addresses and the split are dynamic.
        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& attrs,
            const tensor_args_t& t,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<VsaSdpaProgramFactory>;

    // Runtime-arg slots, named so create_descriptor's emplace order and override_runtime_arguments'
    // in-place writes reference the same symbols instead of agreeing on bare positions.
    enum ReaderArg : uint32_t {
        kReaderQAddr,
        kReaderKAddr,
        kReaderVAddr,
        kReaderIdxAddr,
        kReaderCountsAddr,
        kReaderWorkStart,
        kReaderWorkCount,
        kReaderArgCount,
    };
    enum WriterArg : uint32_t {
        kWriterOutAddr,
        kWriterKAddr,
        kWriterVAddr,
        kWriterWorkStart,
        kWriterWorkCount,
        kWriterArgCount,
    };
    enum ComputeArg : uint32_t { kComputeWorkStart, kComputeWorkCount, kComputeArgCount };

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    // Work split: total_work = H * (S / 64), one unit per (head, query tile), contiguous per core.
    struct DispatchArgs {
        tt::tt_metal::CoreCoord grid;  // per-core arg order: core i == {i % grid.x, i / grid.x}
        uint32_t num_cores = 0;
        uint32_t base_work = 0;
        uint32_t extra = 0;
    };
    static DispatchArgs compute_dispatch_args(const operation_attributes_t& attrs, const tensor_args_t& t);
};

Tensor vsa_sdpa(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& indices,
    const Tensor& block_counts,
    float scale,
    uint32_t block_size,
    uint32_t k_chunk_blocks,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);

}  // namespace ttnn::prim
