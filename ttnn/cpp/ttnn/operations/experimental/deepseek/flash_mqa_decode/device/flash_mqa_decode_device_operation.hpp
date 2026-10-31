// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::flash_mqa_decode {

// The SDPA custom matmul works on (8, 32) Q tiles: one tile row of 8 heads per core.
inline constexpr uint32_t kHeadsPerCore = 8;

struct FlashMqaDecodeDeviceOperation {
    struct operation_attributes_t {
        float scale = 1.0f;
        uint32_t k_chunk_size = 64;
        uint32_t num_s_blocks = 8;
        // Row of selected_kv that selected row 0 sits in.
        uint32_t selected_row_offset = 0;
        // Most ring rows attended; 0 for all of kv's rows.
        uint32_t ring_rows = 0;
        MemoryConfig output_mem_config;
        ttnn::DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& q;
        const Tensor& kv;
        const Tensor& lengths;
        std::optional<Tensor> selected_kv;
        std::optional<Tensor> attention_sink;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::flash_mqa_decode

namespace ttnn::prim {

ttnn::Tensor flash_mqa_decode(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& lengths,
    const std::optional<ttnn::Tensor>& selected_kv,
    uint32_t selected_row_offset,
    uint32_t ring_rows,
    const std::optional<ttnn::Tensor>& attention_sink,
    std::optional<float> scale,
    uint32_t k_chunk_size,
    uint32_t num_s_blocks,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config);

}  // namespace ttnn::prim
