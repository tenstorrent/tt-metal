// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::csa_pool {

inline tt::tt_metal::Tile rm_face_tile() { return tt::tt_metal::Tile({1, tt::constants::TILE_WIDTH}, false); }

struct CsaPoolDeviceOperation {
    struct operation_attributes_t {
        uint32_t users = 0;
        uint32_t compress_rate = 0;
        uint32_t head_dim = 0;
        MemoryConfig output_mem_config;
        ttnn::DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& prev_kv;
        const Tensor& prev_gate;
        const Tensor& win_kv;
        const Tensor& win_gate;
        const Tensor& position_bias;
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

MemoryConfig default_output_memory_config(const Tensor& prev_kv, uint32_t users, uint32_t head_dim);

}  // namespace ttnn::operations::experimental::deepseek::csa_pool

namespace ttnn::prim {

ttnn::Tensor csa_pool_window(
    const ttnn::Tensor& prev_kv,
    const ttnn::Tensor& prev_gate,
    const ttnn::Tensor& win_kv,
    const ttnn::Tensor& win_gate,
    const ttnn::Tensor& position_bias,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config);

}  // namespace ttnn::prim
