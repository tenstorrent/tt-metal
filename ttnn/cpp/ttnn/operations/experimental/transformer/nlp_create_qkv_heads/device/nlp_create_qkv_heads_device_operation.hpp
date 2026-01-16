// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operation.hpp"
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::transformer {

struct NlpCreateHeadsDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_q_heads;
        uint32_t num_kv_heads;
        uint32_t head_dim;
        bool transpose_k_heads;
        MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_q;
        const std::optional<Tensor>& input_tensor_kv;
        std::vector<std::optional<Tensor>> optional_output_tensors;
    };

    using spec_return_value_t = std::tuple<ttnn::TensorSpec, ttnn::TensorSpec, ttnn::TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor>;

    struct Interleaved {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
            bool read_from_input_tensor_kv;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct Sharded {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id{};
            tt::tt_metal::KernelHandle writer_kernel_id{};
            std::size_t num_cores{};
            std::size_t num_cores_y{};
            bool read_from_input_tensor_kv{};
            tt::tt_metal::CBHandle cb_q_output{};
            tt::tt_metal::CBHandle cb_k_output{};
            tt::tt_metal::CBHandle cb_v_output{};
            std::vector<CoreCoord> cores;
            uint32_t head_size{};
            uint32_t per_risc0_out_q_heads{};
            uint32_t per_risc1_out_q_heads{};
            uint32_t per_core_in_q_heads{};
            uint32_t per_core_out_kv_heads{};
            uint32_t per_core_in_kv_heads{};
            uint32_t head_tiles{};
            uint32_t num_kv_cores{};
            uint32_t single_tile_size{};
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<Interleaved, Sharded>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::prim {
std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads(
    const Tensor& input_tensor_q,
    const std::optional<Tensor>& input_tensor_kv,
    uint32_t num_q_heads,
    std::optional<uint32_t> num_kv_heads,
    uint32_t head_dim,
    bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors);
}  // namespace ttnn::prim
