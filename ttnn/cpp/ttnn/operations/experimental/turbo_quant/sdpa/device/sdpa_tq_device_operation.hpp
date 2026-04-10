// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::turbo_quant {

struct SDPATQDeviceOperation {
    struct operation_attributes_t {
        float scale;
        std::vector<float> centroids;  // centroid values (size = 2^bits)
        tt::tt_metal::MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor& q;           // [B, NQH, 1, DH] BF16
        const Tensor& k_indices;   // [B, NKH, Sk, DH] BFP4 paged
        const Tensor& k_norms;     // [B, NKH, Sk, 1] BF16
        const Tensor& v_indices;   // [B, NKH, Sk, vDH] BFP4 paged
        const Tensor& v_norms;     // [B, NKH, Sk, 1] BF16
        const Tensor& page_table;  // [B, max_pages] Int32
        const Tensor& cur_pos;     // [B] Int32
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct MultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
            std::size_t grid_size_x;
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

    using program_factory_t = std::variant<MultiCore>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::turbo_quant

namespace ttnn::prim {

Tensor turbo_quant_sdpa_decode(
    const Tensor& q,
    const Tensor& k_indices,
    const Tensor& k_norms,
    const Tensor& v_indices,
    const Tensor& v_norms,
    const Tensor& page_table,
    const Tensor& cur_pos,
    const std::vector<float>& centroids,
    float scale);

}  // namespace ttnn::prim
