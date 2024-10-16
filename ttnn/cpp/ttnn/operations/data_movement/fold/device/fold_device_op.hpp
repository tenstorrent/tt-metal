// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::data_movement {

struct Fold {
    struct operation_attributes_t {
        uint32_t stride_h;
        uint32_t stride_w;
        bool is_sharded;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
    };

    using shape_return_value_t = ttnn::SimpleShape;
    using tensor_return_value_t = Tensor;

    struct SingleCore {
        struct shared_variables_t {
            KernelHandle reader_kernel_id;
            KernelHandle writer_kernel_id;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(const operation_attributes_t& operation_attributes,
                                       const tensor_args_t& tensor_args,
                                       tensor_return_value_t& output_tensor);
        static void override_runtime_arguments(cached_program_t& cached_program,
                                               const operation_attributes_t& operation_attributes,
                                               const tensor_args_t& tensor_args,
                                               tensor_return_value_t& output_tensor);
    };

    struct MultiCore {
        struct shared_variables_t {
            KernelHandle writer_kernel_id;
            uint32_t stride_h;
            uint32_t stride_w;
            uint32_t cb_src0;
            uint32_t cb_dst0;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(const operation_attributes_t& operation_attributes,
                                       const tensor_args_t& tensor_args,
                                       tensor_return_value_t& output_tensor);
        static void override_runtime_arguments(cached_program_t& cached_program,
                                               const operation_attributes_t& operation_attributes,
                                               const tensor_args_t& tensor_args,
                                               tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<SingleCore, MultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        uint32_t stride_h,
        uint32_t stride_w,
        const std::optional<const tt::tt_metal::LegacyShape>& output_shape,
        uint32_t pad_c,
        uint32_t pad_h,
        uint32_t pad_w);
};

} // namespace ttnn::operations::data_movement

namespace ttnn::prim {
constexpr auto fold = ttnn::register_operation<"ttnn::prim::fold", ttnn::operations::data_movement::Fold>();
} // namespace ttnn::prim
