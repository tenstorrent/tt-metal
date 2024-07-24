// SPDX-FileCopyrightText: © 2024 BOS
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace test_ops {

struct Concat {
    
    struct operation_attributes_t {
        unint32_t dim;
        const MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        std::vector<Tensor> &input_tensors;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    struct SingleCore {
        struct shared_variables_t {
            uint32_t var1;
        }
        static cached_program_t create(
            const operation_attributes_t &operation_attributes,
            const tensor_args_t &tensor_args,
            Tensor &tensor_return_value
        );
        static void override_runtime_arguments(
            cached_program_t cached_program,
            const operation_attributes_t &operation_attributes,
            const tensor_args_t &tensor_args,
            Tensor &tensor_return_value
        );
    };

    struct MultiCore {
        struct shared_variables_t {
            tt_metal::KernelHandle unary_reader_kernel_id;
            tt_metal::KernelHandle unary_writer_kernel_id;
            uint32_t num_input_tensors;
            uint32_t num_output_rows_per_core;
            uint32_t output_stick_size;
        }
        static cached_program_t create(
            const operation_attributes_t &operation_attributes,
            const tensor_args_t &input_tensors,
            Tensor &output
        );
        static void override_runtime_arguments(
            cached_program_t cached_program,
            const operation_attributes_t &operation_attributes,
            const tensor_args_t &input_tensors,
            Tensor &output
        );
    };

    using program_factory_t = std::variant<SingleCore, MultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttnn::Shape compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);

    static Tensor create_output_tensors(const operation_attributes_t& operation_attributes, const tensor_args_t&);

};

}
}
constexpr auto test_concat = ttnn::test_concat<operations::test_ops::Concat>("ttnn::test_concat");
}
