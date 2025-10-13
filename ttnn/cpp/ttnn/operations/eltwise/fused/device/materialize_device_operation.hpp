// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/lazy/expression.hpp"

namespace ttnn {
namespace operations::fused {

struct MaterializeDeviceOperation {
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {
        std::string compute_kernel_source;
        std::size_t circular_buffers;
        std::map<tt::CBIndex, std::size_t> inputs;
        tt::CBIndex output;
        ttsl::SmallVector<lazy::Param> params;
        tt::tt_metal::MemoryConfig memory_config;
        DataType dtype;
        CoreRangeSet worker_grid;
    };

    struct tensor_args_t {
        std::vector<Tensor> input_tensors;
    };

    struct ProgramFactory {
        // NOLINTNEXTLINE(*-member-init)
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            ttsl::SmallVector<tt::tt_metal::CBHandle> cbs;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& outout);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(lazy::FunctionView expression);
};

}  // namespace operations::fused

namespace prim {

constexpr auto materialize =
    ttnn::register_operation<"ttnn::prim::materialize", ttnn::operations::fused::MaterializeDeviceOperation>();

}  // namespace prim
}  // namespace ttnn
