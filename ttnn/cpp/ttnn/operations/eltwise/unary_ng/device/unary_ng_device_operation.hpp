// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::unary_ng {

struct UnaryNgDeviceOperation {
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {
        std::vector<unary::EltwiseUnaryWithParam> op_chain;
        DataType output_dtype;
        tt::tt_metal::MemoryConfig memory_config;
        bool fp32_dest_acc_en = false;
        bool preserve_fp32_precision = false;
        bool bfp8_pack_precise = false;
        const CoreRangeSet worker_grid;
        std::optional<CoreRangeSet> sub_core_grids;

        tt::stl::hash::hash_t to_hash() const;
    };

    struct tensor_args_t {
        const Tensor& input;
        std::optional<Tensor> output_tensor;
    };

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id{};
            tt::tt_metal::KernelHandle writer_kernel_id{};
            tt::tt_metal::KernelHandle compute_kernel_id{};
            tt::tt_metal::CBHandle cb_src{};
            tt::tt_metal::CBHandle cb_out{};
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
};

}  // namespace ttnn::operations::unary_ng

namespace ttnn::prim {

Tensor unary_ng(
    const Tensor& input,
    const std::vector<ttnn::operations::unary::EltwiseUnaryWithParam>& op_chain,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    bool bfp8_pack_precise,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::prim
