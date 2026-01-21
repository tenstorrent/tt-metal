// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::pool {
// Generic pool uop -- called from the macro-ops
struct Pool2D {
    struct operation_attributes_t {
        sliding_window::SlidingWindowConfig sliding_window_config_;
        Pool2DType pool_type_;
        DataType output_dtype_;
        Layout output_layout_;
        MemoryConfig memory_config_;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config_;
        bool count_include_pad_;
        std::optional<int32_t> divisor_override_;
        bool return_indices_;
        uint32_t memory_used;
        bool config_tensor_in_dram;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = std::vector<Tensor>;

    struct MultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader0_kernel{};
            tt::tt_metal::KernelHandle reader1_kernel{};
            tt::tt_metal::KernelHandle compute_kernel{};
            tt::tt_metal::CBHandle raw_in_cb{};
            tt::tt_metal::CBHandle out_cb{};
            tt::tt_metal::CBHandle out_idx_cb{};
            tt::tt_metal::CBHandle in_scalar_cb_0{};
            tt::tt_metal::CBHandle in_scalar_cb_1{};
            tt::tt_metal::CBHandle clear_value_cb{};
            tt::tt_metal::CBHandle in_reader_indices_cb{};
            tt::tt_metal::CBHandle in_cb_0{};
            tt::tt_metal::CBHandle in_cb_1{};
            tt::tt_metal::CBHandle pre_tilize_cb{};
            tt::tt_metal::CBHandle config_cb{};
            tt::tt_metal::CBHandle in_idx_cb{};
            tt::tt_metal::CBHandle pack_tmp_cb{};
            tt::tt_metal::CBHandle pack_idx_tmp_cb{};
            tt::tt_metal::CBHandle right_inc_cb{};
            tt::tt_metal::CBHandle down_left_wrap_inc_cb{};
            tt::tt_metal::CBHandle up_left_wrap_inc_cb{};
            uint32_t ncores{};
            tt::tt_metal::DeviceStorage reader_indices_storage;
            tt::tt_metal::DeviceStorage scalar_config_storage;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& op_attr,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<MultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
};

}  // namespace ttnn::operations::pool

namespace ttnn::prim {
std::vector<ttnn::Tensor> pool2d(
    const Tensor& input_tensor,
    const ttnn::operations::sliding_window::SlidingWindowConfig& sliding_window_config,
    ttnn::operations::pool::Pool2DType pool_type,
    DataType output_dtype,
    Layout output_layout,
    MemoryConfig memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    bool return_indices,
    uint32_t memory_used,
    bool config_tensor_in_dram);
}  // namespace ttnn::prim
