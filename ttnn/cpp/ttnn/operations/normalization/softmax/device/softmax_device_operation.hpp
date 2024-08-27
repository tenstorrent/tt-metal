// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include "common/base_types.hpp"
#include "common/core_coord.h"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"


#include "softmax_types.hpp"

namespace ttnn::operations::normalization {

struct SoftmaxDeviceOperation {
    struct operation_attributes_t {
        const std::optional<float> scale = std::nullopt;
        const bool inplace = false;
        const std::optional<MemoryConfig> memory_config = std::nullopt;
        const SoftmaxProgramConfig program_config = SoftmaxDefaultProgramConfig{};
        const bool is_causal_mask = false;
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
        const bool is_scale_causal_mask_hw_dims_softmax = false;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
        const std::optional<const Tensor> mask;
    };

    using shape_return_value_t = tt::tt_metal::Shape;
    using tensor_return_value_t = Tensor;

    struct SoftmaxMultiCoreProgramFactory {
        struct shared_variables_t {
            KernelHandle reader_kernels_id;
            KernelHandle writer_kernels_id;
            KernelHandle softmax_kernels_id;
            CoreCoord grid_size;
            uint32_t scalar_tile_size;
            uint32_t in0_tile_size;
            uint32_t im_tile_size;
            uint32_t out0_tile_size;
            uint32_t mask_tile_size;
            CBHandle cb_in0_id;
            CBHandle cb_out0_id;
            CBHandle cb_in2_id;
            CBHandle cb_intermed0_id;
            CBHandle cb_intermed1_id;
            std::optional<CBHandle> cb_intermed3_id;
            std::optional<CBHandle> cb_in3_id;
            std::optional<CBHandle> cb_in4_id;
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
    struct SoftmaxMultiCoreShardedProgramFactory {
        struct shared_variables_t {
            KernelHandle reader_kernels_id;
            CBHandle cb_in0_id;
            CBHandle cb_out0_id;
            std::optional<CBHandle> cb_in3_id;
            uint32_t num_cores;
            CoreCoord grid_size;
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

    using program_factory_t = std::variant<SoftmaxMultiCoreProgramFactory, SoftmaxMultiCoreShardedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::normalization

namespace ttnn::prim {
constexpr auto softmax = ttnn::register_operation<
    "ttnn::prim::softmax",
    ttnn::operations::normalization::SoftmaxDeviceOperation>();
}  // namespace ttnn::prim
