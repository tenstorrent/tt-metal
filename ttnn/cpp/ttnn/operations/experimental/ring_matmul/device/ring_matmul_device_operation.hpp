// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::operations::experimental::ring_matmul {

struct RingMatmulConfig {
    uint32_t in0_block_w = 1;
    uint32_t out_subblock_h = 1;
    uint32_t out_subblock_w = 1;
    uint32_t per_core_M = 1;
    uint32_t per_core_N = 1;
    bool packer_l1_acc = true;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;

    static constexpr auto attribute_names = std::make_tuple(
        "in0_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "per_core_M",
        "per_core_N",
        "packer_l1_acc",
        "fp32_dest_acc_en",
        "dst_full_sync_en");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->in0_block_w,
            this->out_subblock_h,
            this->out_subblock_w,
            this->per_core_M,
            this->per_core_N,
            this->packer_l1_acc,
            this->fp32_dest_acc_en,
            this->dst_full_sync_en);
    }
};

struct RingMatmulDeviceOperation {
    struct operation_attributes_t {
        std::optional<RingMatmulConfig> config;
        std::optional<unary::UnaryWithParam> fused_activation;
        std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
        std::optional<tt::tt_metal::DataType> output_dtype;
        DeviceComputeKernelConfig compute_kernel_config;
        CoreRangeSet hop_cores;
        std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
        uint32_t num_global_cb_receivers;
        std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
        std::optional<CoreRangeSet> restricted_cores;
        bool untilize_out;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
        const Tensor& weight_tensor;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            std::vector<tt::tt_metal::KernelHandle> kernels;
            std::vector<tt::tt_metal::CBHandle> cbs;
            std::vector<CoreCoord> cores;
            uint32_t num_cores;
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

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        std::optional<unary::UnaryWithParam> fused_activation,
        const std::optional<RingMatmulConfig>& config,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<DataType> dtype,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config,
        const CoreRangeSet& hop_cores,
        const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
        uint32_t num_global_cb_receivers,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        std::optional<CoreRangeSet> restricted_cores,
        bool untilize_out);
};

}  // namespace ttnn::operations::experimental::ring_matmul

namespace ttnn::prim {
constexpr auto ring_matmul = ttnn::register_operation<
    "ttnn::prim::ring_matmul",
    ttnn::operations::experimental::ring_matmul::RingMatmulDeviceOperation>();
}  // namespace ttnn::prim
