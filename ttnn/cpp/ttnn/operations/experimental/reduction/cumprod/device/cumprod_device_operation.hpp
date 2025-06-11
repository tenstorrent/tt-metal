// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <type_traits>
#include <variant>

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::reduction {

using namespace tt::tt_metal;
using namespace tt::stl;

struct CumprodDeviceOperation {
    struct operation_attributes_t {
        const int32_t dim;
        const DataType dtype;
        const MemoryConfig output_memory_config;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
        std::optional<Tensor> optional_out{std::nullopt};
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct MultiCoreCumprodProgramFactory {
        enum class CumprodCB : std::underlying_type_t<tt::CBIndex> {
            SRC = tt::CBIndex::c_0,
            DST = tt::CBIndex::c_1,
            ONE = tt::CBIndex::c_2,
            ACC = tt::CBIndex::c_3
        };

        static constexpr std::array<const char*, 3> KERNEL_PATHS{
            "ttnn/cpp/ttnn/operations/experimental/reduction/cumprod/device/kernels/dataflow/"
            "reader_multicore_cumprod.cpp",
            "ttnn/cpp/ttnn/operations/experimental/reduction/cumprod/device/kernels/compute/cumprod_multicore.cpp",
            "ttnn/cpp/ttnn/operations/experimental/reduction/cumprod/device/kernels/dataflow/"
            "writer_multicore_cumprod.cpp"};
        struct shared_variables_t {
            KernelHandle cumprod_reader_kernel_id;
            KernelHandle cumprod_compute_kernel_id;
            KernelHandle cumprod_writer_kernel_id;
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

        static CBHandle create_cb(
            Program& program,
            const DataType& dtype,
            const CumprodCB& cumprod_cb,
            const CoreRangeSet& core_range_set,
            const uint32_t& tiles_num);

        static KernelHandle create_kernel(
            Program& program,
            const char* kernel_path,
            const CoreRangeSet& core_range_set,
            const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
            const std::vector<uint32_t>& runtime_args = {});

        static uint32_t calc_input_tile_offset(const Shape& input_shape, const int32_t& dim);
    };

    using program_factory_t = std::variant<MultiCoreCumprodProgramFactory>;
    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static invocation_result_t invoke(
        const Tensor& input_tensor,
        const int32_t& dim,
        std::optional<DataType>& dtype,
        std::optional<Tensor> optional_out,
        const MemoryConfig& memory_config,
        const QueueId& queue_id = DefaultQueueId);
};

}  // namespace ttnn::operations::experimental::reduction

namespace ttnn::prim {
constexpr auto cumprod = ttnn::
    register_operation<"ttnn::prim::cumprod", ttnn::operations::experimental::reduction::CumprodDeviceOperation>();
}  // namespace ttnn::prim
