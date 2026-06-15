// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "accumulation_device_operation_types.hpp"

#include <optional>
#include <type_traits>
#include <variant>

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace ttsl;

struct AccumulationDeviceOperation {
    using operation_attributes_t = AccumulationParams;
    using tensor_args_t = AccumulationInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct AccumulationProgramFactory {
        enum class AccumulationCB : std::underlying_type_t<tt::CBIndex> {
            SRC = tt::CBIndex::c_0,
            DST = tt::CBIndex::c_1,
            ACC = tt::CBIndex::c_2
        };

        static constexpr std::array<const char*, 3> KERNEL_PATHS{
            "ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/dataflow/"
            "accumulation_reader.cpp",
            "ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_compute.cpp",
            "ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/dataflow/"
            "accumulation_writer.cpp"};

        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static uint32_t calc_input_tile_offset(
            const Shape& input_shape, const int32_t& dim, uint32_t tile_height = 32, uint32_t tile_width = 32);
    };

    using program_factory_t = std::variant<AccumulationProgramFactory>;

    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

ttnn::Tensor accumulation(
    const Tensor& input_tensor,
    const int32_t& dim,
    const std::optional<DataType>& dtype,
    const bool& reverse_order,
    std::optional<Tensor> optional_out,
    const std::optional<MemoryConfig>& memory_config,
    AccumulationOp op);

}  // namespace ttnn::prim
