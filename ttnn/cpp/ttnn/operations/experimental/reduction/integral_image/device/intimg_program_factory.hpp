// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "intimg_work_split.hpp"
#include "intimg_device_operation_types.hpp"

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

using namespace intimg::common;

struct IntImgProgramFactory {
    static constexpr std::array<const char*, 3> KERNEL_PATHS{
        "ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/device/kernels/"
        "intimg_reader.cpp",
        "ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/device/kernels/intimg_compute.cpp",
        "ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/device/kernels/"
        "intimg_writer.cpp"};
    struct shared_variables_t {
        KernelHandle reader_kernel_id{};
        KernelHandle compute_kernel_id{};
        KernelHandle writer_kernel_id{};
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
        const IntImgCB& intimg_cb,
        const CoreRangeSet& core_range_set,
        const uint32_t& tiles_num);

    static KernelHandle create_kernel(
        Program& program,
        const char* kernel_path,
        const CoreRangeSet& core_range_set,
        const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
        const std::vector<uint32_t>& runtime_args = {});

    static void set_runtime_args(
        Program& program,
        KernelHandle reader_kernel_id,
        KernelHandle compute_kernel_id,
        KernelHandle writer_kernel_id,
        const IntImgPerCoreSetWorkSplit& per_core_set_work_split,
        uint32_t input_buffer_address,
        uint32_t zero_tile_buffer_address,
        uint32_t output_buffer_address);
};

}  // namespace ttnn::operations::experimental::reduction
