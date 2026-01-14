// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

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
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::reduction
