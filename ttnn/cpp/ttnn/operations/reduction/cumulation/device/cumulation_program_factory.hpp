// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cumulation_device_operation_types.hpp"

#include <optional>
#include <type_traits>
#include <variant>

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::reduction::cumulation {

using namespace tt::tt_metal;
using namespace tt::stl;

struct CumulationProgramFactory {
    enum class CumulationCB : std::underlying_type_t<tt::CBIndex> {
        SRC = tt::CBIndex::c_0,
        DST = tt::CBIndex::c_1,
        START = tt::CBIndex::c_2,
        ACC = tt::CBIndex::c_3
    };

    static constexpr std::array<const char*, 3> KERNEL_PATHS{
        "ttnn/cpp/ttnn/operations/reduction/cumulation/device/kernels/dataflow/"
        "cumulation_reader.cpp",
        "ttnn/cpp/ttnn/operations/reduction/cumulation/device/kernels/compute/cumulation_compute.cpp",
        "ttnn/cpp/ttnn/operations/reduction/cumulation/device/kernels/dataflow/"
        "cumulation_writer.cpp"};
    struct shared_variables_t {
        KernelHandle cumulation_reader_kernel_id;
        KernelHandle cumulation_compute_kernel_id;
        std::optional<KernelHandle> cumulation_compute_kernel_id_2;
        KernelHandle cumulation_writer_kernel_id;
        std::vector<CoreCoord> cores;
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
        const CumulationCB& cumulation_cb,
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

}  // namespace ttnn::operations::reduction::cumulation
