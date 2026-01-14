// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "accumulation_device_operation_types.hpp"

#include <optional>
#include <type_traits>
#include <variant>

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::reduction::accumulation {

using namespace tt::tt_metal;
using namespace tt::stl;

struct AccumulationProgramFactory {
    enum class AccumulationCB : std::underlying_type_t<tt::CBIndex> {
        SRC = tt::CBIndex::c_0,
        DST = tt::CBIndex::c_1,
        START = tt::CBIndex::c_2,
        ACC = tt::CBIndex::c_3
    };

    static constexpr std::array<const char*, 3> KERNEL_PATHS{
        "ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/dataflow/"
        "accumulation_reader.cpp",
        "ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_compute.cpp",
        "ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/dataflow/"
        "accumulation_writer.cpp"};
    struct shared_variables_t {
        KernelHandle accumulation_reader_kernel_id{};
        KernelHandle accumulation_compute_kernel_id{};
        std::optional<KernelHandle> accumulation_compute_kernel_id_2;
        KernelHandle accumulation_writer_kernel_id{};
        std::vector<CoreCoord> cores;
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

    static CBHandle create_cb(
        Program& program,
        const DataType& dtype,
        const AccumulationCB& accumulation_cb,
        const CoreRangeSet& core_range_set,
        const uint32_t& num_tiles);

    static KernelHandle create_kernel(
        Program& program,
        const char* kernel_path,
        const CoreRangeSet& core_range_set,
        const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
        const std::vector<uint32_t>& runtime_args = {});

    static uint32_t calc_input_tile_offset(const Shape& input_shape, const int32_t& dim);
};

}  // namespace ttnn::operations::reduction::accumulation
