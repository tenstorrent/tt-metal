// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scatter_device_operation_types.hpp"
#include "tt-metalium/allocator.hpp"
#include "tt-metalium/device.hpp"
#include "common/common.hpp"  // Data movement common utilities

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;

enum class ScatterCB : std::underlying_type_t<tt::CBIndex> {
    INPUT = CBIndex::c_0,
    SRC = CBIndex::c_1,
    INDEX = CBIndex::c_2,
    DST = CBIndex::c_3,
    FP32_TEMP = CBIndex::c_4,
};

constexpr uint32_t BIT_MASK_32 = 32 - 1;

inline uint64_t ceil32(const uint64_t& number) {
    return ((number & BIT_MASK_32) == 0) ? number : ((number | BIT_MASK_32) + 1);
}

// maximal input/index/source/output chunk size, divisible by 32, calculated as follows:
// BH available L1 mem size of nearly 1.5 MB...
// ... divided by 4 to be able to allocate four equally long row chunks (coming from input/index/source/output
// tensors)
// ... divided by 4 to account for 4-byte datum sizes of each tensor (fp32, int32)
// ... minimized by ~10% to account for reserved memory
inline uint32_t calculate_optimal_chunk_size(const Tensor& input_tensor) {
    uint32_t l1_per_chunk = (ttnn::operations::data_movement::get_max_l1_space(input_tensor) / 4) / 4;
    return ceil32((l1_per_chunk * 9 / 10) - 32);
}

inline CBHandle create_cb(
    Program& program,
    const DataType& dtype,
    const ScatterCB& scatter_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& page_size_bytes) {
    const uint32_t cb_id{static_cast<uint32_t>(scatter_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const auto cb_config{
        CircularBufferConfig{page_size_bytes, {{cb_id, cb_data_format}}}.set_page_size(cb_id, page_size_bytes)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

inline KernelHandle create_kernel(
    Program& program,
    const char* kernel_path,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::vector<uint32_t>& runtime_args = {}) {
    auto kernel_id{CreateKernel(program, kernel_path, core_range_set, config)};

    if (!runtime_args.empty()) {
        SetRuntimeArgs(program, kernel_id, core_range_set, runtime_args);
    }

    return kernel_id;
}

}  // namespace ttnn::prim
